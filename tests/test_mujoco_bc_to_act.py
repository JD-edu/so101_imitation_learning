"""CPU regression tests. No viewer, network, or existing checkpoint writes."""
import ast
import importlib.util
import io
import os
from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BC_ROOT = REPO / '05_mujoco_bc'
ROOT = REPO / '06_mujoco_act'
TRAIN = ['201_MLP_MUJOCO.py', '202_RNN_MUJOCO.py', '203_Transformer_MUJOCO.py',
         '204_Action_Chunking_MUJOCO.py', '205_Action_Various_start.py',
         '206_ACT_RH_TE.py', '207_ACT_CAVE.py', '208_ACT_Image.py']
INFER = ['201_MLP_MUJOCO_inference.py', '202_RNN_MUJOCO_inference.py',
         '203_Transformer_MUJOCO_infer.py', '204_Action_Chunking_MUJOCO_infer.py',
         '205_Action_Various_start_infer.py', '206_ACT_RH_TE_infer.py',
         '207_ACT_CVAE_infer.py', '208_ACT_Image_infer.py']
CLASSES = ['BehaviorCloningMLP', 'VanillaRNN_BC', 'TransformerEncoderBC',
           'ActionChunkTransformer', 'ActionChunkTransformer', 'ActionChunkTransformer',
           'GoalCVAE_ACT', 'VisualACT']

def load(name):
    folder = BC_ROOT if name[:3] in ['201', '202', '203'] else ROOT
    spec = importlib.util.spec_from_file_location(Path(name).stem, folder / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class BCToACTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.train = [load(name) for name in TRAIN]
        cls.infer = [load(name) for name in INFER]
        cls.mj = mujoco.MjModel.from_xml_path(str(ROOT / 'scene.xml'))

    def test_syntax_and_scene_contract(self):
        for path in list(ROOT.glob('*.py')) + list(BC_ROOT.glob('*.py')):
            ast.parse(path.read_text(), filename=str(path))
        self.assertEqual((self.mj.nq, self.mj.nv, self.mj.nu), (6, 6, 6))
        np.testing.assert_array_equal(self.mj.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE] * 6)
        self.assertGreaterEqual(mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_CAMERA, 'front_cam'), 0)

    def test_checkpoint_and_scene_paths(self):
        paths = set()
        for train, infer in zip(self.train, self.infer):
            self.assertEqual(train.MODEL_PATH, infer.MODEL_PATH)
            folder = Path(train.__file__).parent
            self.assertEqual(Path(train.XML_PATH), folder / 'scene.xml')
            self.assertEqual(train.XML_PATH, infer.XML_PATH)
            self.assertEqual(Path(train.MODEL_PATH).parent, folder)
            self.assertTrue(train.MODEL_PATH.endswith('_v2.pt'))
            paths.add(train.MODEL_PATH)
        self.assertEqual(len(paths), 8)

    def test_chapter4_models_match_chapter5(self):
        path = REPO / '04_sequence_models' / 'sequence_models.py'
        spec = importlib.util.spec_from_file_location('chapter4_models', path)
        chapter4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(chapter4)
        for module, name in zip(self.train[1:3], CLASSES[1:3]):
            reference = getattr(module, name)().eval()
            introductory = getattr(chapter4, name)().eval()
            introductory.load_state_dict(reference.state_dict(), strict=True)
            with torch.no_grad():
                x = torch.randn(2, 10, 6)
                torch.testing.assert_close(introductory(x), reference(x))
        bc_model = mujoco.MjModel.from_xml_path(str(BC_ROOT / 'scene.xml'))
        self.assertEqual((bc_model.nq, bc_model.nv, bc_model.nu), (6, 6, 6))

    def test_all_window_labels_and_episode_boundaries(self):
        actions = torch.arange(2 * 100 * 6).reshape(2, 100, 6).float()
        for index, module in enumerate(self.train[1:7], 1):
            states = actions + 10000
            if index == 6:
                states = torch.cat([states, states + 10000], dim=-1)
            cls = getattr(module, 'P2PTrajectoryDataset', None) or getattr(module, 'ActionChunkDataset', None) or module.GoalActionChunkDataset
            dataset = cls(states, actions)
            count = 91 if index < 3 else 62
            self.assertEqual(len(dataset), 2 * count)
            for episode in range(2):
                for offset in [0, count - 1]:
                    t = offset + 9
                    x, y = dataset[episode * count + offset]
                    torch.testing.assert_close(x, states[episode, t-9:t+1])
                    target = actions[episode, t] if index < 3 else actions[episode, t:t+30]
                    torch.testing.assert_close(y, target)
            with self.assertRaises(ValueError):
                cls(states[:, :5], actions[:, :5])
        imgs = torch.zeros(2, 100, 3, 8, 8, dtype=torch.uint8)
        dataset = self.train[7].VisualACTDataset(imgs, actions, actions)
        self.assertEqual(len(dataset), 142)
        for ep in range(2):
            for t in [0, 70]:
                img, q, chunk = dataset[ep*71+t]
                self.assertEqual(img.dtype, torch.uint8)
                torch.testing.assert_close(q, actions[ep, t])
                torch.testing.assert_close(chunk, actions[ep, t:t+30])

    def test_mlp_collection_alignment(self):
        module = self.train[0]
        def step(model, data):
            # Encode the action in the subsequent state so a one-step shift is observable.
            data.qpos[:] = data.ctrl[:]
        with patch.object(module, 'NUM_EPISODES', 2), patch.object(module.mujoco, 'mj_step', side_effect=step):
            states, actions = module.generate_p2p_episodes(self.mj, mujoco.MjData(self.mj))
        self.assertEqual(states.shape, (200, 6))
        for start in [0, 100]:
            torch.testing.assert_close(states[start+1:start+100], actions[start:start+99])

    def test_actual_mujoco_collection(self):
        for index, module in enumerate(self.train[:7]):
            generator = next(getattr(module, name) for name in vars(module) if name.startswith('generate_'))
            with patch.object(module, 'NUM_EPISODES', 2):
                obs, actions = generator(self.mj, mujoco.MjData(self.mj))
            self.assertTrue(torch.isfinite(obs).all())
            self.assertTrue(torch.isfinite(actions).all())
            if index == 0:
                self.assertEqual(obs.shape, (200, 6))
            else:
                self.assertEqual(obs.shape, (2, 100, 12 if index == 6 else 6))
                if index == 6:
                    torch.testing.assert_close(obs[:, -1, 6:], actions[:, -1])
                else:
                    goal = torch.tensor([.7, .5, -.4, .6, -.5, .035])
                    self.assertLessEqual(float((actions[:, -1] - goal).abs().max()), .002501)

    def test_train_backward_checkpoint_roundtrip_and_inference(self):
        for index, (train, infer, name) in enumerate(zip(self.train, self.infer, CLASSES)):
            with self.subTest(stage=201+index):
                kwargs = {'pretrained_backbone': False} if index == 7 else {}
                model = getattr(train, name)(**kwargs)
                deployed = getattr(infer, name)(**kwargs)
                x = torch.randn(2, 6) if index == 0 else torch.randn(2, 10, 12 if index == 6 else 6)
                args = (torch.randint(256, (2, 3, 128, 128), dtype=torch.uint8), torch.randn(2, 6)) if index == 7 else (x,)
                target = torch.randn(2, 6) if index < 3 else torch.randn(2, 30, 6)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                if index >= 6:
                    pred, mu, logvar = model(*args, target)
                    loss, _, _ = train.compute_loss(pred, target, mu, logvar)
                else:
                    pred = model(*args)
                    loss = torch.nn.functional.mse_loss(pred, target)
                self.assertEqual(pred.shape, target.shape)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                checkpoint = io.BytesIO()
                torch.save(model.state_dict(), checkpoint)
                checkpoint.seek(0)
                deployed.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
                model.eval(); deployed.eval()
                with torch.no_grad():
                    expected = model(*args)
                    actual = deployed(*args)
                    repeated = deployed(*args)
                if index >= 6:
                    self.assertIsNone(actual[1]); self.assertIsNone(actual[2])
                    expected, actual, repeated = expected[0], actual[0], repeated[0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual, repeated)

    def test_visual_uint8_float_preprocessing_equivalence(self):
        model = self.train[7].VisualACT(pretrained_backbone=False).eval()
        rgb = torch.randint(256, (2, 3, 128, 128), dtype=torch.uint8)
        joints = torch.zeros(2, 6)
        seen = []
        handle = model.vision_backbone.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        try:
            with torch.no_grad():
                torch.testing.assert_close(model(rgb, joints)[0], model(rgb.float()/255, joints)[0])
            mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
            std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
            torch.testing.assert_close(seen[0], (rgb.float()/255 - mean) / std)
        finally:
            handle.remove()

    def test_goal_marker_uses_graspframe(self):
        q = np.array([.7, .5, -.4, .6, -.5, .035])
        data = mujoco.MjData(self.mj); data.qpos[:] = q
        mujoco.mj_forward(self.mj, data)
        site = mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_SITE, 'graspframe')
        for module in [self.infer[5], self.train[6], self.infer[6], self.infer[7]]:
            np.testing.assert_allclose(module.get_target_3d_position(self.mj, q), data.site_xpos[site])

    def test_inference_temporal_weights_goal_history_and_reset(self):
        for index in [5, 6, 7]:
            module = self.infer[index]
            inputs, controls = [], []
            policy = MagicMock()
            def predict(*args):
                call = len(inputs)
                inputs.append(tuple(arg.clone() for arg in args))
                chunk = (torch.arange(30).float() * .00001 + call * .0001).view(1, 30, 1).expand(1, 30, 6)
                return (chunk, None, None) if index >= 6 else chunk
            policy.side_effect = predict
            fake_glfw = MagicMock()
            fake_glfw.window_should_close.side_effect = [False] * 122 + [True]
            fake_glfw.get_framebuffer_size.return_value = (1200, 900)
            def step(model, data):
                controls.append(data.ctrl.copy())
                data.qpos[:] = data.ctrl
                mujoco.mj_forward(model, data)
            with ExitStack() as stack:
                stack.enter_context(patch.object(module, 'glfw', fake_glfw))
                stack.enter_context(patch.object(module, CLASSES[index], return_value=policy))
                stack.enter_context(patch.object(torch, 'load', return_value={}))
                stack.enter_context(patch.object(mujoco, 'MjrContext', return_value=object()))
                stack.enter_context(patch.object(mujoco, 'mjr_render'))
                stack.enter_context(patch.object(mujoco, 'mj_step', side_effect=step))
                stack.enter_context(patch.object(module.time, 'sleep'))
                if index == 7:
                    stack.enter_context(patch.object(module, 'capture_foreground_camera', return_value=torch.zeros(3, 128, 128, dtype=torch.uint8)))
                module.main()
            self.assertEqual(len(inputs), 122)
            np.testing.assert_allclose(controls[0], 0, atol=1e-8)
            expected = (.00001 * np.exp(-.05) + .0001) / (np.exp(-.05) + 1)
            np.testing.assert_allclose(controls[1], expected, rtol=1e-6)
            # A reset must discard ALL predictions from the previous episode.
            np.testing.assert_allclose(controls[120], .012, rtol=1e-6)
            if index == 6:
                self.assertEqual(inputs[0][0].shape, (1, 10, 12))
                torch.testing.assert_close(inputs[0][0][..., 6:], inputs[119][0][..., 6:])
                self.assertFalse(torch.equal(inputs[0][0][..., 6:], inputs[120][0][..., 6:]))
                torch.testing.assert_close(inputs[1][0][0, -1, :6], torch.zeros(6))

    def test_206_training_viewer_replans_and_clears_ensemble(self):
        module = self.train[5]
        predictions, controls = [], []
        def predict(x):
            value = len(predictions) * .0001
            predictions.append(x.clone())
            return torch.full((1, 30, 6), value)
        def step(model, data):
            controls.append(data.ctrl.copy())
        fake_glfw = MagicMock()
        fake_glfw.window_should_close.side_effect = [False] * 121 + [True]
        fake_glfw.get_framebuffer_size.return_value = (1200, 900)
        with patch.object(module, 'glfw', fake_glfw), patch.object(mujoco, 'MjrContext', return_value=object()), patch.object(mujoco, 'mjr_render'), patch.object(mujoco, 'mj_step', side_effect=step), patch.object(module.time, 'sleep'):
            module.run_glfw_eval(self.mj, mujoco.MjData(self.mj), predict)
        self.assertEqual(len(predictions), 121)
        np.testing.assert_allclose(controls[1], .0001 / (1 + np.exp(-.05)), rtol=1e-6)
        np.testing.assert_allclose(controls[120], .012, rtol=1e-6)

    @unittest.skipUnless(os.environ.get('MUJOCO_GL') == 'egl', 'Run with MUJOCO_GL=egl for real RGB rendering')
    def test_visual_collection_with_real_egl_rendering(self):
        module = self.train[7]
        context = mujoco.GLContext(128, 128)
        context.make_current()
        con = None
        try:
            scene = mujoco.MjvScene(self.mj, maxgeom=1000)
            con = mujoco.MjrContext(self.mj, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
            with patch.object(module, 'NUM_EPISODES', 2), patch.object(module, 'STEPS_PER_EP', 32):
                images, joints, actions = module.generate_visual_p2p_dataset(self.mj, mujoco.MjData(self.mj), scene, con)
            self.assertEqual(images.shape, (2, 32, 3, 128, 128))
            self.assertEqual(images.dtype, torch.uint8)
            self.assertGreater(int(images.max()), int(images.min()))
            goal = torch.tensor([.7, .5, -.4, .6, -.5, .035]).expand(2, 6)
            torch.testing.assert_close(actions[:, -1], goal)
            self.assertEqual(len(module.VisualACTDataset(images, joints, actions)), 6)
            self.assertTrue(torch.isfinite(joints).all())
        finally:
            if con is not None:
                con.free()
            context.free()

if __name__ == '__main__':
    unittest.main(verbosity=2)
