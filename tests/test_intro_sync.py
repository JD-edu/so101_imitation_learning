"""Verify the chapter 1–4 source contracts without camera, GUI, downloads or training files."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
BASIC = ROOT / '01_02_pytorch_basic'
MUJOCO = ROOT / '03_mujoco_basic'

def tree(path):
    return ast.parse(path.read_text())

def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])), '<selected source>', 'exec'), scope)

def definition(path, name, scope):
    node = next(n for n in ast.walk(tree(path)) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    execute([node], scope)
    return scope[name]

class IntroSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_python_syntax(self):
        paths = list(BASIC.glob('*.py')) + list(MUJOCO.rglob('*.py'))
        self.assertEqual(len(paths), 25)
        for path in paths:
            tree(path)

    def test_mlp_cnn_minivgg_training_inference_parity(self):
        cases = [('101_pytorch_mnist', '101_pytorch_mnist_inference', 'MultiLayerPerceptron', (2,1,28,28), 10),
                 ('102_pytorch_CNN_cifar10','102_pytorch_CNN_cifar10_inference','ImprovedCNN',(2,3,32,32),10),
                 ('103_pytorch_miniVGGnet','103_pytorch_miniVGGnet_inference','MiniVGGNet',(2,3,64,64),3)]
        for train, infer, name, shape, classes in cases:
            with self.subTest(model=name):
                scope = {'torch':torch, 'nn':nn, 'F':F}
                cls = definition(BASIC/(train+'.py'),name,scope.copy())
                inf = definition(BASIC/(infer+'.py'),name,scope.copy())
                model = cls(); deployed = inf()
                x=torch.randn(shape)
                out=model(x)
                loss=F.cross_entropy(out,torch.arange(2)%classes)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                deployed.load_state_dict(model.state_dict(),strict=True)
                model.eval();deployed.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(x),deployed(x))
                self.assertEqual(out.shape,(2,classes))

    def test_transfer_inference_loaders(self):
        for filename, constructor, fn, last in [
            ('104_pytorch_VGGnet_transfer_feature_inference.py',models.vgg16,'load_vgg16_model','classifier'),
            ('105_pytorch_Resnet_transfer_fine_inference.py',models.resnet18,'load_resnet18_model','fc')]:
            with self.subTest(file=filename):
                trained=constructor(weights=None)
                if last=='classifier':trained.classifier[6]=nn.Linear(trained.classifier[6].in_features,3)
                else:trained.fc=nn.Linear(trained.fc.in_features,3)
                scope={'models':models,'nn':nn,'torch':torch,'num_classes':3,'device':torch.device('cpu')}
                loader=definition(BASIC/filename,fn,scope)
                with patch.object(torch,'load',return_value=trained.state_dict()), contextlib.redirect_stdout(io.StringIO()):
                    deployed=loader('in-memory-checkpoint')
                trained.eval()
                x=torch.randn(1,3,224,224)
                with torch.no_grad():torch.testing.assert_close(trained(x),deployed(x))
                del trained,deployed

    def test_resnet_validation_split_and_transform(self):
        path=BASIC/'105_pytorch_Resnet_transfer_fine.py'
        nodes=[]
        for node in tree(path).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='model' for t in node.targets):break
            if not isinstance(node,(ast.Import,ast.ImportFrom)):
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DATA_PATH' for t in node.targets):continue
                nodes.append(node)
        with tempfile.TemporaryDirectory() as directory:
            for cls in ['a','b']:
                d=Path(directory)/cls;d.mkdir()
                for i in range(5):Image.fromarray(np.random.default_rng(i).integers(0,256,(40,40,3),dtype=np.uint8)).save(d/f'{i}.png')
            scope={'torch':torch,'transforms':transforms,'datasets':datasets,'DataLoader':DataLoader,'random_split':random_split,'Subset':Subset,'DATA_PATH':directory}
            execute(nodes,scope)
            train=scope['train_dataset']; val=scope['val_dataset']
            self.assertEqual((len(train),len(val)),(8,2))
            self.assertFalse(set(train.indices)&set(val.indices))
            self.assertEqual(train.dataset.samples,val.dataset.samples)
            self.assertFalse(any(isinstance(t,transforms.RandomHorizontalFlip) for t in val.dataset.transform.transforms))
            torch.testing.assert_close(val[0][0],val[0][0])
            self.assertEqual(val[0][0].shape,(3,224,224))

    def test_resnet_feature_extraction_freezes_bn_but_finetuning_updates(self):
        model=models.resnet18(weights=None);model.fc=nn.Linear(model.fc.in_features,3)
        for p in model.parameters():p.requires_grad=False
        for p in model.fc.parameters():p.requires_grad=True
        batch=(torch.randn(2,3,64,64),torch.tensor([0,1]))
        scope={'model':model,'nn':nn,'torch':torch,'device':torch.device('cpu'),'train_loader':[batch],'val_loader':[batch],'criterion':nn.CrossEntropyLoss(),'optimizer':torch.optim.Adam(model.fc.parameters(),lr=.001)}
        train=definition(BASIC/'105_pytorch_Resnet_transfer_fine.py','train_model',scope)
        before=model.bn1.running_mean.clone()
        with contextlib.redirect_stdout(io.StringIO()):train(1,'Feature-Ex')
        torch.testing.assert_close(before,model.bn1.running_mean)
        for p in model.parameters():p.requires_grad=True
        scope['optimizer']=torch.optim.Adam(model.parameters(),lr=.0001)
        with contextlib.redirect_stdout(io.StringIO()):train(1,'Fine-Tuning')
        self.assertFalse(torch.equal(before,model.bn1.running_mean))

    def test_resnet_summary_preserves_pretrained_bn_statistics(self):
        from torchsummary import summary
        model=models.resnet18(weights=None)
        before=model.bn1.running_mean.clone()
        nodes=tree(BASIC/'105_pytorch_Resnet_transfer_fine.py').body
        index=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Expr)
                   and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                   and n.value.func.id=='summary')
        scope={'model':model,'summary':summary,'device':torch.device('cpu')}
        with contextlib.redirect_stdout(io.StringIO()):
            execute(nodes[index-1:index+2],scope)
        torch.testing.assert_close(before,model.bn1.running_mean)
        self.assertTrue(model.training)

    def test_imu_sensor_addresses(self):
        model=mujoco.MjModel.from_xml_path(str(MUJOCO/'106_mujoco_imu_sensor/scene.xml'))
        data=mujoco.MjData(model);data.sensordata[:]=np.arange(model.nsensordata)+10
        t=tree(MUJOCO/'106_mujoco_imu_sensor/106_mujoco_imu_sensor.py')
        block=next(n for n in ast.walk(t) if isinstance(n,ast.Try) and any(isinstance(s,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='gyro_id' for a in s.targets) for s in n.body))
        scope={'model':model,'data':data,'np':np,'yaw':0.,'dt':model.opt.timestep}
        execute(block.body,scope)
        np.testing.assert_array_equal(scope['acc_data'],[10,11,12])
        np.testing.assert_array_equal(scope['gyro_data'],[13,14,15])
        self.assertAlmostEqual(scope['yaw'],15*model.opt.timestep)

    def test_position_commands_and_physics(self):
        for folder in ['107_jdcobot100_MUJOCO_load','109_so101_mujoco_load']:
            path=MUJOCO/folder/'answer/robot_move.py';t=tree(path)
            scope={'np':np}
            execute([n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id in ['pose_A','pose_B','switch_interval'] for a in n.targets)],scope)
            model=mujoco.MjModel.from_xml_path(str(path.with_name('scene.xml')));data=mujoco.MjData(model)
            scope.update(model=model,data=data)
            assignment=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(a,ast.Subscript) and isinstance(a.value,ast.Attribute) and a.value.attr=='ctrl' for a in n.targets))
            for i in range(2200):
                target=scope['pose_A'] if (data.time//scope['switch_interval'])%2==0 else scope['pose_B']
                scope['target_qpos']=target
                execute([assignment],scope)
                np.testing.assert_allclose(data.ctrl,target)
                mujoco.mj_step(model,data)
                self.assertTrue(np.isfinite(data.qpos).all())
                self.assertTrue(np.isfinite(data.qvel).all())

    def test_urdf_converters_preserve_control_xml(self):
        folders=['107_jdcobot100_MUJOCO_load','108_puppypi_mujoco_load','109_so101_mujoco_load']
        outputs=['jdcobot100_converted.xml','puppy2_converted.xml','so101_converted.xml']
        for folder,expected in zip(folders,outputs):
            path=MUJOCO/folder/'answer/urdf_to_mjcf.py';t=tree(path)
            urdf=next(n.value.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='urdf_filename' for a in n.targets))
            call=next(n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='mj_saveLastXML')
            self.assertEqual(call.args[0].value,expected)
            model=mujoco.MjModel.from_xml_path(str(path.with_name(urdf)))
            with tempfile.TemporaryDirectory() as d:
                mujoco.mj_saveLastXML(str(Path(d)/expected),model)
                self.assertTrue((Path(d)/expected).is_file())
            self.assertEqual(model.nu,0)

if __name__=='__main__':unittest.main(verbosity=2)
