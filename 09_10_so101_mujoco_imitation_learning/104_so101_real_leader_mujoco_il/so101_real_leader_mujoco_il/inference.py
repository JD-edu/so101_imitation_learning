from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import TASK
from .aruco_reference import DEFAULT_MARKER_ID, ArucoReferenceDetector
from .runtime import DEFAULT_XML, MujocoFollower
from .viewer import GLFWFollowerViewer


def _validate_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.exists():
        available = []
        if checkpoint.parent.is_dir():
            available = sorted(
                item.name
                for item in checkpoint.parent.iterdir()
                if item.is_dir() and (item / "config.json").is_file()
            )
        hint = f" Available local checkpoints: {', '.join(available)}." if available else ""
        raise FileNotFoundError(
            f"ACT checkpoint directory does not exist: {checkpoint}.{hint} "
            "Run the train command first or pass an existing --checkpoint directory."
        )
    if not checkpoint.is_dir():
        raise NotADirectoryError(f"ACT checkpoint must be a directory: {checkpoint}")
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"ACT checkpoint is incomplete: {checkpoint}; missing files: {missing}"
        )
    return checkpoint


def run_inference(
    checkpoint: str | Path,
    xml_path: str | Path = DEFAULT_XML,
    device: str | None = None,
    seed: int = 1000,
    aruco_id: int = DEFAULT_MARKER_ID,
    assist_view: bool = False,
) -> None:
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.control_utils import predict_action
    except ImportError as exc:
        raise RuntimeError(f"Inference requires torch and lerobot: {exc}") from exc

    checkpoint = _validate_checkpoint(checkpoint)
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or restore CUDA access")
    device_obj = torch.device(requested_device)
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(policy_config, ACTConfig):
        raise TypeError(
            f"checkpoint is {type(policy_config).__name__}, not an ACT policy: {checkpoint}"
        )
    policy_config.device = str(device_obj)
    policy = ACTPolicy.from_pretrained(
        checkpoint, config=policy_config, local_files_only=True
    ).to(device_obj).eval()
    aruco_feature_key = "observation.environment_state"
    uses_aruco_state = aruco_feature_key in policy_config.input_features
    if uses_aruco_state and tuple(policy_config.input_features[aruco_feature_key].shape) != (6,):
        raise ValueError(
            "this pipeline expects observation.environment_state shape (6,) for "
            "the ArUco reference"
        )
    aruco_detector = ArucoReferenceDetector(marker_id=aruco_id) if uses_aruco_state else None
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )
    follower = MujocoFollower(xml_path=xml_path, seed=seed)
    viewer = GLFWFollowerViewer(
        follower.model,
        follower.data,
        "SO-101 LeRobot ACT inference",
        assist_view=assist_view,
    )
    if hasattr(policy, "reset"):
        policy.reset()
    print("ACT inference: r resets the scene, q quits")
    try:
        with torch.inference_mode():
            step = 0
            while viewer.alive:
                start = time.monotonic()
                image = viewer.render_and_capture()
                observation = {
                    # predict_action prepares NumPy HWC images and vectors for
                    # inference; dataset frames, by contrast, are torch CHW.
                    "observation.images.overview": image.copy(),
                    "observation.state": follower.state,
                }
                aruco_detection = (
                    aruco_detector.detect(image) if aruco_detector is not None else None
                )
                if aruco_detection is not None:
                    observation[aruco_feature_key] = aruco_detection.feature()
                action = predict_action(
                    observation=observation,
                    policy=policy,
                    device=device_obj,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=(
                        device_obj.type == "cuda"
                        and bool(getattr(policy.config, "use_amp", False))
                    ),
                    task=TASK,
                    robot_type="so101_real_leader_mujoco_follower",
                ).squeeze(0).detach().cpu().numpy()
                if not np.isfinite(action).all():
                    raise RuntimeError(
                        f"ACT produced NaN/Inf at inference step={step}. "
                        "The action was not sent to MuJoCo. Retry with --device cpu "
                        "and verify that the checkpoint matches its dataset."
                    )
                follower.apply_target(np.asarray(action, dtype=np.float64))
                follower.step_for(1.0 / 30.0)
                viewer.present()

                preview_rgb = (
                    aruco_detector.draw_preview(image, aruco_detection)
                    if aruco_detector is not None and aruco_detection is not None
                    else image
                )
                preview = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("ACT observation", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    follower.reset()
                    if hasattr(policy, "reset"):
                        policy.reset()
                elif key == ord("q"):
                    break
                delay = 1.0 / 30.0 - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
                step += 1
    finally:
        viewer.close()
        cv2.destroyAllWindows()
