from __future__ import annotations

import time
from pathlib import Path

import numpy as np


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
    episodes: int = 10,
    seed: int = 1000,
    headless: bool = True,
    device: str | None = None,
    max_steps: int = 240,
) -> None:
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.control_utils import predict_action
    except ImportError as exc:
        raise RuntimeError("Inference requires torch and lerobot; install requirements.txt") from exc

    from .env import SO101PickPlaceEnv

    checkpoint = _validate_checkpoint(checkpoint)
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device. "
            "Use '--device cpu' or restore the NVIDIA driver/device."
        )
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
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device_obj)}},
    )
    env = SO101PickPlaceEnv(seed=seed)
    viewer = None
    if not headless:
        from .viewer import GLFWViewer

        viewer = GLFWViewer(env.model, env.data, "SO-101 LeRobot ACT inference")
    successes = 0
    try:
        with torch.inference_mode():
            for episode in range(episodes):
                env.reset()
                if hasattr(policy, "reset"):
                    policy.reset()
                for step in range(max_steps):
                    obs = env.observation(render=viewer is None)
                    if viewer is not None:
                        if not viewer.alive:
                            return
                        obs["image"] = viewer.capture()
                    observation = {
                        "observation.images.overview": obs["image"],
                        "observation.state": obs["state"],
                    }
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
                        task="Pick up the red block and place it on the green target.",
                        robot_type="so101_mujoco",
                    ).squeeze(0).detach().cpu().numpy()
                    if not np.isfinite(action).all():
                        raise RuntimeError(
                            f"ACT produced NaN/Inf at episode={episode + 1}, step={step}. "
                            "The action was not sent to MuJoCo. Retry with --device cpu "
                            "and verify that the checkpoint matches its dataset."
                        )
                    env.step(action)
                    if viewer is not None:
                        viewer.render()
                        time.sleep(1 / 30)
                error = float(np.linalg.norm(env.block_position[:2] - env.workspace.target_xy))
                success = error < 0.045
                successes += int(success)
                print(f"episode {episode + 1}: success={success} target_error={error:.3f} m")
    finally:
        env.close()
        if viewer is not None:
            viewer.close()
    print(f"success rate: {successes}/{episodes} ({successes / episodes:.1%})")
