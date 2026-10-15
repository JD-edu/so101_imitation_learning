from __future__ import annotations

from pathlib import Path


def train_act(
    dataset_root: str | Path,
    repo_id: str,
    output_dir: str | Path,
    steps: int = 100_000,
    batch_size: int = 32,
    device: str | None = None,
    log_freq: int = 100,
) -> None:
    try:
        import torch
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError as exc:
        raise RuntimeError("Training requires torch and lerobot; install requirements.txt") from exc

    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device. "
            "Run with '--device cpu' for a smoke test, or restore the NVIDIA "
            "driver/device before CUDA training. Check `nvidia-smi` and "
            "`python -c \"import torch; print(torch.cuda.is_available())\"`."
        )
    device_obj = torch.device(requested_device)
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    features = dataset_to_policy_features(metadata.features)
    outputs = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    inputs = {k: v for k, v in features.items() if k not in outputs}
    config = ACTConfig(
        input_features=inputs, output_features=outputs, device=str(device_obj)
    )
    policy = ACTPolicy(config).to(device_obj).train()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=metadata.stats)

    def deltas(indices):
        return [0.0] if indices is None else [i / metadata.fps for i in indices]

    # ACT needs a future action chunk, but this version expects the current
    # camera frame as BCHW. Adding an image delta of [0] would introduce an
    # extra singleton time axis (BTCHW) and break the ResNet backbone.
    delta_timestamps = {"action": deltas(config.action_delta_indices)}
    dataset = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=delta_timestamps)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device_obj.type == "cuda",
    )
    optimizer = config.get_optimizer_preset().build(policy.parameters())
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = preprocessor(batch)
        loss, info = policy.forward(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % log_freq == 0:
            print(f"step {step:07d}/{steps:07d} loss={loss.item():.6f}")
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(f"checkpoint saved to {output_dir}")
