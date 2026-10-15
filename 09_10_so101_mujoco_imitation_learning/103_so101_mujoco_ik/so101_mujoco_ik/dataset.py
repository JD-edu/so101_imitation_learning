from __future__ import annotations

from pathlib import Path

from .expert import IKPickPlaceExpert


TASK = "Pick up the red block and place it on the green target."


def create_dataset(
    root: str | Path,
    repo_id: str = "local/so101_mujoco_pick_place",
    episodes: int = 100,
    seed: int = 0,
    fps: int = 30,
    show_ui: bool = False,
) -> None:
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset export requires torch and a compatible lerobot installation. "
            f"Original import error: {exc}"
        ) from exc

    from .env import SO101PickPlaceEnv

    root = Path(root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"dataset directory is not empty: {root}")
    features = {
        "observation.images.overview": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["joints"],
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["joints"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id, root=root, fps=fps, features=features,
        robot_type="so101_mujoco", use_videos=False,
    )
    env = SO101PickPlaceEnv(seed=seed)
    expert = IKPickPlaceExpert(env, fps=fps)
    viewer = None
    if show_ui:
        from .viewer import GLFWViewer

        viewer = GLFWViewer(env.model, env.data, "SO-101 synthetic data generation")
    try:
        for episode in range(episodes):
            frames = expert.generate(
                capture=True,
                callback=viewer.render if viewer is not None else None,
                image_provider=viewer.capture if viewer is not None else None,
            )
            for frame in frames:
                dataset.add_frame(
                    {
                        "observation.images.overview": torch.from_numpy(frame.image).permute(2, 0, 1),
                        "observation.state": torch.from_numpy(frame.state),
                        "action": torch.from_numpy(frame.action),
                        "task": TASK,
                    }
                )
            dataset.save_episode()
            print(f"episode {episode + 1:03d}/{episodes:03d} saved")
        if hasattr(dataset, "finalize"):
            dataset.finalize()
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
