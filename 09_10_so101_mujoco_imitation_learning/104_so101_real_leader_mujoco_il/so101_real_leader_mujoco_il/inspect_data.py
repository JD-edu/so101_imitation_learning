from __future__ import annotations

import argparse

import cv2
import numpy as np


def inspect_dataset(
    dataset_root: str,
    repo_id: str,
    episode: int = 0,
    delay_ms: int = 33,
) -> None:
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(f"Inspection requires torch and lerobot: {exc}") from exc

    dataset = LeRobotDataset(repo_id, root=dataset_root)
    print(f"episodes={dataset.num_episodes} frames={dataset.num_frames} fps={dataset.fps}")
    print("features:", ", ".join(dataset.meta.features))
    indices = [
        index
        for index, value in enumerate(dataset.hf_dataset["episode_index"])
        if int(value) == episode
    ]
    if not indices:
        raise ValueError(f"episode {episode} does not exist")

    image_key = next(
        key for key in dataset.meta.features if key.startswith("observation.images.")
    )
    try:
        for frame_number, index in enumerate(indices):
            frame = dataset[index]
            image = frame[image_key].permute(1, 2, 0).cpu().numpy()
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            preview = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.putText(
                preview, f"frame {frame_number}", (8, 24), 0, 0.65, (0, 255, 0), 2
            )
            state = frame.get("observation.state", torch.tensor([])).cpu().numpy()
            environment_state = frame.get(
                "observation.environment_state", torch.tensor([])
            ).cpu().numpy()
            action = frame.get("action", torch.tensor([])).cpu().numpy()
            aruco_text = (
                f" aruco={np.round(environment_state, 3)}"
                if environment_state.size
                else ""
            )
            print(
                f"frame={frame_number:04d} state={np.round(state, 3)} "
                f"action={np.round(action, 3)}{aruco_text}",
                end="\r",
            )
            cv2.imshow("LeRobot episode", preview)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        print()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="local/so101_real_leader_mujoco")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--delay-ms", type=int, default=33)
