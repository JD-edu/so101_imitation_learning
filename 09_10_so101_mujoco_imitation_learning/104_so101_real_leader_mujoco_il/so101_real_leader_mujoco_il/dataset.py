from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import JOINT_NAMES, TASK
from .aruco_reference import (
    ARUCO_FEATURE_NAMES,
    DEFAULT_MARKER_ID,
    ArucoReferenceDetector,
)
from .hardware import MiniFeetechDriver
from .runtime import DEFAULT_CALIBRATION, DEFAULT_XML, MujocoFollower
from .runtime import load_leader_calibration, raw_to_joint_target
from .viewer import GLFWFollowerViewer


def _prepare_dataset_root(root: Path, recover_empty: bool) -> Path | None:
    """Optionally preserve and replace a verified zero-frame dataset skeleton."""
    if not root.exists() or not any(root.iterdir()):
        return None

    info_path = root / "meta" / "info.json"
    is_empty_skeleton = False
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            files = [path for path in root.rglob("*") if path.is_file()]
            is_empty_skeleton = (
                int(info.get("total_episodes", -1)) == 0
                and int(info.get("total_frames", -1)) == 0
                and files == [info_path]
            )
        except (OSError, ValueError, TypeError):
            is_empty_skeleton = False

    if not recover_empty:
        detail = (
            " It is a zero-frame initialization skeleton; rerun with "
            "--recover-empty to preserve it under a backup name and start fresh."
            if is_empty_skeleton
            else " Use a new --dataset-root to protect the existing data."
        )
        raise FileExistsError(f"dataset directory is not empty: {root}.{detail}")
    if not is_empty_skeleton:
        raise FileExistsError(
            f"refusing --recover-empty because the directory may contain data: {root}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = root.with_name(f"{root.name}.incomplete_{timestamp}")
    root.rename(backup)
    print(f"preserved zero-frame dataset skeleton at: {backup}")
    return backup


def collect_dataset(
    root: str | Path,
    repo_id: str = "local/so101_real_leader_mujoco",
    leader_port: str = "/dev/ttyUSB0",
    calibration_path: str | Path = DEFAULT_CALIBRATION,
    xml_path: str | Path = DEFAULT_XML,
    fps: int = 30,
    seed: int = 0,
    recover_empty: bool = False,
    aruco_state: bool = False,
    aruco_id: int = DEFAULT_MARKER_ID,
    assist_view: bool = False,
) -> None:
    try:
        import torch
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(f"Collection requires torch and lerobot: {exc}") from exc

    root = Path(root).expanduser().resolve()
    _prepare_dataset_root(root, recover_empty)

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
    if aruco_state:
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (len(ARUCO_FEATURE_NAMES),),
            "names": list(ARUCO_FEATURE_NAMES),
        }
    # Validate the model/calibration and connect hardware/viewer before creating
    # anything on disk. A serial or display failure must not leave another
    # zero-frame dataset skeleton behind.
    follower = MujocoFollower(xml_path=xml_path, seed=seed)
    calibration = load_leader_calibration(calibration_path)
    leader = MiniFeetechDriver(port=leader_port)
    for name in JOINT_NAMES:
        leader.set_torque(int(calibration[name]["id"]), False)
    viewer = GLFWFollowerViewer(
        follower.model,
        follower.data,
        "SO-101 real leader -> MuJoCo dataset",
        assist_view=assist_view,
    )
    aruco_detector = ArucoReferenceDetector(marker_id=aruco_id) if aruco_state else None
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=features,
        robot_type="so101_real_leader_mujoco_follower",
        use_videos=False,
    )

    recording = False
    episode_frames = 0
    next_frame_time = 0.0
    target = follower.state.astype(np.float64)
    print("r: record | s: save episode/reset | d: discard/reset | q: quit")
    try:
        while viewer.alive:
            valid = True
            for index, name in enumerate(JOINT_NAMES):
                raw = leader.get_position(int(calibration[name]["id"]))
                if raw is None:
                    valid = False
                    break
                target[index] = raw_to_joint_target(
                    raw, calibration[name], follower.joint_limits[index]
                )
            if valid:
                follower.apply_target(target)
            follower.step_for(1.0 / 60.0)
            image = viewer.render_and_capture()
            aruco_detection = (
                aruco_detector.detect(image) if aruco_detector is not None else None
            )
            viewer.present()

            if recording and valid and follower.data.time + 1e-9 >= next_frame_time:
                frame = {
                    "observation.images.overview": torch.from_numpy(image.copy()).permute(2, 0, 1),
                    "observation.state": torch.from_numpy(follower.state),
                    "action": torch.from_numpy(target.astype(np.float32).copy()),
                    "task": TASK,
                }
                if aruco_detection is not None:
                    frame["observation.environment_state"] = torch.from_numpy(
                        aruco_detection.feature()
                    )
                dataset.add_frame(frame)
                episode_frames += 1
                next_frame_time += 1.0 / fps

            preview_rgb = (
                aruco_detector.draw_preview(image, aruco_detection)
                if aruco_detector is not None and aruco_detection is not None
                else image
            )
            preview = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
            if recording:
                cv2.putText(preview, f"REC {episode_frames}", (8, 28), 0, 0.8, (0, 0, 255), 2)
            cv2.imshow("ACT observation", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                recording = True
                episode_frames = 0
                next_frame_time = follower.data.time
                print("recording started")
            elif key == ord("s") and recording:
                if episode_frames:
                    dataset.save_episode()
                    print(f"episode saved ({episode_frames} frames)")
                recording = False
                episode_frames = 0
                follower.reset()
                target = follower.state.astype(np.float64)
            elif key == ord("d"):
                if hasattr(dataset, "clear_episode_buffer"):
                    dataset.clear_episode_buffer()
                else:
                    dataset.episode_buffer = dataset.create_episode_buffer()
                recording = False
                episode_frames = 0
                follower.reset()
                target = follower.state.astype(np.float64)
                print("episode discarded")
            elif key == ord("q"):
                break
    finally:
        # An unfinished episode is intentionally not saved: partial motions are
        # usually harmful demonstrations and quitting must not silently add one.
        if recording and episode_frames:
            if hasattr(dataset, "clear_episode_buffer"):
                dataset.clear_episode_buffer()
        try:
            if hasattr(dataset, "finalize"):
                dataset.finalize()
        finally:
            leader.close()
            viewer.close()
            cv2.destroyAllWindows()
