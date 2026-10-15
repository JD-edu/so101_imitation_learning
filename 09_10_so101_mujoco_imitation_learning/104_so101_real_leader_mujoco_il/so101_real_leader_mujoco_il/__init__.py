"""Real SO-101 leader to MuJoCo follower ACT pipeline."""

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

TASK = (
    "Use ArUco marker 23 as the workspace reference. "
    "Pick up the red block and place it at the target location."
)

__all__ = ["JOINT_NAMES", "TASK"]
