from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco

from ..runtime import DEFAULT_XML, MujocoFollower
from ..viewer import configure_overview_camera
from .detector import DEFAULT_DICTIONARY, DEFAULT_MARKER_ID, ArucoReferenceDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the 104 MuJoCo overview camera and verify its ArUco marker"
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=Path("aruco_scene_check.png"))
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--size", type=int, default=224)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive")
    follower = MujocoFollower(xml_path=args.xml, seed=0)
    camera = mujoco.MjvCamera()
    configure_overview_camera(camera)
    # Match GLFWFollowerViewer's 1024x768 framebuffer, then apply the same
    # 224x224-style resize used by render_and_capture().
    with mujoco.Renderer(follower.model, height=768, width=1024) as renderer:
        renderer.update_scene(follower.data, camera=camera)
        image_rgb = renderer.render()
    image_rgb = cv2.resize(
        image_rgb, (args.size, args.size), interpolation=cv2.INTER_AREA
    )

    detector = ArucoReferenceDetector(args.marker_id, args.dictionary)
    detection = detector.detect(image_rgb)
    preview_rgb = detector.draw_preview(image_rgb, detection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(args.output), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    ):
        raise OSError(f"failed to write preview: {args.output}")
    print(json.dumps(detection.as_dict(), indent=2, ensure_ascii=False))
    print(f"preview: {args.output.resolve()}")
    if not detection.visible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
