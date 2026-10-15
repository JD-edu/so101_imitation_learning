from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detector import DEFAULT_DICTIONARY, DEFAULT_MARKER_ID, ArucoReferenceDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect the reference ArUco in an image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="optional annotated preview PNG")
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detector = ArucoReferenceDetector(args.marker_id, args.dictionary)
    detection = detector.detect(image_rgb)
    print(json.dumps(detection.as_dict(), indent=2, ensure_ascii=False))
    if args.output is not None:
        preview_rgb = detector.draw_preview(image_rgb, detection)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(args.output), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
        ):
            raise OSError(f"failed to write preview: {args.output}")
        print(f"preview: {args.output.resolve()}")
    if not detection.visible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
