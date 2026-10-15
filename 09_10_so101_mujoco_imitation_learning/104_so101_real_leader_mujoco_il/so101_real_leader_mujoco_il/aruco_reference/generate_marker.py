from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import DEFAULT_DICTIONARY, DEFAULT_MARKER_ID, _aruco_dictionary


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "assets" / "aruco_6x6_250_id23.png"
)


def generate_marker(
    output: str | Path = DEFAULT_OUTPUT,
    marker_id: int = DEFAULT_MARKER_ID,
    dictionary: str = DEFAULT_DICTIONARY,
    pixels: int = 512,
    quiet_zone: int = 56,
    border_bits: int = 1,
    physical_size_m: float = 0.12,
) -> Path:
    output = Path(output).expanduser().resolve()
    if pixels <= 0 or quiet_zone < 0 or pixels <= 2 * quiet_zone:
        raise ValueError("pixels must be positive and larger than twice quiet_zone")
    if border_bits <= 0:
        raise ValueError("border_bits must be positive")
    if physical_size_m <= 0:
        raise ValueError("physical_size_m must be positive")
    aruco_dictionary = _aruco_dictionary(dictionary)
    if not 0 <= marker_id < len(aruco_dictionary.bytesList):
        raise ValueError(
            f"marker ID {marker_id} is outside {dictionary} range "
            f"0..{len(aruco_dictionary.bytesList) - 1}"
        )

    core_size = pixels - 2 * quiet_zone
    core = cv2.aruco.generateImageMarker(
        aruco_dictionary, marker_id, core_size, borderBits=border_bits
    )
    image = np.full((pixels, pixels), 255, dtype=np.uint8)
    image[quiet_zone : quiet_zone + core_size, quiet_zone : quiet_zone + core_size] = core
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"failed to write marker image: {output}")
    metadata = {
        "dictionary": dictionary,
        "marker_id": marker_id,
        "pixels": pixels,
        "quiet_zone_pixels": quiet_zone,
        "border_bits": border_bits,
        "physical_outer_size_m": physical_size_m,
        "feature_note": "Physical size includes the white quiet zone.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the MuJoCo ArUco reference PNG")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dictionary", default=DEFAULT_DICTIONARY)
    parser.add_argument("--marker-id", type=int, default=DEFAULT_MARKER_ID)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--quiet-zone", type=int, default=56)
    parser.add_argument("--border-bits", type=int, default=1)
    parser.add_argument("--physical-size-m", type=float, default=0.12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = generate_marker(
        output=args.output,
        marker_id=args.marker_id,
        dictionary=args.dictionary,
        pixels=args.pixels,
        quiet_zone=args.quiet_zone,
        border_bits=args.border_bits,
        physical_size_m=args.physical_size_m,
    )
    print(f"marker PNG: {output}")
    print(f"metadata: {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
