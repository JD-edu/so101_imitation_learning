from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import cv2
import numpy as np


DEFAULT_DICTIONARY = "DICT_6X6_250"
DEFAULT_MARKER_ID = 23
ARUCO_FEATURE_NAMES = (
    "visible",
    "center_x_normalized",
    "center_y_normalized",
    "side_length_ratio",
    "rotation_sin",
    "rotation_cos",
)


def _aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV was built without the aruco module. Install an OpenCV build "
            "that provides cv2.aruco."
        )
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None:
        raise ValueError(f"unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


@dataclass(frozen=True)
class ArucoDetection:
    marker_id: int
    image_width: int
    image_height: int
    corners: np.ndarray | None = None

    @property
    def visible(self) -> bool:
        return self.corners is not None

    def feature(self) -> np.ndarray:
        """Return a stable six-value ACT environment-state feature."""
        if self.corners is None:
            return np.zeros(len(ARUCO_FEATURE_NAMES), dtype=np.float32)

        points = np.asarray(self.corners, dtype=np.float32).reshape(4, 2)
        center = points.mean(axis=0)
        edges = np.roll(points, -1, axis=0) - points
        side = float(np.linalg.norm(edges, axis=1).mean())
        angle = atan2(float(edges[0, 1]), float(edges[0, 0]))
        width_scale = max(self.image_width - 1, 1)
        height_scale = max(self.image_height - 1, 1)
        return np.asarray(
            [
                1.0,
                2.0 * float(center[0]) / width_scale - 1.0,
                2.0 * float(center[1]) / height_scale - 1.0,
                side / min(self.image_width, self.image_height),
                sin(angle),
                cos(angle),
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, object]:
        feature = self.feature()
        result: dict[str, object] = {
            "marker_id": self.marker_id,
            "visible": self.visible,
            "feature": {
                name: float(value)
                for name, value in zip(ARUCO_FEATURE_NAMES, feature, strict=True)
            },
        }
        if self.corners is not None:
            result["corners_px"] = np.asarray(self.corners).reshape(4, 2).tolist()
        return result


class ArucoReferenceDetector:
    def __init__(
        self,
        marker_id: int = DEFAULT_MARKER_ID,
        dictionary: str = DEFAULT_DICTIONARY,
    ):
        self.marker_id = int(marker_id)
        self.dictionary_name = dictionary
        self.dictionary = _aruco_dictionary(dictionary)
        if not 0 <= self.marker_id < len(self.dictionary.bytesList):
            raise ValueError(
                f"marker ID {self.marker_id} is outside {dictionary} range "
                f"0..{len(self.dictionary.bytesList) - 1}"
            )
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)

    def detect(self, image_rgb: np.ndarray) -> ArucoDetection:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB image shaped HxWx3, got {image.shape}")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _rejected = self.detector.detectMarkers(gray)
        selected = None
        if ids is not None:
            flat_ids = ids.reshape(-1)
            matches = np.flatnonzero(flat_ids == self.marker_id)
            if matches.size:
                selected = np.asarray(corners[int(matches[0])], dtype=np.float32).reshape(4, 2)
        return ArucoDetection(
            marker_id=self.marker_id,
            image_width=int(image.shape[1]),
            image_height=int(image.shape[0]),
            corners=selected,
        )

    @staticmethod
    def draw_preview(image_rgb: np.ndarray, detection: ArucoDetection) -> np.ndarray:
        """Draw detection on a copy; never modify the policy observation image."""
        preview = np.asarray(image_rgb).copy()
        if detection.corners is None:
            cv2.putText(
                preview,
                f"ArUco {detection.marker_id}: not visible",
                (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
            )
            return preview
        points = np.rint(detection.corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(preview, [points], True, (0, 255, 0), 2)
        center = np.rint(detection.corners.mean(axis=0)).astype(int)
        cv2.circle(preview, tuple(center), 4, (0, 255, 0), -1)
        cv2.putText(
            preview,
            f"ArUco {detection.marker_id}",
            (8, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        return preview
