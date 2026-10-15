"""ArUco workspace-reference tools shared by collection and inference."""

from .detector import (
    ARUCO_FEATURE_NAMES,
    DEFAULT_DICTIONARY,
    DEFAULT_MARKER_ID,
    ArucoDetection,
    ArucoReferenceDetector,
)

__all__ = [
    "ARUCO_FEATURE_NAMES",
    "DEFAULT_DICTIONARY",
    "DEFAULT_MARKER_ID",
    "ArucoDetection",
    "ArucoReferenceDetector",
]
