from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import mujoco

from ..cli import build_parser
from ..runtime import DEFAULT_XML
from ..viewer import (
    ASSIST_AZIMUTH,
    ASSIST_ELEVATION,
    ASSIST_LOOKAT,
    configure_assist_camera,
)
from .detector import ARUCO_FEATURE_NAMES, ArucoReferenceDetector
from .generate_marker import generate_marker


class ArucoReferenceTest(unittest.TestCase):
    def test_generated_marker_is_detectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = generate_marker(Path(directory) / "marker.png")
            image_bgr = cv2.imread(str(output))
            self.assertIsNotNone(image_bgr)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            detection = ArucoReferenceDetector().detect(image_rgb)
            self.assertTrue(detection.visible)
            feature = detection.feature()
            self.assertEqual(feature.shape, (len(ARUCO_FEATURE_NAMES),))
            self.assertEqual(float(feature[0]), 1.0)
            self.assertAlmostEqual(float(feature[1]), 0.0, places=2)
            self.assertAlmostEqual(float(feature[2]), 0.0, places=2)

    def test_missing_marker_is_an_explicit_zero_feature(self) -> None:
        blank = cv2.cvtColor(
            cv2.imread(str(Path(__file__).resolve().parents[1] / "assets" / "aruco_6x6_250_id23.png")),
            cv2.COLOR_BGR2RGB,
        )
        blank[:] = 255
        detection = ArucoReferenceDetector().detect(blank)
        self.assertFalse(detection.visible)
        self.assertTrue((detection.feature() == 0).all())

    def test_default_xml_contains_reference_body_and_sensors(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "aruco_reference_frame"
            ),
            0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "aruco_reference_marker"
            ),
            0,
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SENSOR, "aruco_reference_pos"
            ),
            0,
        )

    def test_cli_exposes_aruco_collection_and_inference_options(self) -> None:
        collect = build_parser().parse_args(
            [
                "collect",
                "--dataset-root",
                "new-data",
                "--aruco-state",
                "--aruco-id",
                "23",
                "--assist-view",
            ]
        )
        self.assertTrue(collect.aruco_state)
        self.assertEqual(collect.aruco_id, 23)
        self.assertTrue(collect.assist_view)
        infer = build_parser().parse_args(
            [
                "infer",
                "--checkpoint",
                "checkpoint",
                "--aruco-id",
                "23",
                "--assist-view",
            ]
        )
        self.assertEqual(infer.aruco_id, 23)
        self.assertTrue(infer.assist_view)

    def test_assist_camera_is_front_oblique(self) -> None:
        camera = mujoco.MjvCamera()
        configure_assist_camera(camera)
        self.assertEqual(tuple(camera.lookat), ASSIST_LOOKAT)
        self.assertEqual(camera.azimuth, ASSIST_AZIMUTH)
        self.assertEqual(camera.elevation, ASSIST_ELEVATION)
        self.assertNotEqual(camera.azimuth, 180.0)

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene = mujoco.MjvScene(model, maxgeom=1000)
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scene,
        )
        eye = (scene.camera[0].pos + scene.camera[1].pos) / 2
        self.assertGreater(float(eye[0]), ASSIST_LOOKAT[0])
        self.assertNotAlmostEqual(float(eye[1]), ASSIST_LOOKAT[1], places=2)


if __name__ == "__main__":
    unittest.main()
