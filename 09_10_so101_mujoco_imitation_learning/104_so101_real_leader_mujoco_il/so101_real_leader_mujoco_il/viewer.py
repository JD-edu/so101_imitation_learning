from __future__ import annotations

import cv2
import mujoco
import numpy as np
from mujoco.glfw import glfw


OVERVIEW_LOOKAT = (0.45, -0.15, 0.03)
OVERVIEW_DISTANCE = 0.83
OVERVIEW_AZIMUTH = 0.0
OVERVIEW_ELEVATION = -48.4

ASSIST_LOOKAT = (0.32, 0.0, 0.12)
ASSIST_DISTANCE = 0.65
ASSIST_AZIMUTH = 150.0
ASSIST_ELEVATION = -30.0


def configure_overview_camera(camera: mujoco.MjvCamera) -> None:
    """Apply the camera shared by dataset collection and inference."""
    camera.lookat[:] = OVERVIEW_LOOKAT
    camera.distance = OVERVIEW_DISTANCE
    camera.azimuth = OVERVIEW_AZIMUTH
    camera.elevation = OVERVIEW_ELEVATION


def configure_assist_camera(camera: mujoco.MjvCamera) -> None:
    """Apply the optional front-oblique camera used only by the operator."""
    camera.lookat[:] = ASSIST_LOOKAT
    camera.distance = ASSIST_DISTANCE
    camera.azimuth = ASSIST_AZIMUTH
    camera.elevation = ASSIST_ELEVATION


class GLFWFollowerViewer:
    """Policy overview camera with an optional operator-only assist window."""

    def __init__(
        self,
        model,
        data,
        title: str,
        width: int = 1024,
        height: int = 768,
        assist_view: bool = False,
        assist_width: int = 720,
        assist_height: int = 540,
    ):
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        self.window = glfw.create_window(width, height, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        self.assist_window = None
        if assist_view:
            self.assist_window = glfw.create_window(
                assist_width,
                assist_height,
                f"{title} - front assist (display only)",
                None,
                self.window,
            )
            if self.assist_window is None:
                glfw.destroy_window(self.window)
                glfw.terminate()
                raise RuntimeError("front assist GLFW window creation failed")
            glfw.set_window_pos(self.assist_window, width + 30, 40)
            glfw.make_context_current(self.window)
        self.model = model
        self.data = data
        self.scene = mujoco.MjvScene(model, maxgeom=1000)
        self.assist_scene = (
            mujoco.MjvScene(model, maxgeom=1000)
            if self.assist_window is not None
            else None
        )
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        configure_overview_camera(self.camera)
        self.assist_camera = None
        if self.assist_window is not None:
            self.assist_camera = mujoco.MjvCamera()
            configure_assist_camera(self.assist_camera)
        self.context = mujoco.MjrContext(
            model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed and not glfw.window_should_close(self.window)

    def render_and_capture(self, size: tuple[int, int] = (224, 224)) -> np.ndarray:
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        rect = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.option,
            None,
            self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(rect, self.scene, self.context)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, rect, self.context)
        rgb = np.flipud(rgb)
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)

    def present(self) -> None:
        glfw.make_context_current(self.window)
        glfw.swap_buffers(self.window)
        if self.assist_window is not None:
            if glfw.window_should_close(self.assist_window):
                glfw.destroy_window(self.assist_window)
                self.assist_window = None
                self.assist_scene = None
                self.assist_camera = None
            else:
                glfw.make_context_current(self.assist_window)
                width, height = glfw.get_framebuffer_size(self.assist_window)
                rect = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.option,
                    None,
                    self.assist_camera,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.assist_scene,
                )
                mujoco.mjr_render(rect, self.assist_scene, self.context)
                glfw.swap_buffers(self.assist_window)
        glfw.poll_events()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        glfw.make_context_current(self.window)
        if self.assist_window is not None:
            glfw.destroy_window(self.assist_window)
            self.assist_window = None
        glfw.destroy_window(self.window)
        glfw.terminate()
