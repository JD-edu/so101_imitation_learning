from __future__ import annotations

import time

import mujoco
import numpy as np
from mujoco.glfw import glfw


class GLFWViewer:
    """Minimal GLFW renderer; deliberately does not use mujoco.viewer."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, title: str):
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize (use --headless without a display)")
        self.window = glfw.create_window(960, 720, title, None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.model, self.data = model, data
        self.scene = mujoco.MjvScene(model, maxgeom=2000)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.option = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed and not glfw.window_should_close(self.window)

    def render(self) -> None:
        glfw.make_context_current(self.window)
        width, height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model, self.data, self.option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def capture(self, width: int = 224, height: int = 224) -> np.ndarray:
        glfw.make_context_current(self.window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            self.model, self.data, self.option, None, self.camera,
            mujoco.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        return np.flipud(rgb).copy()

    def close(self) -> None:
        if self._closed:
            return
        glfw.destroy_window(self.window)
        glfw.terminate()
        self._closed = True


def replay_actions(env, actions: np.ndarray, fps: int = 30) -> None:
    viewer = GLFWViewer(env.model, env.data, "SO-101 IK pick and place")
    try:
        for action in actions:
            if not viewer.alive:
                break
            start = time.monotonic()
            env.step(action)
            viewer.render()
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - start)))
    finally:
        viewer.close()
