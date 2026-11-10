"""Fix for corrupted camera frames after a RoboCasa hard reset (found 2026-11-07).

A hard reset builds a new MuJoCo simulation with a new offscreen render context. The old context is freed later, when
Python's garbage collector gets to it. robosuite's ``MjRenderContext.__del__`` then calls ``mjr_freeContext`` without
first making the old GL context current, so it deletes textures and framebuffers of whichever context *is* current:
the live one. Every later camera frame is noise, or shows another camera's image. Because it depends on when the
garbage collector runs, it looked random (imports and allocations changed the timing).

The fix makes the old context current before freeing it. robosuite makes the live context current again before
each render, so nothing else changes. Applied on import; third_party code is not edited.

``mujoco.Renderer.close`` (the free cameras of the video and of the Hydra mapping pass) has the same flaw the other
way round: it destroys its GL context first and then calls ``mjr_freeContext`` with whatever context is current,
which can be robosuite's. It gets the same fix: free the MuJoCo context with its own GL context current, then the
GL context.
"""

import mujoco
from robosuite.utils.binding_utils import MjRenderContext


def _free_own_context(self) -> None:
    try:
        self.gl_ctx.make_current()  # free the GL objects of *this* context, not of the current one
        self.con.free()
        self.gl_ctx.free()
    except Exception:  # interpreter shutdown or an already freed context: nothing left to free
        pass


MjRenderContext.__del__ = _free_own_context


def _close_renderer(self) -> None:
    gl, mjr = getattr(self, "_gl_context", None), getattr(self, "_mjr_context", None)
    self._mjr_context = self._gl_context = None
    try:
        if gl and mjr:
            gl.make_current()  # mjr_freeContext acts on the current context: make it this renderer's own
            mjr.free()
        if gl:
            gl.free()
    except Exception:  # interpreter shutdown: the display is already gone
        pass


mujoco.Renderer.close = _close_renderer
