"""Fix for corrupted camera frames after a RoboCasa hard reset (found 2026-11-07).

A hard reset builds a new MuJoCo simulation with a new offscreen render context. The old context is freed later, when
Python's garbage collector gets to it. robosuite's ``MjRenderContext.__del__`` then calls ``mjr_freeContext`` without
first making the old GL context current, so it deletes textures and framebuffers of whichever context *is* current:
the live one. Every later camera frame is noise, or shows another camera's image. Because it depends on when the
garbage collector runs, it looked random (imports and allocations changed the timing).

The fix makes the old context current before freeing it. robosuite makes the live context current again before
each render, so nothing else changes. Applied on import; third_party code is not edited.
"""

from robosuite.utils.binding_utils import MjRenderContext


def _free_own_context(self) -> None:
    try:
        self.gl_ctx.make_current()  # free the GL objects of *this* context, not of the current one
        self.con.free()
        self.gl_ctx.free()
    except Exception:  # interpreter shutdown or an already freed context: nothing left to free
        pass


MjRenderContext.__del__ = _free_own_context
