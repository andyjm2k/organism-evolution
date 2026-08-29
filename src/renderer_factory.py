"""Factory for pygame vs ModernGL simulation render backends."""

from logging_util import log_always


def create_renderer(size, logging_level="normal", backend="pygame"):
    """
    Return a renderer for the requested backend.

    Falls back to pygame when moderngl or an OpenGL context is unavailable.
    """
    chosen = (backend or "pygame").strip().lower()
    if chosen == "moderngl":
        try:
            from renderer_gl import ModernGLRenderer

            return ModernGLRenderer(size, logging_level=logging_level)
        except Exception as exc:
            log_always(
                f"ModernGL renderer unavailable ({exc}); falling back to pygame"
            )
    from renderer import PygameRenderer

    return PygameRenderer(size, logging_level=logging_level)
