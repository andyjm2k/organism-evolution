"""Probe whether pygame + moderngl can create an OpenGL context (for GPU smoke tests)."""

_opengl_available = None


def opengl_available():
    """Return True when a minimal OpenGL context can be created on this host."""
    global _opengl_available
    if _opengl_available is not None:
        return _opengl_available
    try:
        import moderngl
        import pygame
        from pygame import DOUBLEBUF, OPENGL

        if not pygame.get_init():
            pygame.init()
        pygame.display.set_mode((64, 64), OPENGL | DOUBLEBUF)
        moderngl.create_context()
        _opengl_available = True
    except Exception:
        _opengl_available = False
    finally:
        try:
            import pygame

            if pygame.get_init():
                if pygame.display.get_init():
                    pygame.display.quit()
                pygame.quit()
        except Exception:
            pass
    return _opengl_available


def skip_reason():
    """Human-readable reason when OpenGL is unavailable."""
    if opengl_available():
        return None
    return "OpenGL display/context not available (install libGL and use a display)"
