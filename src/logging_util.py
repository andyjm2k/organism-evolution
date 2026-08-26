"""Central logging helpers gated by simulation logging level."""


def should_log_detailed(logging_level):
    """Return True when verbose/debug messages should be emitted."""
    # Normalize level so callers can pass any casing.
    return str(logging_level).lower() == "detailed"


def log_detailed(logging_level, message):
    """Print message only when logging level is detailed."""
    # Skip I/O entirely on the hot path when logging is normal/minimal.
    if should_log_detailed(logging_level):
        # Emit the provided message to stdout for operators.
        print(message)


def log_always(message):
    """Print an essential message regardless of logging level."""
    # Essential lifecycle messages always reach the operator.
    print(message)
