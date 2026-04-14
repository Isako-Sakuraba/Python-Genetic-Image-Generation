from __future__ import annotations

import random

Color = tuple[int, int, int]


def next_color(rng: random.Random) -> Color:
    """Generate an RGB color with a fixed channel sampling order."""
    red = rng.randrange(0, 256)
    blue = rng.randrange(0, 256)
    green = rng.randrange(0, 256)
    return red, green, blue


def remap(clamped: float, start: float, end: float) -> float:
    """Map a normalized value from 0..1 into [start, end]."""
    return start + clamped * (end - start)
