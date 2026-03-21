from __future__ import annotations

import random

Color = tuple[int, int, int]


def next_color(rng: random.Random) -> Color:
    """Generate next random RGB color using C# channel sampling order."""
    red = rng.randrange(0, 256)
    blue = rng.randrange(0, 256)
    green = rng.randrange(0, 256)
    return red, green, blue


def remap(clamped: float, start: float, end: float) -> float:
    """Remap 0..1 value into a custom range."""
    return start + clamped * (end - start)
