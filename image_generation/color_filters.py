from __future__ import annotations

import colorsys

Color = tuple[int, int, int]


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_unit_rgb(color: Color) -> tuple[float, float, float]:
    return color[0] / 255.0, color[1] / 255.0, color[2] / 255.0


def _from_unit_rgb(color: tuple[float, float, float]) -> Color:
    return (
        int(_clamp_01(color[0]) * 255.0),
        int(_clamp_01(color[1]) * 255.0),
        int(_clamp_01(color[2]) * 255.0),
    )


def adjust_brightness(color: Color, multiplier: float) -> Color:
    red, green, blue = _to_unit_rgb(color)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    adjusted = (hue, saturation, _clamp_01(value * multiplier))
    return _from_unit_rgb(colorsys.hsv_to_rgb(*adjusted))


def adjust_saturation(color: Color, multiplier: float) -> Color:
    red, green, blue = _to_unit_rgb(color)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    adjusted = (hue, _clamp_01(saturation * multiplier), value)
    return _from_unit_rgb(colorsys.hsv_to_rgb(*adjusted))


def shift_hue(color: Color, degrees: float) -> Color:
    red, green, blue = _to_unit_rgb(color)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    shifted_hue = (hue + (degrees / 360.0)) % 1.0
    return _from_unit_rgb(colorsys.hsv_to_rgb(shifted_hue, saturation, value))


def adjust_lightness(color: Color, multiplier: float) -> Color:
    red, green, blue = _to_unit_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    adjusted = (hue, _clamp_01(lightness * multiplier), saturation)
    return _from_unit_rgb(colorsys.hls_to_rgb(*adjusted))


def desaturate(color: Color) -> Color:
    return adjust_saturation(color, 0.0)


def invert(color: Color) -> Color:
    return 255 - color[0], 255 - color[1], 255 - color[2]


def blend(color_1: Color, color_2: Color, ratio: float) -> Color:
    ratio = _clamp_01(ratio)
    return (
        int(color_1[0] + (color_2[0] - color_1[0]) * ratio),
        int(color_1[1] + (color_2[1] - color_1[1]) * ratio),
        int(color_1[2] + (color_2[2] - color_1[2]) * ratio),
    )
