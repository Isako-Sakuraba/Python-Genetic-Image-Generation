from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int


class QuadTreeImage:
    def __init__(self, image: np.ndarray, quads: list[Rectangle] | None = None) -> None:
        self.image = image
        if quads is None:
            height, width = image.shape[:2]
            quads = [Rectangle(0, 0, width, height)]
        self.quads = list(quads)

    @classmethod
    def from_image(cls, original: np.ndarray) -> "QuadTreeImage":
        height, width = original.shape[:2]
        return cls(original.copy(), [Rectangle(0, 0, width, height)])

    @classmethod
    def blank(cls, width: int, height: int) -> "QuadTreeImage":
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return cls(image, [Rectangle(0, 0, width, height)])

    @classmethod
    def from_quadtree(cls, original: "QuadTreeImage", clone_image: bool = False) -> "QuadTreeImage":
        image = original.image.copy() if clone_image else original.image
        return cls(image, list(original.quads))

    def can_split(self, quad_index: int) -> bool:
        index = quad_index % len(self.quads)
        rectangle = self.quads[index]
        return rectangle.height != 1 and rectangle.width != 1

    def try_split_non_alloc(self, quad_index: int, result_indexes: list[int]) -> bool:
        if len(result_indexes) != 4:
            return False

        index = quad_index % len(self.quads)
        original = self.quads[index]
        if original.height == 1 or original.width == 1:
            return False

        self.quads[index] = self.get_quadrant(original, quadrant=0)
        result_indexes[0] = index

        for quadrant_index in range(1, 4):
            quadrant = self.get_quadrant(original, quadrant=quadrant_index)
            self.quads.append(quadrant)
            result_indexes[quadrant_index] = index

        return True

    def draw_circle(self, rectangle: Rectangle, color: Color) -> None:
        if rectangle.width <= 4 or rectangle.height <= 4:
            return

        offset = rectangle.width / 2.0
        center_x = rectangle.x + offset
        center_y = rectangle.y + offset
        radius_squared = offset * offset

        x_start = rectangle.x
        x_end = rectangle.x + rectangle.width
        y_start = rectangle.y
        y_end = rectangle.y + rectangle.height

        yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_squared

        region = self.image[y_start:y_end, x_start:x_end]
        region[mask] = np.asarray(color, dtype=np.uint8)

    def fill(self, rectangle: Rectangle, color: Color) -> None:
        x_start = rectangle.x
        x_end = rectangle.x + rectangle.width
        y_start = rectangle.y
        y_end = rectangle.y + rectangle.height
        self.image[y_start:y_end, x_start:x_end] = np.asarray(color, dtype=np.uint8)

    def draw(self, quad_index: int, main: Color, secondary: Color) -> None:
        rectangle = self.quads[quad_index % len(self.quads)]
        self.fill(rectangle, main)
        self.draw_circle(rectangle, secondary)

    @staticmethod
    def get_quadrant(original: Rectangle, quadrant: int) -> Rectangle:
        half_width = original.width // 2
        half_height = original.height // 2
        x = original.x
        y = original.y

        if quadrant == 0:
            return Rectangle(x + half_width, y, half_width, half_height)
        if quadrant == 1:
            return Rectangle(x, y, half_width, half_height)
        if quadrant == 2:
            return Rectangle(x, y + half_height, half_width, half_height)
        if quadrant == 3:
            return Rectangle(x + half_width, y + half_height, half_width, half_height)
        return original
