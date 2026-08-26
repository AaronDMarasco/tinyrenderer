from __future__ import annotations

import sys
from typing import Final

from lib.tgaimage import TGAColor, TGAImage


def _signed_triangle_area(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx))


def triangle_barycentric_lesson_3_homework(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    framebuffer: TGAImage,
) -> None:
    ax, ay, _az = a
    bx, by, _bz = b
    cx, cy, _cz = c
    bb_min_x: Final[int] = min(ax, bx, cx)
    bb_max_x: Final[int] = max(ax, bx, cx)
    bb_min_y: Final[int] = min(ay, by, cy)
    bb_max_y: Final[int] = max(ay, by, cy)
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            color1 = round(alpha * 255)
            color2 = round(beta * 255)
            color3 = round(gamma * 255)
            assert color1 <= 255 and color2 <= 255 and color3 <= 255, f"Bad {color1=} {color2=} {color3=}"
            if alpha <= 0.1 or beta <= 0.1 or gamma <= 0.1:
                framebuffer.set(x, y, TGAColor(color1, color2, color3))


def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
    triangle_barycentric_lesson_3_homework((17, 4, 13), (55, 39, 128), (23, 59, 255), framebuffer)
    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
