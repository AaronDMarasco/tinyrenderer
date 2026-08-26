from __future__ import annotations

import sys
from typing import Final

from lesson_2_triangles import line
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage, blue


def _signed_triangle_area(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx))


def triangle_barycentric_lesson_3(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    framebuffer: TGAImage,
) -> None:
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
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
            z = round(alpha * az + beta * bz + gamma * cz)
            assert z <= 255, f"Bad {z=}"

            framebuffer.set(x, y, TGAColor(z))


def triangle_scanlines(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    framebuffer: TGAImage,
    color: TGAColor_t,
    *,
    fill: bool = True,
) -> None:
    """Draw an unfilled triangle"""
    points = line(*a, *b, framebuffer, color, draw=not fill)
    points.extend(line(*b, *c, framebuffer, color, draw=not fill))
    points.extend(line(*c, *a, framebuffer, color, draw=not fill))

    if fill:
        # Scan-line algorithm: no need to sort the Y values just throw them in a set
        for y in {pt[1] for pt in points}:
            # Get all the points drawn at that Y
            this_line = [(px, py) for px, py in points if py == y]
            min_x = min(pt[0] for pt in this_line)
            max_x = max(pt[0] for pt in this_line)
            line(min_x, y, max_x, y, framebuffer, color)


def main() -> int:
    width: Final = 64
    height: Final = 64

    if False:
        framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
        triangle_scanlines((17, 4), (55, 39), (23, 59), framebuffer, blue)
    else:
        framebuffer = TGAImage(width, height, TGAImage.Format.GRAYSCALE)
        triangle_barycentric_lesson_3((17, 4, 13), (55, 39, 128), (23, 59, 255), framebuffer)
    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
