from __future__ import annotations

import sys
from typing import Final

from lib.plot import plot
from lib.tgaimage import TGAColor_t, TGAImage, green, red, white


def line(
    ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t, *, draw: bool = True
) -> list[tuple[int, int]]:
    """Draw a line (Lesson 1)"""
    steep: Final = abs(ax - bx) < abs(ay - by)
    if steep:
        ax, ay = ay, ax
        bx, by = by, bx
    if ax > bx:  # make it left-to-right
        ax, bx = bx, ax
        ay, by = by, ay
    y: int = ay
    error: int = 0
    error_inc: Final[int] = 2 * (bx - ax)
    points = []
    for x in range(ax, min(bx + 1, framebuffer.width)):
        if steep:  # if transposed, de-transpose
            if draw:
                framebuffer.set(y, x, color)
            points.append((y, x))
        else:
            if draw:
                framebuffer.set(x, y, color)
            points.append((x, y))
        error += 2 * abs(by - ay)
        if error > (bx - ax):
            y += 1 if by > ay else -1
            error -= error_inc
    return points


def triangle(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    framebuffer: TGAImage,
    color: TGAColor_t,
    *,
    fill: bool = False,
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
    width: Final = 128
    height: Final = 128

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
    triangle((7, 45), (35, 100), (45, 60), framebuffer, red)
    triangle((120, 35), (90, 5), (45, 110), framebuffer, white, fill=True)
    triangle((115, 83), (80, 90), (85, 120), framebuffer, green)
    framebuffer.write_tga_file("framebuffer.tga")
    plot(framebuffer)

    return 0


if __name__ == "__main__":
    sys.exit(main())
