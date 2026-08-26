from __future__ import annotations

import sys
from typing import Final

from lib.tgaimage import TGAColor_t, TGAImage, blue, green, red, white, yellow


def line(ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t) -> None:
    _CPP_SOURCE = """
    if (ax>bx) { // make it left-to-right
        std::swap(ax, bx);
        std::swap(ay, by);
    }
    """
    if ax > bx:  # make it left-to-right
        ax, bx = bx, ax
        ay, by = by, ay
    for x in range(ax, bx + 1):
        t = (x - ax) / (bx - ax)
        y = round(ay + (by - ay) * t)
        framebuffer.set(x, y, color)


def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

    ax = 7
    ay = 3
    bx = 12
    by = 37
    cx = 62
    cy = 53

    line(ax, ay, bx, by, framebuffer, blue)
    line(cx, cy, bx, by, framebuffer, green)
    line(cx, cy, ax, ay, framebuffer, yellow)
    line(ax, ay, cx, cy, framebuffer, red)

    framebuffer.set(ax, ay, white)
    framebuffer.set(bx, by, white)
    framebuffer.set(cx, cy, white)

    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
