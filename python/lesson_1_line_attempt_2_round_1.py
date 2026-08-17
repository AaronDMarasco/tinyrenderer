from __future__ import annotations

import sys
from typing import Final

from lib.tgaimage import TGAColor, TGAColor_t, TGAImage

white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)


def line(ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t) -> None:
    _CPP_SOURCE = """
    for (int x=ax; x<=bx; x++) {
        float t = (x-ax) / static_cast<float>(bx-ax);
        int y = std::round( ay + (by-ay)*t );
        framebuffer.set(x, y, color);
    """
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
