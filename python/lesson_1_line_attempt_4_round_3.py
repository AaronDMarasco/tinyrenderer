from __future__ import annotations

import sys
from typing import Final

import numpy as np

from lib.tgaimage import TGAColor, TGAColor_t, TGAImage

white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)


def line(ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t) -> None:
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
    # No more division, so no more divide by zero
    for x in range(ax, min(bx + 1, framebuffer.width)):
        if steep:  # if transposed, de-transpose
            framebuffer.set(y, x, color)
        else:
            framebuffer.set(x, y, color)
        error += 2 * abs(by - ay)
        if error > (bx - ax):
            y += 1 if by > ay else -1
            error -= error_inc


def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

    rng = np.random.default_rng()  # seed=42)
    for _ in range(1 << 18):  # 3.4s (was 3.7s in original)
        ax = int(rng.integers(width))
        ay = int(rng.integers(width))
        bx = int(rng.integers(width))
        by = int(rng.integers(width))
        line(
            ax,
            ay,
            bx,
            by,
            framebuffer,
            TGAColor(int(rng.integers(255)), int(rng.integers(255)), int(rng.integers(255))),
        )

    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
