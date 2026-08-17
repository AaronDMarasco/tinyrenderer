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
    error: float = 0.0
    if bx == ax:  # BUG: C++ allowed the div by zero here?
        bx += 1
    y_inc: Final = (by - ay) / (bx - ax)
    for x in range(ax, min(bx + 1, framebuffer.width)):
        y += y_inc
        fixed_y = min(round(y), framebuffer.height - 1)
        if steep:  # if transposed, de-transpose
            framebuffer.set(fixed_y, x, color)
        else:
            framebuffer.set(x, fixed_y, color)
        _CPP_SOURCE = """
        error += std::abs(by-ay)/static_cast<float>(bx-ax);
        if (error>.5) {
            y += by > ay ? 1 : -1;
            error -= 1.;
        }
        """
        error += abs(by-ay)/(bx-ax)
        if error  > 0.5:
            y += 1 if by > ay else -1
            error -= 1

def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

    rng = np.random.default_rng()  # seed=42)
    for _ in range(1 << 18):  # 4.6s (even worse!)
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
