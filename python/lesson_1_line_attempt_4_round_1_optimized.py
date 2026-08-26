from __future__ import annotations

import sys
from typing import Final

import numpy as np

from lib.tgaimage import TGAColor, TGAColor_t, TGAImage


def line(ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t) -> None:
    steep: Final = abs(ax - bx) < abs(ay - by)
    if steep:
        ax, ay = ay, ax
        bx, by = by, bx
    if ax > bx:  # make it left-to-right
        ax, bx = bx, ax
        ay, by = by, ay
    y: float = ay
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


def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

    _CPP_SOURCE = """
    std::srand(std::time({}));
    for (int i=0; i<(1<<24); i++) {
        int ax = rand()%width, ay = rand()%height;
        int bx = rand()%width, by = rand()%height;
        line(ax, ay, bx, by, framebuffer, { rand()%255, rand()%255, rand()%255, rand()%255 });
    }
    """

    rng = np.random.default_rng()  # seed=42)
    for _ in range(1 << 18):  # 4.0s (LOL - C++ and Python optimize differently!)
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
