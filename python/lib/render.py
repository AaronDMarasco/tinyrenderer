from __future__ import annotations

from typing import Final

from .tgaimage import TGAColor_t, TGAImage


def line(ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t) -> None:
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
    for x in range(ax, min(bx + 1, framebuffer.width)):
        if steep:  # if transposed, de-transpose
            framebuffer.set(y, x, color)
        else:
            framebuffer.set(x, y, color)
        error += 2 * abs(by - ay)
        if error > (bx - ax):
            y += 1 if by > ay else -1
            error -= error_inc
