from __future__ import annotations

import sys
from typing import Final

from lib.model import OBJ_Data
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage

red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)


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
    width: Final = 2047
    height: Final = 2048

    # The numbers go from -1..1 so we need to map them from the center of the image...
    width_center: Final[int] = width // 2 - 1
    height_center: Final[int] = height // 2 - 1

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

    obj_data = OBJ_Data.from_file("../obj/diablo3_pose/diablo3_pose.obj")

    for face in obj_data.faces:
        # Get the indices of the vertices
        idx = (face[0].vertex, face[1].vertex, face[2].vertex)
        # Read those out
        points = (obj_data.vertices[idx[0]], obj_data.vertices[idx[1]], obj_data.vertices[idx[2]])
        # Draw the lines
        for i in range(3):
            this = i % 3
            that = (i + 1) % 3
            line(
                round((points[this].x + 1) * (width_center)),
                round((points[this].y + 1) * (height_center)),
                round((points[that].x + 1) * (width_center)),
                round((points[that].y + 1) * (height_center)),
                framebuffer,
                red,
            )

    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
