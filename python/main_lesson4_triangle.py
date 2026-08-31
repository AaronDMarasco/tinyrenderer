from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np

from lib.model import OBJ_Data
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage


def _signed_triangle_area(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx))


def triangle_barycentric_lesson_4(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    z_buffer: TGAImage,
    framebuffer: TGAImage,
    color: TGAColor_t,
) -> None:
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    bb_min_x: Final[int] = min(ax, bx, cx)  # bounding box for the triangle
    bb_max_x: Final[int] = max(ax, bx, cx)  # defined by its top left and bottom right corners
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
            z_color = TGAColor(z)
            if z_color <= z_buffer.get(x, y):  # Behind what we've already drawn
                continue
            z_buffer.set(x, y, z_color)
            framebuffer.set(x, y, color)


def main() -> int:
    width: Final = 2047
    height: Final = 2048

    # The numbers go from -1..1 so we need to map them from the center of the image...
    width_center: Final[int] = width // 2 - 1
    height_center: Final[int] = height // 2 - 1

    rng = np.random.default_rng()  # seed=42)

    find_output = """
../obj/boggie/head.obj
../obj/boggie/body.obj
../obj/boggie/eyes.obj
../obj/african_head/african_head.obj
../obj/african_head/african_head_eye_inner.obj
../obj/african_head/african_head_eye_outer.obj
../obj/floor.obj
../obj/diablo3_pose/diablo3_pose.obj
"""
    for fname in find_output.split():
        basename = Path(fname).name[:-4]
        try:
            framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
            z_buffer = TGAImage(width, height, TGAImage.Format.GRAYSCALE)
            obj_data = OBJ_Data.from_file(fname)
            for face in obj_data.faces:
                idx = (face[0].vertex, face[1].vertex, face[2].vertex)
                # Read those out
                points = (obj_data.vertices[idx[0]], obj_data.vertices[idx[1]], obj_data.vertices[idx[2]])
                # Fix quadrant and scaling
                a = (
                    round((points[0].x + 1) * width_center),
                    round((points[0].y + 1) * height_center),
                    round(127 * (points[0].z + 1)),
                )
                b = (
                    round((points[1].x + 1) * width_center),
                    round((points[1].y + 1) * height_center),
                    round(127 * (points[1].z + 1)),
                )
                c = (
                    round((points[2].x + 1) * width_center),
                    round((points[2].y + 1) * height_center),
                    round(127 * (points[2].z + 1)),
                )

                triangle_barycentric_lesson_4(
                    a,
                    b,
                    c,
                    z_buffer,
                    framebuffer,
                    # NEW: TGAColor_t.random(),
                    TGAColor(int(rng.integers(255)), int(rng.integers(255)), int(rng.integers(255))),
                )

            framebuffer.write_tga_file(f"{basename}.tga")
            z_buffer.write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
