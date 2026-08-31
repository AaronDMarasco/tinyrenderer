from __future__ import annotations

import sys
from math import cos, pi, sin
from pathlib import Path
from typing import Final

import numpy as np

from lib.model import Model
from lib.tgaimage import TGAColor_t, TGAImage
from lib.trtypes import ZBuffer, vec3

width: Final = 800
height: Final = 800


def _signed_triangle_area(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx))


def triangle_barycentric_lesson_5(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    z_buffer: ZBuffer,
    framebuffer: TGAImage,
    color: TGAColor_t,
) -> None:
    assert isinstance(z_buffer, ZBuffer), "Wrong version of triangle_barycentric called!"
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    bb_min_x: Final[int] = max(0, min(ax, bx, cx))  # bounding box for the triangle
    bb_max_x: Final[int] = min(framebuffer.width, max(ax, bx, cx))  # defined by its top left and bottom right corners
    bb_min_y: Final[int] = max(0, min(ay, by, cy))  # but bound by canvas size
    bb_max_y: Final[int] = min(framebuffer.height, max(ay, by, cy))

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z = alpha * az + beta * bz + gamma * cz
            if z <= z_buffer.vals[x][y]:  # Behind what we've already drawn
                continue
            z_buffer.vals[x][y] = z
            framebuffer.set(x, y, color)


def rot(v: vec3, rotation: float = pi / 6) -> vec3:
    Ry: Final = np.array(
        [[cos(rotation), 0, sin(rotation)], [0, 1, 0], [-sin(rotation), 0, cos(rotation)]], dtype=float
    )
    res = Ry @ v.np  # numpy notation for matrix x vector = vector
    return vec3.from_np(res)


def project(v: vec3, width: int = width, height: int = height) -> tuple[int, int, int]:
    """
    Orthogonal Projection

    First of all, (x,y) is an orthogonal projection of the vector (x,y,z).
    Second, since the input models are scaled to have fit in the [-1,1]^3 world coordinates,
    we want to shift the vector (x,y) and then scale it to span the entire screen.
    """
    return (
        min(width - 1, round((v.x + 1) * width / 2)),
        min(height - 1, round((v.y + 1) * height / 2)),
        round((v.z + 1) * 255 / 2),
    )


def persp(v: vec3, c: float = 3.0) -> vec3:
    # return vec3.from_np(v.np / (1 - v.z / c))
    ratio: Final = 1 - v.z / c
    return vec3(v.x / ratio, v.y / ratio, v.z / ratio)


def main() -> int:

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
            z_buffer = ZBuffer(width=width, height=height)
            model = Model.from_file(fname)
            for face in model.faces:
                idx = (face[0].vertex, face[1].vertex, face[2].vertex)
                a = project(persp(rot(model.vertices[idx[0]])))
                b = project(persp(rot(model.vertices[idx[1]])))
                c = project(persp(rot(model.vertices[idx[2]])))
                triangle_barycentric_lesson_5(a, b, c, z_buffer, framebuffer, TGAColor_t.random())

            framebuffer.write_tga_file(f"{basename}.tga")
            z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
