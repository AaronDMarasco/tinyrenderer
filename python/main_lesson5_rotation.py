from __future__ import annotations

import sys
from math import cos, pi, sin
from pathlib import Path
from typing import Final

import numpy as np

from lib.objreader import OBJ_Data
from lib.render import triangle_barycentric_lesson_4
from lib.tgaimage import TGAColor_t, TGAImage
from lib.trtypes import vec3

width: Final = 2048
height: Final = 2048

# white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
# green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
# red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
# blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
# yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)

# Not sure where this will end up:


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
            z_buffer = TGAImage(width, height, TGAImage.Format.GRAYSCALE)
            obj_data = OBJ_Data.from_file(fname)
            for face in obj_data.faces:
                idx = (face[0].vertex, face[1].vertex, face[2].vertex)
                a = project(rot(obj_data.vertices[idx[0]]))
                b = project(rot(obj_data.vertices[idx[1]]))
                c = project(rot(obj_data.vertices[idx[2]]))

                triangle_barycentric_lesson_4(a, b, c, z_buffer, framebuffer, TGAColor_t.random())

            framebuffer.write_tga_file(f"{basename}.tga")
            z_buffer.write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
