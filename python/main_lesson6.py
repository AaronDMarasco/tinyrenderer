from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np

from lib.objreader import OBJ_Data
from lib.render import rasterize_lesson_6
from lib.tgaimage import TGAColor_t, TGAImage
from lib.trtypes import Matrix4f, ZBuffer, vec3, vec4

width: Final = 800
height: Final = 800

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector


def viewport_gen(x: int, y: int, width: int = width, height: int = height) -> Matrix4f:
    h_2: Final = height / 2
    w_2: Final = width / 2
    # fmt: off
    # ruff: ignore[multiple-spaces-after-comma,whitespace-after-open-bracket]
    return np.array([
        [w_2,   0,  0, x + w_2],
        [  0, h_2,  0, y + h_2],
        [  0,   0,  1,       0],
        [  0,   0,  0,       1],
    ])


# fmt: on


def perspective_gen(f: float) -> Matrix4f:
    # fmt: off
    # ruff: ignore[multiple-spaces-after-comma,missing-whitespace-around-arithmetic-operator]
    return np.array([
        [1, 0,    0,  0],
        [0, 1,    0,  0],
        [0, 0,    1,  0],
        [0, 0, -1/f,  1],
    ])


# fmt: on


def lookat(eye: vec3, center: vec3, up: vec3) -> Matrix4f:
    # See https://haqr.eu/tinyrenderer/camera/ for vector naming
    n_vec = (eye - center).normalized
    l_vec = up.cross(n_vec).normalized
    m_vec = n_vec.cross(l_vec).normalized
    # fmt: off
    # ruff: ignore[multiple-spaces-after-comma,whitespace-after-open-bracket]
    return np.array([
        [l_vec.x, l_vec.y, l_vec.z, 0],
        [m_vec.x, m_vec.y, m_vec.z, 0],
        [n_vec.x, n_vec.y, n_vec.z, 0],
        [      0,       0,       0, 1],
    ]) @ np.array([
        [1, 0, 0, -center.x],
        [0, 1, 0, -center.y],
        [0, 0, 1, -center.z],
        [0, 0, 0,         1],
    ])


# fmt: on


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

    perspective: Final = perspective_gen((eye - center).norm)
    model_view: Final = lookat(eye, center, up)
    persp_xform: Final = perspective @ model_view  # Compute outside the loop
    viewport: Final = viewport_gen(width // 16, height // 16, width * 7 // 8, height * 7 // 8)

    for fname in find_output.split():
        basename = Path(fname).name[:-4]
        try:
            framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
            z_buffer = ZBuffer(width=width, height=height)
            obj_data = OBJ_Data.from_file(fname)
            for face in obj_data.faces:
                clip: list[vec4] = []
                for entry in face.data:  # Assemble the primitive
                    v = obj_data.vertices[entry.vertex]
                    clip.append(vec4.from_np(persp_xform @ vec4.from_vec3(v, w=1).np))
                rasterize_lesson_6(clip, viewport, z_buffer, framebuffer, TGAColor_t.random())

            framebuffer.write_tga_file(f"{basename}.tga")
            z_buffer.to_tga(nan_zero=True).write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")
            raise RuntimeError from err

    return 0


if __name__ == "__main__":
    sys.exit(main())
