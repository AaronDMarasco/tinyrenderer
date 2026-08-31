from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np

from lib.model import OBJ_Data
from lib.tgaimage import TGAColor_t, TGAImage
from lib.trtypes import Matrix3f, Matrix4f, ZBuffer, vec2, vec3, vec4

width: Final = 800
height: Final = 800

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector


def rasterize_lesson_6(
    clip: Sequence[vec4],
    viewport: Matrix4f,
    z_buffer: ZBuffer,
    framebuffer: TGAImage,
    color: TGAColor_t,
) -> None:
    ndc: Final[list[vec4]] = [
        clip[0] / clip[0].w,
        clip[1] / clip[1].w,
        clip[2] / clip[2].w,
    ]  # normalized device coordinates
    screen: Final[list[vec2]] = [
        vec4.from_np(viewport @ ndc[0].np).xy,
        vec4.from_np(viewport @ ndc[1].np).xy,
        vec4.from_np(viewport @ ndc[2].np).xy,
    ]  # screen coordinates
    ABC: Final[Matrix3f] = np.array([
        [screen[0].x, screen[0].y, 1.0],
        [screen[1].x, screen[1].y, 1.0],
        [screen[2].x, screen[2].y, 1.0],
    ])
    if np.linalg.det(ABC) < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    # bounding box for the triangle defined by its top left and bottom right corners but bound by canvas size
    bb_min_x: Final[int] = int(max(0, min(screen[0].x, screen[1].x, screen[2].x)))
    bb_max_x: Final[int] = int(min(framebuffer.width - 1, max(screen[0].x, screen[1].x, screen[2].x)))
    bb_min_y: Final[int] = int(max(0, min(screen[0].y, screen[1].y, screen[2].y)))
    bb_max_y: Final[int] = int(min(framebuffer.height - 1, max(screen[0].y, screen[1].y, screen[2].y)))

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            ABC_invert_transpose = np.linalg.inv(ABC.T)
            # bc = barycentric coordinates of {x,y} w.r.t the triangle
            bc: vec3 = vec3.from_np(ABC_invert_transpose @ vec3(x, y, 1).np)
            if bc.x < 0 or bc.y < 0 or bc.z < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z = bc * vec3(ndc[0].z, ndc[1].z, ndc[2].z)
            if z <= z_buffer.vals[x][y]:  # Behind what we've already drawn
                continue
            z_buffer.vals[x][y] = z
            framebuffer.set(x, y, color)


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
            z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")
            raise RuntimeError from err

    return 0


if __name__ == "__main__":
    sys.exit(main())
