from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final, Self

import numpy as np

from .tgaimage import TGAColor_t, TGAImage
from .trtypes import Matrix3f, Matrix4f, Triangle, ZBuffer, empty_matrix, vec2, vec3, vec4

# Global module-level state variables (ugh):
model_view: Matrix4f = empty_matrix(4)
model_view_IT: Matrix4f = empty_matrix(4)  # Inverse/Transposed version
view_port: Matrix4f = empty_matrix(4)
perspective: Matrix4f = empty_matrix(4)
z_buffer: ZBuffer = ZBuffer(width=1, height=1)

# The fragment 'bar' parameter is the barycentric coordinates of {x,y,z} w.r.t the triangle
# and NOT a "real" XYZ, but also represented as a 'vec3' in the C++, so instead in python they
# they will just be list[float]


class IShader(ABC):
    @abstractmethod
    def fragment(self: Self, bar: list[float]) -> tuple[bool, TGAColor_t]: ...


def lookat(eye: vec3, center: vec3, up: vec3) -> None:
    # See https://haqr.eu/tinyrenderer/camera/ for vector naming
    n_vec = (eye - center).normalized
    l_vec = up.cross(n_vec).normalized
    m_vec = n_vec.cross(l_vec).normalized
    global model_view
    # fmt: off
    # ruff: ignore[multiple-spaces-after-comma,whitespace-after-open-bracket]
    model_view = np.array([
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
    global model_view_IT
    model_view_IT = np.linalg.inv(model_view.T)


def init_perspective(f: float) -> None:
    global perspective
    # fmt: off
    # ruff: ignore[multiple-spaces-after-comma,missing-whitespace-around-arithmetic-operator]
    perspective = np.array([
        [1, 0,    0,  0],
        [0, 1,    0,  0],
        [0, 0,    1,  0],
        [0, 0, -1/f,  1],
    ])


# fmt: on


def init_viewport(x: int, y: int, width: int, height: int) -> None:
    h_2: Final = height / 2
    w_2: Final = width / 2
    global view_port
    # fmt: off
    # ruff: ignore[multiple-spaces-after-comma,whitespace-after-open-bracket]
    view_port = np.array([
        [w_2,   0,  0, x + w_2],
        [  0, h_2,  0, y + h_2],
        [  0,   0,  1,       0],
        [  0,   0,  0,       1],
    ])


# fmt: on


def init_zbuffer(width: int, height: int) -> None:
    global z_buffer
    z_buffer = ZBuffer(width=width, height=height)


def rasterize(
    clip: Triangle,
    shader: IShader,
    framebuffer: TGAImage,
) -> None:
    ndc: Final[list[vec4]] = [
        clip[0] / clip[0].w,
        clip[1] / clip[1].w,
        clip[2] / clip[2].w,
    ]  # normalized device coordinates
    screen: Final[list[vec2]] = [
        vec4.from_np(view_port @ ndc[0]).xy,
        vec4.from_np(view_port @ ndc[1]).xy,
        vec4.from_np(view_port @ ndc[2]).xy,
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
            bc = ABC_invert_transpose @ [x, y, 1]
            if any(v < 0 for v in bc):
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z: float = (bc @ [ndc[0].z, ndc[1].z, ndc[2].z]).item()
            if z <= z_buffer.vals[x][y]:  # Behind what we've already drawn
                continue
            discard, color = shader.fragment(bc.tolist())
            if discard:
                continue
            z_buffer.vals[x][y] = z
            framebuffer.set(x, y, color)
