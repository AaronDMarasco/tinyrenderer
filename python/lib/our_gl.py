from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final, Self

import numpy as np

from .trtypes import Matrix3f, Matrix4f, Triangle, ZBuffer, empty_matrix, vec2, vec3, vec4

if TYPE_CHECKING:
    from .tgaimage import TGAColor_t, TGAImage

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

    # We don't need this because our Model handles scaling automatically when using ext_color()
    # # @staticmethod
    # # def sample_2d(img: TGAImage, uvf: vec2) -> TGAColor_t:
    # #     return img.get(uvf.x * img.width, uvf.y * img.height)


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
    try:
        model_view_IT = np.linalg.inv(model_view.T)
    except np.linalg.LinAlgError:  # "Singular matrix" because the columns matched
        model_view_IT = np.linalg.pinv(model_view.T)


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


def init_zbuffer(width: int, height: int, *, init_val: float | None = None) -> None:
    global z_buffer
    z_buffer = ZBuffer(width=width, height=height)  # CPP inits to -1000
    if init_val is not None:
        z_buffer.fix_nan(init_val)


def rasterize(
    clip: Triangle,
    shader: IShader,
    framebuffer: TGAImage,
) -> None:
    ndc: Final[list[vec4]] = [
        clip[0] / (clip[0].w + 1e-9),
        clip[1] / (clip[1].w + 1e-9),
        clip[2] / (clip[2].w + 1e-9),
    ]  # normalized device coordinates
    screen: Final[list[vec2]] = [
        vec4.from_np(view_port @ ndc[0]).xy,
        vec4.from_np(view_port @ ndc[1]).xy,
        vec4.from_np(view_port @ ndc[2]).xy,
    ]  # screen coordinates
    ABC: Final[Matrix3f] = np.array(  # ruff: ignore[non-lowercase-variable-in-function]
        [
            [screen[0].x, screen[0].y, 1.0],
            [screen[1].x, screen[1].y, 1.0],
            [screen[2].x, screen[2].y, 1.0],
        ]
    )
    if np.linalg.det(ABC) < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    # bounding box for the triangle defined by its top left and bottom right corners but bound by canvas size
    bb_min_x: Final[int] = int(max(0, min(screen[0].x, screen[1].x, screen[2].x)))
    bb_max_x: Final[int] = int(min(framebuffer.width - 1, max(screen[0].x, screen[1].x, screen[2].x)))
    bb_min_y: Final[int] = int(max(0, min(screen[0].y, screen[1].y, screen[2].y)))
    bb_max_y: Final[int] = int(min(framebuffer.height - 1, max(screen[0].y, screen[1].y, screen[2].y)))

    ABC_invert_transpose = np.linalg.inv(ABC.T)  # ruff: ignore[non-lowercase-variable-in-function]

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            # bc = barycentric coordinates of {x,y} w.r.t the triangle
            bc_screen = vec3.from_np(ABC_invert_transpose @ [x, y, 1])
            if any(v < 0 for v in bc_screen.array):
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            # See check https://github.com/ssloy/tinyrenderer/wiki/Technical-difficulties-linear-interpolation-with-perspective-deformations
            # for bc_clip
            bc_clip_raw: tuple[float, ...] = (bc_screen.x / clip[0].w, bc_screen.y / clip[1].w, bc_screen.z / clip[2].w)
            bc_clip: vec3 = vec3(*bc_clip_raw) / sum(bc_clip_raw)
            z: float = (bc_screen.np @ [ndc[0].z, ndc[1].z, ndc[2].z]).item()
            # Make a claim to a ZBuffer value
            if not z_buffer.try_set(x, y, z):  # Behind what we've already drawn
                continue
            discard, color = shader.fragment(bc_clip.array)
            if discard:
                continue
            framebuffer.set(x, y, color)
