from __future__ import annotations

import copy
import logging
import math
import sys
from typing import Final, Self, override

import friendly
import numpy as np

from lib import our_gl
from lib.model_v2 import ModelV2
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage
from lib.trtypes import Matrix4f, Triangle, vec2, vec3, vec4

friendly.install()

PLOT: Final[bool] = False
DIE_ON_FAILURE: Final[bool] = True
CAMERA_POINTS: Final[int] = 5  # Is 1000 in CPP

width: Final = 480
height: Final = 480
shadow_w: Final = width * 5
shadow_h: Final = height * 5

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector
# light: Final = vec3(1, 1, 1)  # Light location

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class BlankShader(our_gl.IShader):
    def __init__(
        self: Self,
        model: ModelV2,
    ) -> None:
        self.model: ModelV2 = model

    def vertex(self: Self, face: int, vert: int) -> vec4:
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ self.model.vert(face, vert))
        return vec4.from_np(our_gl.perspective @ gl_position)  # in clip coordinates

    @override
    def fragment(self: Self, bar: list[float]) -> tuple[bool, TGAColor_t]:
        return (False, TGAColor(b=255, g=255, r=255, a=255))


class Lesson11Shader(our_gl.IShader):
    def __init__(
        self: Self,
        model: ModelV2,
        *,
        sun: vec3,
    ) -> None:
        self.model: ModelV2 = model
        assert "diffuse" in self.model.ext
        assert "nm_tangent" in self.model.ext
        assert "spec" in self.model.ext
        # The list() wrapping is to keep type checkers happy
        # varying_uv: triangle uv coordinates, written by the vertex shader, read by the fragment shader
        self.varying_uv = list(vec2.new_zeros(3))
        self.varying_nrm = list(vec4.new_nans(3))  # normal per vertex to be interpolated by the fragment shader
        self.tri = list(vec4.new_nans(3))  # Triangle in eye coordinates
        self.sun_vector_l: Final[vec4] = vec4.from_np(our_gl.model_view @ vec4.from_vec3(sun, w=0)).normalized

    def vertex(self: Self, face: int, vert: int) -> vec4:
        v: Final[vec4] = self.model.vert(face, vert)  # current vertex in object coordinates
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ v)
        self.varying_uv[vert] = self.model.vert_texture(face, vert)  # current texture U,V (as X,Y)
        self.varying_nrm[vert] = vec4.from_np(our_gl.model_view_IT @ self.model.normal(face, vert))
        self.tri[vert] = gl_position
        return vec4.from_np(our_gl.perspective @ gl_position)  # in clip coordinates

    @override
    def fragment(self: Self, bar: list[float]) -> tuple[bool, TGAColor_t]:
        # Previously had these as inputs... copied constants from CPP
        ambient: Final[float] = 0.4  # ambient light intensity (constant opied from CPP)
        specular_shine: Final = 35

        assert len(bar) == 3, f"Invalid {bar=}"
        # Our matrix types in trtypes are all square, so for now leave as native lists
        # (we'll see if this is a bad decision later or not)
        E_matrix: Final[list[vec4]] = [self.tri[1] - self.tri[0], self.tri[2] - self.tri[0]]
        U_matrix: Final[list[vec2]] = [self.varying_uv[1] - self.varying_uv[0], self.varying_uv[2] - self.varying_uv[0]]
        U_matrix_inv: np.ndarray[tuple[int, int], np.dtype[np.float64]]
        try:
            U_matrix_inv = np.linalg.inv(U_matrix)
        except np.linalg.LinAlgError:  # "Singular matrix" because the columns matched
            U_matrix_inv = np.linalg.pinv(U_matrix)
        T_matrix: Final[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = U_matrix_inv @ E_matrix
        D_matrix: Final[list[vec4]] = [
            vec4.from_np(T_matrix[0]).normalized,  # tangent vector
            vec4.from_np(T_matrix[1]).normalized,  # bitangent vector
            (
                self.varying_nrm[0] * bar[0] + self.varying_nrm[1] * bar[1] + self.varying_nrm[2] * bar[2]
            ).normalized,  # interpolated normal
            vec4(x=0, y=0, z=0, w=1),  # Darboux frame
        ]
        # In CPP, 'color_sample' called 'uv'
        color_sample: Final[vec2] = (
            self.varying_uv[0] * bar[0] + self.varying_uv[1] * bar[1] + self.varying_uv[2] * bar[2]
        )

        nm_t_color: Final[TGAColor_t] = self.model.ext_color("nm_tangent", color_sample)
        # CPP: model.normal => return normalized(vec4{(double)c[2],(double)c[1],(double)c[0],0}*2./255. - vec4{1,1,1,0})
        nm_t_normal: Final[vec4] = (
            vec4(x=nm_t_color.r, y=nm_t_color.g, z=nm_t_color.b, w=0) * 2 / 255 - vec4(x=1, y=1, z=1, w=0)
        ).normalized
        normal_vector_n: Final[vec4] = vec4.from_np(np.transpose(D_matrix) @ nm_t_normal).normalized
        diffuse_raw: Final[float] = self.sun_vector_l * normal_vector_n
        vector_r: Final[vec4] = (
            normal_vector_n * diffuse_raw * 2 - self.sun_vector_l
        ).normalized  # reflected light direction
        diffuse: Final[float] = max(0, diffuse_raw)

        # Get the color from the "spec" file
        spec_color: Final[TGAColor_t] = self.model.ext_color("spec", color_sample)

        # specular intensity - note that the camera lies on the z-axis (in eye coordinates),
        # therefore simple r.z, since (0,0,1)*(r.x, r.y, r.z) = r.z
        specular: Final = (1 + 3 * spec_color.b / 255) * pow(max(0, vector_r.z), specular_shine)
        # Get the color from the "diffuse" file
        # (called 'gl_FragColor' in CPP)
        diff_color: Final[TGAColor_t] = self.model.ext_color("diffuse", color_sample)

        # Now scale it - should be <= 1 but TGAColor doesn't care
        final_scaling: Final[float] = ambient + diffuse + specular

        return (False, diff_color * final_scaling)  # do not discard the pixel


def smooth_step(edge_0: float, edge_1: float, x: float) -> float:
    """
    smoothstep returns 0 if the input is less than the left edge, 1 if the input is greater than the right edge, Hermite
    interpolation in between. The derivative of the smoothstep function is zero at both edges.
    """
    t: Final[float] = np.clip((x - edge_0) / (edge_1 - edge_0), a_min=0, a_max=1)
    return t * t * (3 - 2 * t)


def main() -> int:  # ruff: ignore[too-many-branches, too-many-statements]
    # Going to merge all these files into a single output at the end
    input_files = """
../obj/floor.obj
../obj/diablo3_pose/diablo3_pose.obj
"""
    models: dict[str, ModelV2] = {}
    # First pass - same as Lesson 9
    our_gl.lookat(eye, center, up)  # build global model_view
    our_gl.init_perspective((eye - center).norm)  # build global perspective
    our_gl.init_viewport(width // 16, height // 16, width * 7 // 8, height * 7 // 8)  # build global view_port

    mono_black: Final = TGAColor(0, bpp=1)
    mono_white: Final = TGAColor(255, bpp=1)

    framebuffer = TGAImage(w=width, h=height, bpp=TGAImage.Format.RGBA, c=TGAColor(177, 195, 209, 255))
    our_gl.init_zbuffer(width, height, init_val=-1000)

    for fname in input_files.split():
        try:
            logger.debug("Processing %s...", fname)
            if fname not in models:
                models[fname] = ModelV2.from_file(fname)
            model = models[fname]
            shader = BlankShader(model)
            logger.debug("Rendering %d faces...", len(model.faces))
            for face, _ in enumerate(model.faces):
                if face and face % 250 == 0:
                    logger.debug("%d...", face)
                clip: Triangle = (  # assemble the primitive
                    shader.vertex(face, 0),
                    shader.vertex(face, 1),
                    shader.vertex(face, 2),
                )
                our_gl.rasterize(clip, shader, framebuffer)  # rasterize the primitive
        except Exception as err:
            logger.exception("Could not process %s", fname)
            if DIE_ON_FAILURE:
                raise RuntimeError from err

    framebuffer.write_tga_file("framebuffer.tga")
    framebuffer.plot(PLOT)

    zbuffer_copy: Final = copy.deepcopy(our_gl.z_buffer)
    m_matrix: Final[Matrix4f] = np.linalg.inv(our_gl.view_port @ our_gl.perspective @ our_gl.model_view)
    # Brute-force ambient
    rng = np.random.default_rng()
    for i in range(CAMERA_POINTS):
        # if i % 50 == 0:
        logger.debug("Camera point %d/%d", i, CAMERA_POINTS)
        y: float = rng.normal()
        theta: float = 2 * math.pi * rng.normal()
        logger.debug(f"{y=}, {(1 - y * y)=}")
        r: float = math.sqrt(max(0, 1 - y * y))
        light: vec3 = vec3(x=r * math.cos(theta), y=r * math.sin(theta), z=0) * 1.5
        logger.debug("v %s", light)
        our_gl.lookat(light, center, up)  # build global model_view
        our_gl.model_view[3][3] = (light - center).norm
        our_gl.init_perspective((eye - center).norm)  # build global perspective
        our_gl.init_viewport(
            shadow_w // 16, shadow_h // 16, shadow_w * 7 // 8, shadow_h * 7 // 8
        )  # build global view_port
        our_gl.init_zbuffer(shadow_w, shadow_h, init_val=-1000)
        trash = TGAImage(w=shadow_w, h=shadow_h, bpp=TGAImage.Format.RGBA, c=TGAColor(177, 195, 209, 255))

        for fname in input_files.split():
            try:
                model = models[fname]
                shader = BlankShader(model)
                for face, _ in enumerate(model.faces):
                    if face and face % 250 == 0:
                        logger.debug("%d...", face)
                    clip = (  # assemble the primitive
                        shader.vertex(face, 0),
                        shader.vertex(face, 1),
                        shader.vertex(face, 2),
                    )
                    our_gl.rasterize(clip, shader, trash)  # rasterize the primitive
            except Exception as err:
                logger.exception("Could not process %s", fname)
                if DIE_ON_FAILURE:
                    raise RuntimeError from err

        n_matrix: Matrix4f = our_gl.view_port @ our_gl.perspective @ our_gl.model_view

        logger.debug("Post-processing")
        mask: np.ndarray = np.zeros(shape=(width, height), dtype=bool)
        # # assert not any(mask.ravel())

        for x in range(width):
            if x and x % 100 == 0:
                logger.debug("%d/%d...", x, width)
            for y in range(height):
                fragment: vec4 = vec4.from_np(m_matrix @ vec4(x, y, zbuffer_copy[x][y], 1))
                q: vec4 = vec4.from_np(n_matrix @ fragment)
                p: vec3 = q.xyz / (q.w + 1e-9)
                # logger.debug(f"{x=} {y=} {zbuffer_copy[x][y]=} {q=} {p=} {fragment=}")
                lit: bool = (
                    (fragment.z < -100)  # Background
                    or ((0 <= p.x < shadow_w) and (0 <= p.y < shadow_h))  # not out of bounds of the shadow buffer
                    or (p.z > (our_gl.z_buffer[round(p.x)][round(p.y)] - 0.03))  # it is visible
                )
                mask[x][y] = lit

        logger.debug("Wrote %d masks", width * height)
        mask_img = TGAImage(w=width, h=height, bpp=TGAImage.Format.GRAYSCALE)
        for x in range(width):
            for y in range(height):
                mask_img.set(x, y, mono_black if mask[x][y] else mono_white)

        mask_img.write_tga_file("mask.tga")
        mask_img.plot(PLOT)

    logger.debug("Smoothstep")
    for x in range(width):
        if x and x % 100 == 0:
            logger.debug("%d/%d...", x, width)
        for y in range(height):
            m: float = smooth_step(-1, 1, mask[x][y])
            c: TGAColor_t = framebuffer.get(x, y)
            # Cannot use scaling here because we want alpha left alone...
            framebuffer.set(x, y, TGAColor(g=round(c.g * m), b=round(c.b * m), r=round(c.r * m), a=c.a))
    framebuffer.write_tga_file("shadow.tga")

    threshold: float = 0.15
    Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    logger.debug("Post-processing 2: edge detect")
    for y in range(framebuffer.height):
        for x in range(framebuffer.width):
            sum_ = vec2(x=0, y=0)
            for j in range(-1, 2):
                for i in range(-1, 2):
                    sum_ += vec2(x=Gx[j + 1][i + 1] * zbuffer_copy[x][y], y=Gy[j + 1][i + 1] * zbuffer_copy[x][y])
            if sum_.norm > threshold:
                framebuffer.set(x, y, TGAColor(0, 0, 0, 255))  # L147
    framebuffer.write_tga_file("edges.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
