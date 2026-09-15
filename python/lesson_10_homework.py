from __future__ import annotations

import copy
import logging
import sys
from typing import Final, Self, override

import numpy as np

from lib import our_gl
from lib.model_v2 import ModelV2
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage
from lib.trtypes import Matrix4f, Triangle, vec2, vec3, vec4

PLOT: Final[bool] = True
DIE_ON_FAILURE: Final[bool] = True

width: Final = 320
height: Final = 320
shadow_w: Final = width * 5
shadow_h: Final = height * 5

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector
light: Final = vec3(1, 1, 1)  # Light location

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


class Lesson10Shader(our_gl.IShader):
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


def main() -> int:  # ruff: ignore[too-many-branches, too-many-statements]
    # Going to merge all these files into a single output at the end
    input_files = """
../obj/floor.obj
../obj/diablo3_pose/diablo3_pose.obj
"""
    # First pass - same as Lesson 9
    our_gl.lookat(eye, center, up)  # build global model_view
    our_gl.init_perspective((eye - center).norm)  # build global perspective
    our_gl.init_viewport(width // 16, height // 16, width * 7 // 8, height * 7 // 8)  # build global view_port

    framebuffer = TGAImage(w=width, h=height, bpp=TGAImage.Format.RGBA, c=TGAColor(177, 195, 209, 255))
    our_gl.init_zbuffer(width, height)

    for fname in input_files.split():
        try:
            logger.debug("Processing %s...", fname)
            model = ModelV2.from_file(fname)
            shader = Lesson10Shader(model, sun=light)
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

    # New shading part
    # our_gl.z_buffer.fix_nan(-1000)
    our_gl.z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file("zbuffer1.tga")
    zbuffer_copy: Final = copy.deepcopy(our_gl.z_buffer)
    m_matrix: Final[Matrix4f] = np.linalg.inv(our_gl.view_port @ our_gl.perspective @ our_gl.model_view)

    our_gl.lookat(light, center, up)  # build global model_view
    our_gl.init_perspective((eye - center).norm)  # build global perspective
    our_gl.init_viewport(shadow_w // 16, shadow_h // 16, shadow_w * 7 // 8, shadow_h * 7 // 8)  # build global view_port

    trash = TGAImage(w=shadow_w, h=shadow_h, bpp=TGAImage.Format.RGBA, c=TGAColor(177, 195, 209, 255))
    our_gl.init_zbuffer(shadow_w, shadow_h)

    for fname in input_files.split():
        try:
            logger.debug("(Shadow) Processing %s...", fname)
            model = ModelV2.from_file(fname)
            b_shader = BlankShader(model)
            logger.debug("(Shadow) Rendering %d faces...", len(model.faces))
            for face, _ in enumerate(model.faces):
                if face and face % 250 == 0:
                    logger.debug("(Shadow) %d...", face)
                clip = (  # assemble the primitive
                    b_shader.vertex(face, 0),
                    b_shader.vertex(face, 1),
                    b_shader.vertex(face, 2),
                )
                our_gl.rasterize(clip, b_shader, trash)
        except Exception as err:
            logger.exception("Could not process %s", fname)
            if DIE_ON_FAILURE:
                raise RuntimeError from err

    trash.write_tga_file("shadowmap.tga")
    trash.plot(PLOT)
    our_gl.z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file("zbuffer2.tga")

    n_matrix: Final[Matrix4f] = our_gl.view_port @ our_gl.perspective @ our_gl.model_view

    logger.debug("Post-processing")
    mask: np.ndarray = np.zeros(shape=(width, height), dtype=bool)
    # assert not any(mask.ravel())

    for x in range(width):
        if x and x % 100 == 0:
            logger.debug("%d/%d...", x, width)
        for y in range(height):
            fragment: vec4 = vec4.from_np(m_matrix @ vec4(x, y, zbuffer_copy[x][y], 1))
            q: vec4 = vec4.from_np(n_matrix @ fragment)
            p: vec3 = q.xyz / q.w
            lit: bool = (
                (fragment.z < -100)  # Background
                or (not (0 < p.x <= shadow_w) or not (0 < p.y <= shadow_h))  # out of bounds of shadow buffer
                or (p.z > (our_gl.z_buffer[round(p.x)][round(p.y)] - 0.03))  # it is visible
            )
            mask[x][y] = lit

    logger.debug("Wrote %d masks", width * height)
    mask_img = TGAImage(w=width, h=height, bpp=TGAImage.Format.GRAYSCALE)
    mono_black: Final = TGAColor(0, bpp=1)
    mono_white: Final = TGAColor(255, bpp=1)
    for x in range(width):
        for y in range(height):
            mask_img.set(x, y, mono_black if mask[x][y] else mono_white)

    mask_img.write_tga_file("mask.tga")
    mask_img.plot(PLOT)

    for x in range(width):
        for y in range(height):
            if mask[x][y]:
                continue
            c = framebuffer.get(x, y)
            a: vec3 = vec3(x=c.b, y=c.g, z=c.r)
            if a.norm < 80:
                continue
            a = a.normalized * 80
            framebuffer.set(x, y, TGAColor(round(a.x), round(a.y), round(a.z), 255))

    framebuffer.write_tga_file("output.tga")
    framebuffer.plot(PLOT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
