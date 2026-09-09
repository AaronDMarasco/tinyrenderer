from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final, Self

import numpy as np

import lib.our_gl as our_gl
from lib.model_v2 import ModelV2
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage
from lib.trtypes import Triangle, vec2, vec3, vec4

PLOT: Final[bool] = True

width: Final = 1024
height: Final = 1024

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector
sun: Final = vec3(1, 1, 1)  # Sun location

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Lesson9Shader(our_gl.IShader):
    def __init__(
        self: Self,
        model: ModelV2,
        *,
        sun: vec3,
        diffuse_weight: float = 1,
        specular_shine: int = 3,
    ) -> None:
        assert 0 <= diffuse_weight <= 1, "Diffuse term weight should be 0..1 inclusive"

        self.model: ModelV2 = model
        assert "diffuse" in self.model.ext
        assert "nm" in self.model.ext
        assert "spec" in self.model.ext
        self.color: TGAColor_t = TGAColor()
        # The list() wrapping is to keep type checkers happy
        self.varying_uv = list(
            vec2.new_zeros(3)
        )  # triangle uv coordinates, written by the vertex shader, read by the fragment shader
        self.varying_nrm = list(vec4.new_nans(3))  # normal per vertex to be interpolated by the fragment shader
        self.tri = list(vec4.new_nans(3))  # Triangle in eye coordinates
        self.sun_vector_l: vec4 = vec4.from_np(our_gl.model_view @ vec4.from_vec3(sun, w=0)).normalized
        self.diffuse_weight: float = diffuse_weight
        self.specular_shine: int = specular_shine

    def vertex(self: Self, face: int, vert: int) -> vec4:
        v: Final[vec3] = self.model.vert(face, vert)  # current vertex in object coordinates
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ vec4.from_vec3(v, w=1))
        self.varying_uv[vert] = self.model.vert_texture(face, vert)  # current texture U,V (as X,Y)
        self.varying_nrm[vert] = vec4.from_np(our_gl.model_view_IT @ self.model.normal(face, vert))
        self.tri[vert] = gl_position
        return vec4.from_np(our_gl.perspective @ gl_position)  # in clip coordinates

    def fragment(self: Self, bar: list[float]) -> tuple[bool, TGAColor_t]:
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
        color_sample: Final[vec2] = (
            self.varying_uv[0] * bar[0] + self.varying_uv[1] * bar[1] + self.varying_uv[2] * bar[2]
        )
        nm_t_color: Final[TGAColor_t] = self.model.ext_color("nm_tangent", color_sample)
        # CPP: return normalized(vec4{(double)c[2],(double)c[1],(double)c[0],0}*2./255. - vec4{1,1,1,0});
        nm_t_normal: Final[vec4] = (
            vec4(x=nm_t_color.r, y=nm_t_color.g, z=nm_t_color.b, w=0) * 2 / 255 - vec4(x=1, y=1, z=1, w=0)
        ).normalized
        normal_vector_n: Final[vec4] = vec4.from_np(np.transpose(D_matrix) @ nm_t_normal).normalized

        # Get the color from the "diffuse" file
        diff_color: Final[TGAColor_t] = self.model.ext_color("diffuse", color_sample) / 3  # Scale it by 1/3

        # Compute 0..1 for diffuse term
        diffuse_raw: Final[float] = self.sun_vector_l * normal_vector_n
        diffuse: Final = max(0, diffuse_raw)
        assert 0 <= diffuse <= 1, f"'{diffuse=}' should be 0..1 inclusive?"

        # Compute 0..1 for specular term
        vector_r: Final[vec3] = (normal_vector_n * diffuse_raw * 2 - self.sun_vector_l).normalized

        # specular intensity - note that the camera lies on the z-axis (in eye coordinates),
        # therefore simple r.z, since (0,0,1)*(r.x, r.y, r.z) = r.z
        specular: Final = pow(max(0, vector_r.z), self.specular_shine)
        assert 0 <= specular <= 1, f"'{specular=}' should be 0..1 inclusive?"
        # Get the color from the "spec" file
        spec_color: TGAColor_t = (self.model.ext_color("spec", color_sample) / 3) * specular

        if spec_color.bytespp == TGAImage.Format.GRAYSCALE:
            spec_color = TGAColor(spec_color.b, spec_color.b, spec_color.b, bpp=TGAImage.Format.RGB)

        # Weighted diffuse should be 0..1
        weighted_diffuse = diffuse * self.diffuse_weight
        assert 0 <= weighted_diffuse <= 2, f"Weighted diffused {weighted_diffuse} should be 0..1 inclusive?"
        final_scaled: Final[int] = round(weighted_diffuse * 85)

        # Now scale it to 255
        self.color = TGAColor(
            r=diff_color.r + spec_color.r + final_scaled,
            g=diff_color.g + spec_color.g + final_scaled,
            b=diff_color.b + spec_color.b + final_scaled,
        )

        return (False, self.color)  # do not discard the pixel


def main() -> int:

    find_output = """
../obj/african_head/african_head.obj
../obj/african_head/african_head_eye_inner.obj
../obj/african_head/african_head_eye_outer.obj
../obj/diablo3_pose/diablo3_pose.obj
"""
    our_gl.lookat(eye, center, up)  # build global model_view
    our_gl.init_perspective((eye - center).norm)  # build global persepctive
    our_gl.init_viewport(width // 16, height // 16, width * 7 // 8, height * 7 // 8)  # build global view_port

    for fname in find_output.split():
        basename = Path(fname).name[:-4]
        try:
            logger.debug("Processing %s...", basename)
            framebuffer = TGAImage(w=width, h=height, bpp=TGAImage.Format.RGB, c=TGAColor(127, 127, 127))
            our_gl.init_zbuffer(width, height)  # New zbuffer per image
            model = ModelV2.from_file(fname)
            shader = Lesson9Shader(model, sun=sun, specular_shine=35)
            logger.debug("Rendering %d faces...", len(model.faces))
            for face in range(len(model.faces)):
                if face and face % 250 == 0:
                    logger.debug("%d...", face)
                clip: Triangle = (  # assemble the primitive
                    shader.vertex(face, 0),
                    shader.vertex(face, 1),
                    shader.vertex(face, 2),
                )
                our_gl.rasterize(clip, shader, framebuffer)  # rasterize the primitive

            framebuffer.brighten()
            framebuffer.write_tga_file(f"{basename}.tga")
            our_gl.z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file(f"{basename}_z.tga")
            framebuffer.plot(PLOT)

        except Exception as err:
            print(f"Could not process {fname}: {err}")
            raise RuntimeError from err

    return 0


if __name__ == "__main__":
    sys.exit(main())
