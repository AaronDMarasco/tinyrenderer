from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final, Self

import lib.our_gl as our_gl
from lib.model_v2 import ModelV2
from lib.plot import plot
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


class PhongNormalMappingShader(our_gl.IShader):
    model: ModelV2
    color: TGAColor_t
    vts: list[vec2]  # Vector texture U, V
    # These are for reflection stuff:
    sun_vector_l: vec4  # light direction in eye coordinates
    diffuse_weight: float
    specular_shine: int

    def __init__(
        self: Self,
        model: ModelV2,
        *,
        sun: vec3,
        diffuse_weight: float = 1,
        specular_shine: int = 3,
    ) -> None:
        assert 0 <= diffuse_weight <= 1, "Diffuse term weight should be 0..1 inclusive"

        self.model = model
        assert "diffuse" in self.model.ext
        assert "nm" in self.model.ext
        assert "spec" in self.model.ext
        self.color = TGAColor()
        self.vts = [vec2(x=0, y=0), vec2(x=0, y=0), vec2(x=0, y=0)]
        self.sun_vector_l = vec4.from_np(our_gl.model_view @ vec4.from_vec3(sun, w=0)).normalized
        self.diffuse_weight = diffuse_weight
        self.specular_shine = specular_shine

    def vertex(self: Self, face: int, vert: int) -> vec4:
        v: Final[vec3] = self.model.vert(face, vert)  # current vertex in object coordinates
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ vec4.from_vec3(v, w=1))
        self.vts[vert] = self.model.vert_texture(face, vert)  # current texture U,V (as X,Y)
        return vec4.from_np(our_gl.perspective @ gl_position)  # in clip coordinates

    def fragment(self: Self, bar: list[float]) -> tuple[bool, TGAColor_t]:
        assert len(bar) == 3, f"Invalid {bar=}"
        # For homework 2, we'll read the model's nm instead of the vn from the model
        color_sample: Final[vec2] = self.vts[0] * bar[0] + self.vts[1] * bar[1] + self.vts[2] * bar[2]
        nm_color: Final[TGAColor_t] = self.model.ext_color("nm", color_sample)
        normal_vector_n: Final[vec3] = vec3(x=nm_color.r / 255, y=nm_color.g / 255, z=nm_color.b / 255).normalized

        # Get the color from the "diffuse" file
        diff_color: Final[TGAColor_t] = self.model.ext_color("diffuse", color_sample) / 3  # Scale it by 1/3

        # Compute 0..1 for diffuse term
        diffuse_raw: Final[float] = self.sun_vector_l.xyz * normal_vector_n
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
            shader = PhongNormalMappingShader(model, sun=sun, specular_shine=35)
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

            framebuffer.write_tga_file(f"{basename}.tga")
            our_gl.z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file(f"{basename}_z.tga")
            plot(framebuffer, PLOT)

        except Exception as err:
            print(f"Could not process {fname}: {err}")
            raise RuntimeError from err

    return 0


if __name__ == "__main__":
    sys.exit(main())
