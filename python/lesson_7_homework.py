from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Final, Self

import numpy as np

import lib.our_gl as our_gl
from lib.objreader import OBJ_Data
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage
from lib.trtypes import Triangle, vec3, vec4

width: Final = 800
height: Final = 800

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0.5, 0.5, 0)  # Camera up vector
sun: Final = vec3(0, -0, 100)  # Sun location


class PhongShader(our_gl.IShader):
    model: OBJ_Data
    color: TGAColor_t
    tri: list[vec3]  # Triangle in eye coordinates

    def __init__(self: Self, model: OBJ_Data) -> None:
        self.model = model
        self.color = TGAColor()
        self.tri = [vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0)]

    def vertex(self: Self, face: int, vert: int) -> vec4:
        v: Final[vec3] = self.model.vert(face, vert)  # current vertex in object coordinates
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ vec4.from_vec3(v, w=1))
        self.tri[vert] = gl_position.xyz  # in eye coordinates
        return vec4.from_np(our_gl.perspective @ gl_position)  # in clip coordinates

    def update_color(
        self: Self,
        face: int,
        *,
        sun: vec3,
        ambient: float = 0.5,
        diffuse_weight: float = 1.0,
        specular_weight: float = 1.0,
        specular_shine: int = 3,
    ) -> None:
        """Update color based on face"""
        assert 0 <= ambient <= 1, "Ambient should be 0..1 inclusive"
        assert 0 <= diffuse_weight <= 1, "Diffuse term weight should be 0..1 inclusive"
        assert 0 <= specular_weight <= 1, "Specular term weight should be 0..1 inclusive"

        # To get an orthogonal vector you need two vectors on the plane...
        this_plane_points: Final[tuple[vec3, ...]] = tuple(self.model.vert(face, i) for i in range(3))
        this_plane_vectors: Final[tuple[vec3, vec3]] = (
            this_plane_points[1] - this_plane_points[0],
            this_plane_points[2] - this_plane_points[0],
        )
        orthogonal_vector: Final[vec3] = vec3.from_np(np.cross(*this_plane_vectors)).normalized

        # Is this needed???
        # sun_position: Final = vec4.from_np(our_gl.model_view @ vec4.from_vec3(sun, w=1)).normalized
        sun_position = vec4.from_vec3(sun.normalized, w=1)

        # Compute 0..1 for diffuse term
        diffuse: Final = max(0, sun_position.xyz * orthogonal_vector)
        alpha: Final[float] = math.acos(diffuse)
        assert 0 <= diffuse <= 1, "'diffuse' should be 0..1 inclusive?"

        # Compute 0..1 for specular term
        alpha_plus_beta: Final[float] = math.acos(eye.normalized * orthogonal_vector)
        beta: Final = alpha_plus_beta - alpha
        specular: Final = pow(max(0, math.cos(beta)), specular_shine)
        assert 0 <= specular <= 1, "'specular' should be 0..1 inclusive?"

        # Now combine them (should be 0..3)
        final = ambient + diffuse * diffuse_weight + specular * specular_weight
        assert 0 <= final <= 3, "Final color should be 0..3 inclusive?"

        # Now scale it to 255
        self.color = TGAColor(round(final * 85))

    def fragment(self: Self, _bar: vec3) -> tuple[bool, TGAColor_t]:
        return (False, self.color)  # do not discard the pixel


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
    find_output = """
../obj/african_head/african_head.obj
../obj/diablo3_pose/diablo3_pose.obj
"""
    our_gl.lookat(eye, center, up)  # build global model_view
    our_gl.init_perspective((eye - center).norm)  # build global persepctive
    our_gl.init_viewport(width // 16, height // 16, width * 7 // 8, height * 7 // 8)  # build global view_port

    for fname in find_output.split():
        basename = Path(fname).name[:-4]
        try:
            framebuffer = TGAImage(width, height, TGAImage.Format.GRAYSCALE, TGAColor(255 // 2))
            our_gl.init_zbuffer(width, height)  # New zbuffer per image
            model = OBJ_Data.from_file(fname)
            shader = PhongShader(model)
            for face in range(len(model.faces)):
                # shader.color = TGAColor_t.random()
                clip: Triangle = (  # assemble the primitive
                    shader.vertex(face, 0),
                    shader.vertex(face, 1),
                    shader.vertex(face, 2),
                )
                shader.update_color(face, sun=sun, specular_shine=64)
                our_gl.rasterize(clip, shader, framebuffer)  # rasterize the primitive

            framebuffer.write_tga_file(f"{basename}.tga")
            our_gl.z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")
            raise RuntimeError from err

    return 0


if __name__ == "__main__":
    sys.exit(main())
