from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Self

import lib.our_gl as our_gl
from lib.objreader import OBJ_Data
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage
from lib.trtypes import Triangle, vec3, vec4

width: Final = 800
height: Final = 800

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector


class RandomShader(our_gl.IShader):
    model: OBJ_Data
    color: TGAColor_t
    tri: list[vec3]  # Triangle in eye coordinates

    def __init__(self: Self, model: OBJ_Data) -> None:
        self.model = model
        self.color = TGAColor()
        self.tri = [vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0)]

    def vertex(self: Self, face: int, vert: int) -> vec4:
        v: Final[vec3] = self.model.vert(face, vert)  # current vertex in object coordinates
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ vec4.from_vec3(v, w=1).np)
        self.tri[vert] = gl_position.xyz  # in eye coordinates
        return vec4.from_np(our_gl.perspective @ gl_position.np)  # in clip coordinates

    def fragment(self: Self, _bar: list[float]) -> tuple[bool, TGAColor_t]:
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
    our_gl.lookat(eye, center, up)  # build global model_view
    our_gl.init_perspective((eye - center).norm)  # build global persepctive
    our_gl.init_viewport(width // 16, height // 16, width * 7 // 8, height * 7 // 8)  # build global view_port

    for fname in find_output.split():
        basename = Path(fname).name[:-4]
        try:
            framebuffer = TGAImage(width, height, TGAImage.Format.RGB, TGAColor(177, 195, 209))
            our_gl.init_zbuffer(width, height)  # New zbuffer per image
            model = OBJ_Data.from_file(fname)
            shader = RandomShader(model)
            for face in range(len(model.faces)):
                shader.color = TGAColor_t.random()
                clip: Triangle = (  # assemble the primitive
                    shader.vertex(face, 0),
                    shader.vertex(face, 1),
                    shader.vertex(face, 2),
                )
                our_gl.rasterize(clip, shader, framebuffer)  # rasterize the primitive

            framebuffer.write_tga_file(f"{basename}.tga")
            our_gl.z_buffer.to_tga(allow_nan=True, nan_val=0).write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")
            raise RuntimeError from err

    return 0


if __name__ == "__main__":
    sys.exit(main())
