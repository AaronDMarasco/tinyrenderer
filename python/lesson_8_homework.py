from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Self

import lib.our_gl as our_gl
from lib.model import Model
from lib.tgaimage import TGAColor, TGAColor_t, TGAImage
from lib.trtypes import Triangle, vec3, vec4

width: Final = 800
height: Final = 800

eye: Final = vec3(-1, 0, 2)  # Camera position
center: Final = vec3(0, 0, 0)  # Camera direction
up: Final = vec3(0, 1, 0)  # Camera up vector
sun: Final = vec3(1, 1, 1)  # Sun location


class PhongShader(our_gl.IShader):
    model: Model
    color: TGAColor_t
    tri: list[vec3]  # Triangle in eye coordinates
    vns: list[vec3]  # Vector Normal for each vertex of triangle
    # These are for reflection stuff:
    sun_vector_l: vec3  # light direction in eye coordinates
    ambient: float
    diffuse_weight: float
    specular_shine: int

    def __init__(
        self: Self,
        model: Model,
        *,
        sun: vec3,
        ambient: float = 0.4,
        diffuse_weight: float = 1,
        specular_weight: float = 1,
        specular_shine: int = 3,
    ) -> None:
        assert 0 <= ambient <= 1, "Ambient should be 0..1 inclusive"
        assert 0 <= diffuse_weight <= 1, "Diffuse term weight should be 0..1 inclusive"
        assert 0 <= specular_weight <= 1, "Specular term weight should be 0..1 inclusive"

        self.model = model
        self.color = TGAColor()
        self.tri = [vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0)]
        self.vns = [vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0), vec3(x=0, y=0, z=0)]
        self.sun_vector_l = vec4.from_np(our_gl.model_view @ vec4.from_vec3(sun, w=0)).normalized.xyz
        self.ambient = ambient
        self.diffuse_weight = diffuse_weight
        self.specular_weight = specular_weight
        self.specular_shine = specular_shine

    def vertex(self: Self, face: int, vert: int) -> vec4:
        v: Final[vec3] = self.model.vert(face, vert)  # current vertex in object coordinates
        vn: Final[vec3] = self.model.normal(face, vert)  # current normal vertex in object coordinates
        # This is explained in the "caveat" section of lesson 8:
        # MV_invert_transpose: Final = np.linalg.inv(our_gl.model_view.T)
        # gl_position_vn: Final[vec4] = vec4.from_np(MV_invert_transpose @ vec4.from_vec3(vn, w=0))
        self.vns[vert] = vec4.from_np(our_gl.model_view_IT @ vec4.from_vec3(vn, w=0)).xyz  # in eye coordinates
        gl_position: Final[vec4] = vec4.from_np(our_gl.model_view @ vec4.from_vec3(v, w=1))
        self.tri[vert] = gl_position.xyz  # in eye coordinates
        return vec4.from_np(our_gl.perspective @ gl_position)  # in clip coordinates

    def fragment(self: Self, bar: list[float]) -> tuple[bool, TGAColor_t]:
        assert len(bar) == 3, f"Invalid {bar=}"
        # In Lesson 7, we computed a normal to the entire triangle. Now we use provided ones (self.vns)
        normal_vector_n: Final[vec3] = (self.vns[0] * bar[0] + self.vns[1] * bar[1] + self.vns[2] * bar[2]).normalized

        # Compute 0..1 for diffuse term
        diffuse_raw: Final[float] = self.sun_vector_l * normal_vector_n  # Math Confirmed
        diffuse: Final = max(0, diffuse_raw)
        assert 0 <= diffuse <= 1, f"'{diffuse=}' should be 0..1 inclusive?"

        # Compute 0..1 for specular term
        vector_r: Final[vec3] = (normal_vector_n * diffuse_raw * 2 - self.sun_vector_l).normalized
        # specular intensity - note that the camera lies on the z-axis (in eye coordinates),
        # therefore simple r.z, since (0,0,1)*(r.x, r.y, r.z) = r.z
        specular: Final = pow(max(0, vector_r.z), self.specular_shine)
        assert 0 <= specular <= 1, f"'{specular=}' should be 0..1 inclusive?"

        # Now combine them (should be 0..3)
        final = self.ambient + diffuse * self.diffuse_weight + specular * self.specular_weight
        assert 0 <= final <= 3, f"Final color {final=} should be 0..3 inclusive?"

        # Now scale it to 255
        self.color = TGAColor(round(final * 85))

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
            framebuffer = TGAImage(width, height, TGAImage.Format.GRAYSCALE, TGAColor(255 // 2))
            our_gl.init_zbuffer(width, height)  # New zbuffer per image
            model = Model.from_file(fname)
            shader = PhongShader(model, sun=sun, specular_shine=35)
            for face in range(len(model.faces)):
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
