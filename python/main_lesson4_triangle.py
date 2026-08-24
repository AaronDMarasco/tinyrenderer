from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np

from lib.objreader import OBJ_Data
from lib.render import triangle_barycentric_lesson_4
from lib.tgaimage import TGAColor, TGAImage

# white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
# green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
# red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
# blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
# yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)


def main() -> int:
    width: Final = 2047
    height: Final = 2048

    # The numbers go from -1..1 so we need to map them from the center of the image...
    width_center: Final[int] = width // 2
    height_center: Final[int] = height // 2

    rng = np.random.default_rng()  # seed=42)

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
    for fname in find_output.split():
        basename = Path(fname).name[:-4]
        try:
            framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
            z_buffer = TGAImage(width, height, TGAImage.Format.GRAYSCALE)
            obj_data = OBJ_Data.from_file(fname)
            for face in obj_data.faces:
                idx = (face[0].vertex, face[1].vertex, face[2].vertex)
                # Read those out
                points = (obj_data.vertices[idx[0]], obj_data.vertices[idx[1]], obj_data.vertices[idx[2]])
                # Fix quadrant and scaling
                a = (
                    round((points[0].x - 1) * width_center),
                    round((points[0].y - 1) * height_center),
                    round(127 * (points[0].z + 1)),
                )
                b = (
                    round((points[1].x - 1) * width_center),
                    round((points[1].y - 1) * height_center),
                    round(127 * (points[1].z + 1)),
                )
                c = (
                    round((points[2].x - 1) * width_center),
                    round((points[2].y - 1) * height_center),
                    round(127 * (points[2].z + 1)),
                )

                triangle_barycentric_lesson_4(
                    a,
                    b,
                    c,
                    z_buffer,
                    framebuffer,
                    # NEW: TGAColor_t.random(),
                    TGAColor(int(rng.integers(255)), int(rng.integers(255)), int(rng.integers(255))),
                )

            framebuffer.write_tga_file(f"{basename}.tga")
            z_buffer.write_tga_file(f"{basename}_z.tga")

        except Exception as err:
            print(f"Could not process {fname}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
