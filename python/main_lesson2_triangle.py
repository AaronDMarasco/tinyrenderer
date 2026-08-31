from __future__ import annotations

# import logging
import sys
from pathlib import Path
from typing import Final

from lesson_2_triangles import triangle
from lib.model import Model
from lib.tgaimage import TGAColor_t, TGAImage, uint8_t

# logging.getLogger('lib.model').setLevel(logging.INFO)


def main() -> int:
    width: Final = 2047
    height: Final = 2048

    # The numbers go from -1..1 so we need to map them from the center of the image...
    width_center: Final[int] = width // 2 - 1
    height_center: Final[int] = height // 2 - 1

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
            obj_data = Model.from_file(fname)
            for face in obj_data.faces:
                idx = (face[0].vertex, face[1].vertex, face[2].vertex)
                # Read those out
                points = (obj_data.vertices[idx[0]], obj_data.vertices[idx[1]], obj_data.vertices[idx[2]])
                # Fix quadrant and scaling
                a = (round((points[0].x + 1) * width_center), round((points[0].y + 1) * height_center))
                b = (round((points[1].x + 1) * width_center), round((points[1].y + 1) * height_center))
                c = (round((points[2].x + 1) * width_center), round((points[2].y + 1) * height_center))

                triangle(
                    a,
                    b,
                    c,
                    framebuffer,
                    TGAColor_t.random(bpp=uint8_t(3)),
                )

            framebuffer.write_tga_file(f"{basename}.tga")
        except Exception as err:
            print(f"Could not process {fname}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
