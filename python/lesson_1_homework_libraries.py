from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

from lib.objreader import OBJ_Data

# logging.getLogger('lib.objreader').setLevel(logging.INFO)


def main() -> int:
    width: Final = 2047
    height: Final = 2048

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
        basename = Path(fname).name
        try:
            obj_data = OBJ_Data.from_file(fname)
            framebuffer = obj_data.orthogonal_projection(width=width, height=height)
            framebuffer.write_tga_file(f"{basename}.tga")
        except Exception as err:
            print(f"Could not process {fname}: {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
