from __future__ import annotations

import sys
from typing import Final

from lib.render import triangle_barycentric_lesson_3_homework
from lib.tgaimage import TGAColor, TGAImage

white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)


def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
    triangle_barycentric_lesson_3_homework((17, 4, 13), (55, 39, 128), (23, 59, 255), framebuffer)
    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
