from __future__ import annotations

import sys
from typing import Final

from lib.render import triangle
from lib.tgaimage import TGAColor, TGAImage

white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)


def main() -> int:
    width: Final = 128
    height: Final = 128

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)
    triangle(7, 45, 35, 100, 45, 60, framebuffer, red)
    triangle(120, 35, 90, 5, 45, 110, framebuffer, white, fill=True)
    triangle(115, 83, 80, 90, 85, 120, framebuffer, green)
    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
