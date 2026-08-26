from __future__ import annotations

import sys
from typing import Final

from lib.tgaimage import TGAImage, white

_CPP_SOURCE = """
constexpr TGAColor white   = {255, 255, 255, 255}; // attention, BGRA order
constexpr TGAColor green   = {  0, 255,   0, 255};
constexpr TGAColor red     = {  0,   0, 255, 255};
constexpr TGAColor blue    = {255, 128,  64, 255};
constexpr TGAColor yellow  = {  0, 200, 255, 255};

int main(int argc, char** argv) {
    constexpr int width  = 64;
    constexpr int height = 64;
    TGAImage framebuffer(width, height, TGAImage::RGB);

    int ax =  7, ay =  3;
    int bx = 12, by = 37;
    int cx = 62, cy = 53;

    framebuffer.set(ax, ay, white);
    framebuffer.set(bx, by, white);
    framebuffer.set(cx, cy, white);

    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}
"""


def main() -> int:
    width: Final = 64
    height: Final = 64

    framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

    ax = 7
    ay = 3
    bx = 12
    by = 37
    cx = 62
    cy = 53

    # A difference in the python vs C++ version is that python only stores valid data,
    # so the BPP of the pixel must match the image. If not, you will get a warning, e.g.:
    framebuffer.set(ax, ay, white)  # "Pixel write at (7, 3) changed bpp"
    framebuffer.set(bx, by, white.resize(bpp=3))
    framebuffer.set(cx, cy, white.resize(bpp=3))

    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
