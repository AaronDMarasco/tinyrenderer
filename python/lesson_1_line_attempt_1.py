from __future__ import annotations

import sys
from typing import Final

import numpy as np

from lib.tgaimage import TGAColor_t, TGAImage, blue, green, red, white, yellow

_CPP_SOURCE = """
#include <cmath>
#include "tgaimage.h"

constexpr TGAColor white   = {255, 255, 255, 255}; // attention, BGRA order
constexpr TGAColor green   = {  0, 255,   0, 255};
constexpr TGAColor red     = {  0,   0, 255, 255};
constexpr TGAColor blue    = {255, 128,  64, 255};
constexpr TGAColor yellow  = {  0, 200, 255, 255};

void line(int ax, int ay, int bx, int by, TGAImage &framebuffer, TGAColor color) {
    for (float t=0.; t<1.; t+=.02) {
        int x = std::round( ax + (bx-ax)*t );
        int y = std::round( ay + (by-ay)*t );
        framebuffer.set(x, y, color);
    }
}

int main(int argc, char** argv) {
    constexpr int width  = 64;
    constexpr int height = 64;
    TGAImage framebuffer(width, height, TGAImage::RGB);

    int ax =  7, ay =  3;
    int bx = 12, by = 37;
    int cx = 62, cy = 53;

    line(ax, ay, bx, by, framebuffer, blue);
    line(cx, cy, bx, by, framebuffer, green);
    line(cx, cy, ax, ay, framebuffer, yellow);
    line(ax, ay, cx, cy, framebuffer, red);

    framebuffer.set(ax, ay, white);
    framebuffer.set(bx, by, white);
    framebuffer.set(cx, cy, white);

    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}
"""


def line(ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t) -> None:
    for t in np.linspace(start=0.0, stop=1.0, num=50, endpoint=False):
        x = round(ax + (bx - ax) * t)
        y = round(ay + (by - ay) * t)
        framebuffer.set(x, y, color)


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

    line(ax, ay, bx, by, framebuffer, blue)
    line(cx, cy, bx, by, framebuffer, green)
    line(cx, cy, ax, ay, framebuffer, yellow)
    line(ax, ay, cx, cy, framebuffer, red)

    framebuffer.set(ax, ay, white)
    framebuffer.set(bx, by, white)
    framebuffer.set(cx, cy, white)

    framebuffer.write_tga_file("framebuffer.tga")

    return 0


if __name__ == "__main__":
    sys.exit(main())
