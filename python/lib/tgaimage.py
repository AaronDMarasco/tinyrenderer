from __future__ import annotations
from enum import IntEnum
from typing import BinaryIO, Final, Self, TypeAlias

from icecream import ic  # DEBUG
from numpy import dtype
import friendly  # DEBUG
import numpy as np
# import numpy.typing as npt

friendly.install()  # DEBUG

# Some C++ cross-referencing for simplicity
uint8_t: TypeAlias = np.uint8
uint16_t: TypeAlias = np.uint16

TGAHeader: Final[dtype] = dtype(
    [
        ("idlength", uint8_t),
        ("colormaptype", uint8_t),
        ("datatypecode", uint8_t),
        ("colormaporigin", uint16_t),
        ("colormaplength", uint16_t),
        ("colormapdepth", uint8_t),
        ("x_origin", uint16_t),
        ("y_origin", uint16_t),
        ("width", uint16_t),
        ("height", uint16_t),
        ("bitsperpixel", uint8_t),
        ("imagedescriptor", uint8_t),
    ]
)


class TGAColor:
    # TODO: Still working. Binary dump representation. Tests (hypothesis)!
    bgra_t: Final[dtype] = dtype(
        [
            ("b", uint8_t),
            ("g", uint8_t),
            ("r", uint8_t),
            ("a", uint8_t),
        ]
    )

    def __init__(self: Self, b: int = 0, g: int = 0, r: int = 0, a: int = 0, bpp: uint8_t = uint8_t(4)) -> None:
        self.bgra = np.array([(b, g, r, a)], dtype=TGAColor.bgra_t)
        self.bytespp: uint8_t = bpp
        assert bpp == 4, "Not sure if this is valid?"

    def __getitem__(self: Self, idx: int) -> uint8_t:
        return self.bgra[0][idx]  # Let numpy handle indexing

    def __setitem__(self: Self, idx: int, val: uint8_t) -> None:
        match idx:
            case 0:
                self.bgra[0]["b"] = val
            case 1:
                self.bgra[0]["g"] = val
            case 2:
                self.bgra[0]["r"] = val
            case 3:
                self.bgra[0]["a"] = val
            case _:
                raise IndexError("Only allow 0-3")

    def __eq__(self: Self, other: object) -> bool:
        if not isinstance(other, TGAColor):
            return NotImplemented
        return self.bgra == other.bgra and self.bytespp == other.bytespp


class TGAImage:
    """
        TODO: Everything
        enum Format { GRAYSCALE=1, RGB=3, RGBA=4 };
        TGAImage() = default;
        TGAImage(const int w, const int h, const int bpp, TGAColor c = {});
        bool  read_tga_file(const std::string filename);
        bool write_tga_file(const std::string filename, const bool vflip=true, const bool rle=true) const;
        void flip_horizontally();
        void flip_vertically();
        TGAColor get(const int x, const int y) const;
        void set(const int x, const int y, const TGAColor &c);
        int width()  const;
        int height() const;
    private:
        bool   load_rle_data(std::ifstream &in);
        bool unload_rle_data(std::ofstream &out) const;
        int w = 0, h = 0;
        std::uint8_t bpp = 0;
        std::vector<std::uint8_t> data = {};
    """

    class Format(IntEnum):
        GRAYSCALE = 1
        RGB = 3
        RGBA = 4

    def __init__(self: Self, w: int = 0, h: int = 0, bpp: int = 4, c: TGAColor | None = None) -> None:
        self.width = w
        self.height = h
        self.bpp: uint8_t = uint8_t(bpp)
        self.data: list[uint8_t] = [uint8_t(0)] * (w * h * bpp)  # TODO: Make np.array or np.ndarray? Of TGAColor?
        self.npdata = np.full(shape=(h, w), fill_value=TGAColor())

        if c is not None:
            self.npdata = np.full(shape=(h, w), fill_value=c)

            ic(c)
            # ic(self.data)
            ic(self.npdata)
            for j in range(h):
                for i in range(w):
                    # self.set(i, j, c)
                    pass

        # raise NotImplementedError

    def read_tga_file(self: Self, filename: str) -> bool:
        raise NotImplementedError

    def write_tga_file(self: Self, filename: str, vflip: bool = True, rle: bool = True) -> bool:
        raise NotImplementedError

    def flip_horizontally(self: Self) -> None:
        raise NotImplementedError

    def flip_vertically(self: Self) -> None:
        raise NotImplementedError

    def get(self: Self, x: int, y: int) -> TGAColor:
        # TODO: Test this compared to what the C++ library does?
        if (len(self.data) == 0) or x < 0 or y < 0 or x >= self.width or y >= self.height:
            raise IndexError
        ret = TGAColor(bpp=self.bpp)
        idx = (x + y * self.width) * self.bpp
        for i in range(self.bpp, 0, -1):
            ret.bgra[i] = self.data[idx + i]
        return ret

    def set(self: Self, x: int, y: int, c: TGAColor) -> None:
        raise NotImplementedError

    def load_rle_data(self: Self, in_: BinaryIO) -> bool:
        raise NotImplementedError

    def unload_rle_data(self: Self, out_: BinaryIO) -> bool:
        raise NotImplementedError


if __name__ == "__main__":
    # print(TGAImage(h=5, w=3, c=TGAColor(1, 2, 3, 4)).npdata)
    pass
