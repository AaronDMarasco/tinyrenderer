from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import batched
from pathlib import Path
from typing import BinaryIO, Final, Self, TypeAlias, TypeVar
from warnings import warn

import numpy as np
from icecream import ic
from numpy import dtype

# Some C++ cross-referencing for simplicity
uint8_t: TypeAlias = np.uint8
uint16_t: TypeAlias = np.uint16


# Utility from itertools documentation
def grouper(iterable, n):
    """Collect data into non-overlapping fixed-length chunks or blocks."""
    iterators = [iter(iterable)] * n
    return zip(*iterators, strict=True)


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

bgra_t: Final[dtype] = dtype(
    [
        ("b", uint8_t),
        ("g", uint8_t),
        ("r", uint8_t),
        ("a", uint8_t),
    ]
)


@dataclass(slots=True)
class TGAColor:
    # TODO: Still working. Binary dump representation.
    bgra: np.ndarray
    bpp: uint8_t
    bytespp: uint8_t

    def __init__(self: Self, b: int = 0, g: int = 0, r: int = 0, a: int = 0, bpp: uint8_t | None = None) -> None:
        self.bgra = np.array([(b, g, r, a)], dtype=bgra_t)
        if bpp is None:
            bpp = uint8_t(4)
        self.bytespp: uint8_t = bpp

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
        return hash(self) == hash(other)

    def __hash__(self: Self) -> int:
        return hash((tuple(self.bgra.tolist()), self.bytespp))

    def __repr__(self: Self) -> str:
        return f"TGAColor({self[0]}, {self[1]}, {self[2]}, {self[3]}, {self.bytespp})"

    @staticmethod
    def from_raw(raw_data: bytes, *, bpp: int) -> list[TGAColor]:
        if (ld := len(raw_data)) % bpp:
            warn(f"Possibly bad read of {ld} bytes at {bpp} bpp = remainder {ld % bpp}", stacklevel=2)
        data = [int(v) for v in raw_data]
        if bpp == 1:  # Grayscale
            return [TGAColor(b=v, g=v, r=v, a=255, bpp=uint8_t(1)) for v in data]
        if bpp == 3:  # RGB
            return [TGAColor(b=b, g=g, r=r, a=255, bpp=uint8_t(3)) for (b, g, r) in batched(data, 3)]
        if bpp == 4:  # RGBA
            return [TGAColor(b=b, g=g, r=r, a=a, bpp=uint8_t(4)) for (b, g, r, a) in batched(data, 4)]
        raise NotImplementedError(f"Cannot handle {bpp} BPP")


TI = TypeVar("TI", bound="TGAImage")


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

    FORMAT_VALS: Final = set(x.value for x in Format)

    def __init__(self: Self, w: int = 0, h: int = 0, bpp: int = 4, c: TGAColor | None = None) -> None:
        self.width = w
        self.height = h
        self.bpp: uint8_t = uint8_t(bpp)
        self.npdata: np.ndarray = np.full(shape=(h, w), fill_value=TGAColor())

        if c is not None:
            self.npdata = np.full(shape=(h, w), fill_value=c)

    @classmethod
    def read_tga_file(cls: type[TI], filename: str | Path) -> TI:
        header = np.fromfile(filename, dtype=TGAHeader, count=1)[0]
        # print(header)
        w = header["width"]
        h = header["height"]
        bpp = int(header["bitsperpixel"]) >> 3
        data_size: Final = int(w) * int(h) * int(bpp)
        dtc = header["datatypecode"]
        imgd = header["imagedescriptor"]
        assert w > 0, f"Interpreted width {w} is invalid!"
        assert h > 0, f"Interpreted height {h} is invalid!"
        assert bpp in cls.FORMAT_VALS, f"Interpreted bits-per-pixel {bpp} is invalid!"
        assert dtc in {2, 3, 10, 11}, f"Unknown file format '{dtc}'!"
        # assert bpp == 4, f"Not sure if non bpp=4 works (bpp={bpp})?"
        res = cls(w=int(w), h=int(h), bpp=bpp)
        # Read the data without the header
        raw_data = Path(filename).read_bytes()[TGAHeader.itemsize :]
        if dtc in {10, 11}:
            # RLE data
            raw_data = res.load_rle_data(raw_data)
        if dtc in {2, 3}:
            # Not RLE data
            # Truncate it
            # trailing = raw_data[data_size:]
            raw_data = raw_data[:data_size]
        ic(len(raw_data), data_size)
        pixels = [x for x in grouper(TGAColor.from_raw(raw_data, bpp=bpp), w)]
        res.npdata = np.array(pixels)
        assert res.npdata.shape == (h, w), f"Re-shaping error? {(h, w)=} vs. {res.npdata.shape}"
        if not (imgd & 0x20):
            res.flip_vertically()
        if imgd & 0x10:
            res.flip_horizontally()
        return res

    def write_tga_file(self: Self, filename: str | Path, vflip: bool = True, rle: bool = True) -> bool:
        raise NotImplementedError

    def flip_horizontally(self: Self) -> None:
        self.npdata = np.fliplr(self.npdata)

    def flip_vertically(self: Self) -> None:
        self.npdata = np.flipud(self.npdata)

    def get(self: Self, x: int, y: int) -> TGAColor:
        # TODO: Test this compared to what the C++ library does?
        return self.npdata[y, x]

    def set(self: Self, x: int, y: int, c: TGAColor) -> None:
        self.npdata[y, x] = c

    def load_rle_data(self: Self, in_: bytes) -> bytes:
        # See https://www.fileformat.info/format/tga/egff.htm
        pixel_count: Final[int] = self.width * self.height
        raw_data: list[bytes] = []
        current_pixel: int = 0
        current_byte: int = 0
        while current_pixel < pixel_count:
            # ic(current_pixel, pixel_count, [hex(x) for x in in_[current_byte : current_byte + int(self.bpp)]])
            chunk_header = in_[current_byte]
            current_byte += 1
            if chunk_header < 128:
                # "Raw" pixels - just read them out (up to 127)
                chunk_header += 1
                for _ in range(chunk_header):
                    raw_data.append(in_[current_byte : current_byte + int(self.bpp)])
                    current_byte += int(self.bpp)
                    current_pixel += 1
            else:
                # RLE
                chunk_header -= 127
                raw_data.extend(in_[current_byte : current_byte + int(self.bpp)] for _ in range(chunk_header))
                current_byte += int(self.bpp)
                current_pixel += chunk_header
        return b"".join(raw_data)

    def unload_rle_data(self: Self, out_: bytes) -> bool:
        raise NotImplementedError
