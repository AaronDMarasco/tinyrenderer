from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import batched
from pathlib import Path
from typing import Final, Self, TypeAlias, TypeVar, overload
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


@dataclass(slots=True)
class TGAColor:
    # TODO: Still working. Binary dump representation.
    data: list[bytes]
    bytespp: uint8_t

    @overload
    def __init__(self: Self, b: int = 0, g: int = 0, r: int = 0, a: int = 0, bpp: uint8_t | None = None) -> None: ...
    @overload
    def __init__(self: Self, b: bytes, g: bytes, r: bytes, a: bytes, bpp: uint8_t | None = None) -> None: ...

    def __init__(self: Self, b=0, g=0, r=0, a=0, bpp: uint8_t | None = None) -> None:
        if bpp is None:
            bpp = uint8_t(4)
        if isinstance(b, int):
            assert isinstance(g, int) and isinstance(r, int) and isinstance(a, int), "Bad call!"
            self.data = [b.to_bytes(1), g.to_bytes(1), r.to_bytes(1), a.to_bytes(1)][:bpp]
        elif isinstance(b, bytes):
            assert isinstance(g, bytes) and isinstance(r, bytes) and isinstance(a, bytes), "Bad call!"
            self.data = [b, g, r, a][:bpp]
        else:
            raise ValueError("Bad call types!")
        self.bytespp: uint8_t = bpp
        assert len(self.data) == bpp, "Invalid internal error with bpp?"

    def __getitem__(self: Self, idx: int) -> uint8_t:
        return uint8_t(int.from_bytes(self.data[idx]))

    def __setitem__(self: Self, idx: int, val: uint8_t) -> None:
        self.data[idx] = int(val).to_bytes(1)

    def __hash__(self: Self) -> int:
        return hash((self.bytespp, tuple(self.data)))

    def __repr__(self: Self) -> str:
        return f"TGAColor(b={self[0]}, g={self[1]}, r={self[2]}, a={self[3]}, bpp={self.bytespp})"

    @staticmethod
    def from_raw(data: bytes, *, bpp: int) -> list[TGAColor]:
        if (ld := len(data)) % bpp:
            warn(f"Possibly bad read of {ld} bytes at {bpp} bpp = remainder {ld % bpp}", stacklevel=2)

        if bpp == 1:  # Grayscale
            return [TGAColor(b=v, bpp=uint8_t(1)) for v in data]
        if bpp == 3:  # RGB
            return [TGAColor(b=b, g=g, r=r, bpp=uint8_t(3)) for (b, g, r) in batched(data, 3)]
        if bpp == 4:  # RGBA
            return [TGAColor(b=b, g=g, r=r, a=a, bpp=uint8_t(4)) for (b, g, r, a) in batched(data, 4)]
        raise NotImplementedError(f"Cannot handle {bpp} BPP")


TI = TypeVar("TI", bound="TGAImage")


class TGAImage:
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
        res = cls(w=int(w), h=int(h), bpp=bpp)
        # Read the data without the header
        raw_data = Path(filename).read_bytes()[TGAHeader.itemsize :]
        if dtc in {10, 11}:
            # RLE data
            raw_data = res.load_rle_data(raw_data)
        if dtc in {2, 3}:
            # Not RLE data
            # trailing = raw_data[data_size:]
            # Truncate it
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
