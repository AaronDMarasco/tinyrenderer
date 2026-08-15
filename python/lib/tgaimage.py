from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from enum import IntEnum
from functools import cache
from io import BytesIO
from itertools import batched
from pathlib import Path
from typing import Final, Self, TypeAlias, TypeVar
from warnings import warn

import numpy as np
from numpy import dtype

# Some C++ cross-referencing for simplicity
uint8_t: TypeAlias = np.uint8
uint16_t: TypeAlias = np.uint16


__all__ = ["TGAColor", "TGAColor_from_raw", "TGAHeader", "TGAImage", "uint8_t", "uint16_t"]


# Utility from itertools documentation
def _grouper(iterable, n):
    """Collect data into non-overlapping fixed-length chunks or blocks."""
    iterators = [iter(iterable)] * n
    return zip(*iterators, strict=True)


TGAHeader: Final[dtype] = dtype([
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
])


@dataclass(slots=True)
class TGAColor_t:
    _data: tuple[int, ...] = field(init=False)
    _byte_data: bytes | None = field(default=None, init=False)
    _repr: str | None = field(default=None, init=False)

    def __init__(
        self: Self,
        b: int | None = None,
        g: int | None = None,
        r: int | None = None,
        a: int | None = None,
        *,
        bpp: uint8_t | None = None,
    ) -> None:
        # Cached responses:
        self._byte_data = None  # __bytes__
        self._repr = None  # __repr__

        if bpp is None:
            raise ValueError("BPP shenanigans? Wrappers should have set this!")
        if not (1 <= bpp <= 4):
            raise ValueError(f"Invalid BPP={bpp}!")

        # I hate to ignore type checking, but we ensure right after that it will not have None...
        self._data = (b, g, r, a)[:bpp]  # type: ignore[assignment]
        if None in self._data:
            err_msg = f"Out-of-order None in constructor! ({b=} {g=} {r=} {a=} {bpp=})"
            raise ValueError(err_msg)

    @property
    def bytespp(self: Self) -> int:
        return len(self._data)

    def resize(self: Self, bpp: int | uint8_t) -> TGAColor_t:
        """Converts to a new pixel with a lower BPP"""
        # TODO: Better algorithm if RGB => Mono? Average maybe?
        if bpp > self.bytespp:
            err_msg = f"Asked to increase BPP from {self.bytespp} to {bpp} and don't know how!"
            raise ValueError(err_msg)
        return TGAColor(*(self._data[:bpp]), bpp=int(bpp))

    def __getitem__(self: Self, idx: int) -> uint8_t:
        return uint8_t(self._data[idx])

    def __setitem__(self: Self, idx: int, val: uint8_t) -> None:
        raise FrozenInstanceError

    def __hash__(self: Self) -> int:
        return hash(self._data)

    def __repr__(self: Self) -> str:
        if self._repr is None:
            try:
                bpp = self.bytespp
                res = [f"b={self[0]}"]
                if bpp >= 2:
                    res.append(f"g={self[1]}")
                if bpp >= 3:
                    res.append(f"r={self[2]}")
                if bpp == 4:
                    res.append(f"a={self[3]}")
                self._repr = "TGAColor_t(" + ", ".join(res) + f", bpp={self.bytespp})"
            except Exception:
                return f"TGAColor_t(NOT FULLY INITIALIZED @ {hex(id(self))})"

        return self._repr

    def __bytes__(self: Self) -> bytes:
        if self._byte_data is None:
            self._byte_data = b"".join(c.to_bytes() for c in self._data)
        return self._byte_data


@cache
def _TGAColor_factory(
    bpp: uint8_t,
    b: int | None = None,
    g: int | None = None,
    r: int | None = None,
    a: int | None = None,
) -> TGAColor_t:
    return TGAColor_t(b, g, r, a, bpp=bpp)


def TGAColor(
    b: int | None = None,
    g: int | None = None,
    r: int | None = None,
    a: int | None = None,
    *,
    bpp: int | None = None,
) -> TGAColor_t:
    """Factory helper function that caches since they are immutable"""
    # The bpp logic is here to fix caching of specified or not...
    # Special case: Default constructor is (b=0, bpp=1)
    if (b, g, r, a, bpp) == (None,) * 5:
        b = 0

    bpp_: uint8_t
    if bpp is None:
        match (g, r, a):
            case (None, None, None):
                bpp_ = uint8_t(1)
            case (_, None, None):
                bpp_ = uint8_t(2)
            case (_, _, None):
                bpp_ = uint8_t(3)
            case (_, _, _):
                bpp_ = uint8_t(4)
            case _:
                raise ValueError("BPP shenanigans?")
    else:
        bpp_ = uint8_t(bpp)

    return _TGAColor_factory(bpp_, b, g, r, a)


def TGAColor_from_raw(data: bytes, *, bpp: int) -> list[TGAColor_t]:
    """Factory helper - take bytes and get a list of TGAColors"""
    if (ld := len(data)) % bpp:
        warn(f"Possibly bad read of {ld} bytes at {bpp} bpp = remainder {ld % bpp}", stacklevel=2)

    if bpp == 1:  # Grayscale
        return [TGAColor(b=v, bpp=1) for v in data]
    if bpp == 3:  # RGB
        return [TGAColor(b=b, g=g, r=r, bpp=3) for (b, g, r) in batched(data, 3)]
    if bpp == 4:  # RGBA
        return [TGAColor(b=b, g=g, r=r, a=a, bpp=4) for (b, g, r, a) in batched(data, 4)]
    raise NotImplementedError(f"Cannot handle {bpp} BPP")


TI = TypeVar("TI", bound="TGAImage")


class TGAImage:
    class Format(IntEnum):
        GRAYSCALE = 1
        RGB = 3
        RGBA = 4

    FORMAT_VALS: Final = set(x.value for x in Format)

    def __init__(self: Self, w: int = 0, h: int = 0, bpp: int = 4, c: TGAColor_t | None = None) -> None:
        self.width = w
        self.height = h
        self.bpp: uint8_t = uint8_t(bpp)
        self.fill_value = c if c is not None else TGAColor(*([0] * bpp), bpp=bpp)
        self.npdata: np.ndarray = np.full(shape=(h, w), fill_value=self.fill_value)

        # These are pretty much for testing purposes only:
        self.was_hflipped = False
        self.was_vflipped = False
        self.was_rle = False

    @classmethod
    def read_tga_file(cls: type[TI], filename: str | Path) -> TI:
        header = np.fromfile(filename, dtype=TGAHeader, count=1)[0]
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
            res.was_rle = True
        if dtc in {2, 3}:
            # Not RLE data
            # trailing = raw_data[data_size:]
            # Truncate it
            raw_data = raw_data[:data_size]
        pixels = [x for x in _grouper(TGAColor_from_raw(raw_data, bpp=bpp), w)]
        res.npdata = np.array(pixels)
        assert res.npdata.shape == (h, w), f"Re-shaping error? {(h, w)=} vs. {res.npdata.shape}"
        if not (imgd & 0x20):
            res.flip_vertically()
            res.was_vflipped = True
        if imgd & 0x10:
            res.flip_horizontally()
            res.was_hflipped = True
        return res

    @property
    def _raw_payload(self: Self) -> bytes:
        self.verify()
        return b"".join(bytes(c) for c in self.npdata.flat)

    def verify(self: Self) -> None:
        """Checks that all pixels are the right size"""
        # Generically named if we want to add other verification later
        errs: list[str] = []
        # Check pixels have correct BPP:
        all_bytespp: Final = np.vectorize(lambda x: x.bytespp)(self.npdata)
        for bad_bpp in np.argwhere(all_bytespp != self.bpp).tolist():
            # Internally addressed as (y, x) so swap in report
            errs.append(f"Pixel (x, y)={bad_bpp[1], bad_bpp[0]} has bytespp={all_bytespp[*bad_bpp]} not {self.bpp}")
        if errs:
            err_msg = f"{len(errs)} verification errors!\n" + "\n".join(errs)
            raise ValueError(err_msg)

    def write_tga_file(self: Self, filename: str | Path, vflip: bool = True, rle: bool = True) -> None:
        self.verify()
        developer_area_ref: Final[bytes] = b"\0\0\0\0"
        extension_area_ref: Final[bytes] = b"\0\0\0\0"
        footer: Final[bytes] = b"TRUEVISION-XFILE.\0"
        header = np.zeros(1, dtype=TGAHeader)
        header["bitsperpixel"] = self.bpp << 3
        header["width"] = self.width
        header["height"] = self.height
        header["datatypecode"] = (10 if rle else 2) + (1 if self.bpp == self.Format.GRAYSCALE else 0)
        header["imagedescriptor"] = 0 if vflip else 0x20  # top-left or bottom-left origin
        with open(filename, "wb") as out:
            out.write(header.tobytes())
            out.write(self.unload_rle_data() if rle else self._raw_payload)
            out.write(developer_area_ref)
            out.write(extension_area_ref)
            out.write(footer)

    def flip_horizontally(self: Self) -> None:
        self.npdata = np.fliplr(self.npdata)

    def flip_vertically(self: Self) -> None:
        self.npdata = np.flipud(self.npdata)

    def get(self: Self, x: int, y: int) -> TGAColor_t:
        return self.npdata[y, x]

    def set(self: Self, x: int, y: int, c: TGAColor_t) -> None:
        if c.bytespp != self.bpp:
            old = c
            try:
                c = c.resize(self.bpp)
            except Exception as err:
                err_msg = f"Pixel write at ({x}, {y}) {c} failed resizing to bpp={self.bpp}"
                raise ValueError(err_msg) from err
            warn(f"Pixel write at ({x}, {y}) changed bpp: was {old} now {c}", stacklevel=2)
        self.npdata[y, x] = c

    def load_rle_data(self: Self, in_: bytes) -> bytes:
        # See https://www.fileformat.info/format/tga/egff.htm
        pixel_count: Final[int] = self.width * self.height
        current_pixel: int = 0
        current_byte: int = 0
        with BytesIO() as raw_data:
            while current_pixel < pixel_count:
                chunk_header = in_[current_byte]
                current_byte += 1
                if chunk_header < 128:
                    # "Raw" pixels - just read them out (up to 127)
                    chunk_header += 1
                    chunk_end = chunk_header * int(self.bpp)
                    raw_data.write(in_[current_byte : current_byte + chunk_end])
                else:
                    # RLE - insert N copies of a single pixel into output
                    chunk_header -= 127
                    chunk_end = int(self.bpp)
                    raw_data.writelines(in_[current_byte : current_byte + chunk_end] for _ in range(chunk_header))
                current_pixel += chunk_header
                current_byte += chunk_end
            return raw_data.getvalue()

    def unload_rle_data(self: Self) -> bytes:
        """Weirdly named; actually does the RLE encoding"""
        MAX_CHUNK: Final[int] = 128
        pixel_count: Final[int] = self.width * self.height
        current_pixel: int = 0
        flat_data: Final = self.npdata.flat
        with BytesIO() as res:
            while current_pixel < pixel_count:
                run_length = 1
                raw = True
                while (current_pixel + run_length) < pixel_count and (run_length < MAX_CHUNK):
                    succ_eq = flat_data[current_pixel + run_length - 1] == flat_data[current_pixel + run_length]
                    if run_length == 1:  # First pass determines if chunk is raw or not
                        raw = not succ_eq
                    if raw and succ_eq:  # We're done doing a raw chunk
                        run_length -= 1
                        break
                    if not raw and not succ_eq:  # We're done doing an RLE chunk
                        break
                    run_length += 1
                flag = (run_length - 1) if raw else (run_length + 127)
                res.write(flag.to_bytes(1))
                if raw:
                    res.writelines(bytes(c) for c in flat_data[current_pixel : current_pixel + run_length])
                else:
                    res.write(bytes(flat_data[current_pixel]))
                current_pixel += run_length
            return res.getvalue()
