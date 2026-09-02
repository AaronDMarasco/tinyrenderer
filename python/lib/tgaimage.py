"""##Types, Classes, and Utilities for working with TARGA images"""
from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError, dataclass, field
from enum import IntEnum
from functools import cache, total_ordering
from io import BytesIO
from itertools import batched
from pathlib import Path
from typing import Any, Final, Self, TypeVar
from warnings import warn

import numpy as np
from numpy import dtype

# Some C++ cross-referencing for simplicity
uint8_t = np.uint8
uint16_t = np.uint16

# Top-level utilities
rng = np.random.default_rng()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
_SENTINEL = object()  # Stop any direct calls to TGAColor_t


# Utility from itertools documentation
def _grouper(iterable, n):
    """Collect data into non-overlapping fixed-length chunks or blocks."""
    iterators = [iter(iterable)] * n
    return zip(*iterators, strict=True)


# The binary layout of a TGA header
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
@total_ordering
class TGAColor_t:
    """
    Base type for a TGAColor

    Because we have lots of these, they are considerd quasi-immutable and can only be generated
    with a factory function `TGAColor()`. They are then cached so there's only one copy of each
    color combination in memory.
    """

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
        _guard: Any | None = None,
    ) -> None:
        # Cached responses:
        self._byte_data = None  # __bytes__
        self._repr = None  # __repr__

        if _guard is not _SENTINEL:
            err_msg = "Only call TGAColor() method to get a TGAColor_t"
            raise TypeError(err_msg)

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
        """Bytes-per-pixel"""
        return len(self._data)

    @property
    def b(self: Self) -> int:
        """Blue (or monochrome value)"""
        return self._data[0]

    @property
    def g(self: Self) -> int:
        """Green (if available)"""
        if self.bytespp >= 2:
            return self._data[1]
        err_msg = f"Asked for g (byte 2) when bytespp={self.bytespp}!"
        raise ValueError(err_msg)

    @property
    def r(self: Self) -> int:
        """Red (if available)"""
        if self.bytespp >= 3:
            return self._data[2]
        err_msg = f"Asked for r (byte 3) when bytespp={self.bytespp}!"
        raise ValueError(err_msg)

    @property
    def a(self: Self) -> int:
        """Alpha channel (if available)"""
        if self.bytespp >= 4:
            return self._data[3]
        err_msg = f"Asked for a (byte 4) when bytespp={self.bytespp}!"
        raise ValueError(err_msg)

    @staticmethod
    def random(bpp: uint8_t | None = None) -> TGAColor_t:
        """Provide a new TGAColor with a random color (RGB format by default)"""
        if bpp is None:
            bpp = uint8_t(3)
        coordinates = [int(rng.integers(255)) for _ in range(bpp)]
        return TGAColor(*coordinates)

    def resize(self: Self, bpp: int | uint8_t) -> TGAColor_t:
        """Converts to a new pixel with a lower BPP"""
        # TODO: Better algorithm if RGB => Mono? Average maybe?
        if bpp > self.bytespp:
            err_msg = f"Asked to increase BPP from {self.bytespp} to {bpp} and don't know how!"
            raise ValueError(err_msg)
        return TGAColor(*(self._data[:bpp]), bpp=int(bpp))

    def __getitem__(self: Self, idx: int) -> uint8_t:
        """Allow raw indexing like C++ prefers"""
        return uint8_t(self._data[idx])

    def __setitem__(self: Self, idx: int, val: uint8_t) -> None:
        raise FrozenInstanceError

    def __mul__(self: Self, other: int | float) -> TGAColor_t:
        """Scaling"""
        if not isinstance(other, (int, float, np.integer, np.floating)):
            return NotImplemented
        res = b"".join(round(v * other).to_bytes() for v in self._data)
        return TGAColor_from_raw(res, bpp=self.bytespp, _allow2=True)[0]

    def __rmul__(self: Self, other: int | float) -> TGAColor_t:
        """Scaling"""
        return self * other

    def __truediv__(self: Self, other: int | float) -> TGAColor_t:
        """Scaling"""
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0:
                raise ZeroDivisionError
            res = b"".join(round(v / other).to_bytes() for v in self._data)
            return TGAColor_from_raw(res, bpp=self.bytespp, _allow2=True)[0]
        return NotImplemented

    def __le__(self: Self, other: Any) -> bool:
        if not isinstance(other, TGAColor_t) or self.bytespp != other.bytespp:
            return NotImplemented
        return self._data <= other._data

    def __eq__(self: Self, other: Any) -> bool:
        if not isinstance(other, TGAColor_t):
            return NotImplemented
        # This is almost 7X faster than hashing each and comparing
        return self._data == other._data

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
            if any(not (0 <= c <= 255) for c in self._data):
                logger.warning("Out-of-range value for color byte; clipping (%s)", self._data)
            self._byte_data = b"".join(max(0, min(c, 255)).to_bytes() for c in self._data)
        return self._byte_data


@cache
def _TGAColor_factory(
    bpp: uint8_t,
    b: int | None = None,
    g: int | None = None,
    r: int | None = None,
    a: int | None = None,
) -> TGAColor_t:
    """Helper for the TGAColor() factory"""
    return TGAColor_t(b, g, r, a, bpp=bpp, _guard=_SENTINEL)


def TGAColor(
    b: int | None = None,
    g: int | None = None,
    r: int | None = None,
    a: int | None = None,
    *,
    bpp: int | None = None,
) -> TGAColor_t:
    """
    End user interface to create a TGAColor_t

    Factory helper function that caches since they are "immutable" (but not really)
    """
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

    err_msg = "Invalid value given - must be 0..255!"
    if b is None or not (0 <= b <= 255):
        raise ValueError(err_msg)
    if bpp_ >= 2 and (g is None or not (0 <= g <= 255)):
        raise ValueError(err_msg)
    if bpp_ >= 3 and (r is None or not (0 <= r <= 255)):
        raise ValueError(err_msg)
    if bpp_ >= 4 and (a is None or not (0 <= a <= 255)):
        raise ValueError(err_msg)

    return _TGAColor_factory(bpp_, b, g, r, a)


def TGAColor_from_raw(data: bytes, *, bpp: int, _allow2: bool = False) -> list[TGAColor_t]:
    """Factory helper - take bytes and get a list of TGAColors"""
    if (ld := len(data)) % bpp:
        warn(f"Possibly bad read of {ld} bytes at {bpp} bpp = remainder {ld % bpp}", stacklevel=2)

    if bpp == 1:  # Grayscale
        return [TGAColor(b=v, bpp=1) for v in data]
    if _allow2 and bpp == 2:  # Special mode for scaling tests only(?)
        return [TGAColor(b=b, g=g, bpp=2) for (b, g) in batched(data, 2)]
    if bpp == 3:  # RGB
        return [TGAColor(b=b, g=g, r=r, bpp=3) for (b, g, r) in batched(data, 3)]
    if bpp == 4:  # RGBA
        return [TGAColor(b=b, g=g, r=r, a=a, bpp=4) for (b, g, r, a) in batched(data, 4)]
    raise NotImplementedError(f"Cannot handle {bpp} BPP")


TI = TypeVar("TI", bound="TGAImage")


class TGAImage:
    """An in-memory TGAImage"""

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
        # Reminder: numpy array is arr[rows][cols] so maps to arr[y][x] not arr[x][y]
        # Origin is in bottom left corner of image
        self.npdata: np.ndarray = np.full(shape=(h, w), fill_value=self.fill_value)

        # These are pretty much for testing purposes only:
        self.was_hflipped = False
        self.was_vflipped = False
        self.was_rle = False

    @classmethod
    def read_tga_file(cls: type[TI], filename: str | Path) -> TI:
        """Read a file on disk into memory"""
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
        res.npdata = np.array([x for x in _grouper(TGAColor_from_raw(raw_data, bpp=bpp), w)], dtype=TGAColor_t)
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
        all_bytes: Final = np.vectorize(bytes)(self.npdata)
        return all_bytes.ravel().tobytes()

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
        """Writes image to disk"""
        # logger.debug("Writing to file %s...", filename)
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
        logger.info("Wrote to file %s: %s", filename, self)

    def flip_horizontally(self: Self) -> None:
        self.npdata = np.fliplr(self.npdata)

    def flip_vertically(self: Self) -> None:
        self.npdata = np.flipud(self.npdata)

    def get(self: Self, x: int, y: int) -> TGAColor_t:
        """Reads out a given x, y coordinate"""
        return self.npdata[y, x]

    def set(self: Self, x: int, y: int, c: TGAColor_t) -> None:
        """Writes to a given x, y coordinate"""
        if c.bytespp != self.bpp:
            old = c
            try:
                c = c.resize(self.bpp)
            except Exception as err:
                err_msg = f"Pixel write at ({x}, {y}) {c} failed resizing to bpp={self.bpp}"
                raise ValueError(err_msg) from err
            warn(f"Pixel write at ({x}, {y}) changed bpp: was {old} now {c}", stacklevel=2)
        if not (0 <= x < self.width) or not (0 <= y < self.height):
            logger.warning("TGAImage.set(%s, %s) invalid: Image is %d x %d", x, y, self.width, self.height)
            return
        self.npdata[y, x] = c

    def load_rle_data(self: Self, in_: bytes) -> bytes:
        """Decompresses a TGA RLE stream"""
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
                    for _ in range(chunk_header):
                        raw_data.write(in_[current_byte : current_byte + chunk_end])
                current_pixel += chunk_header
                current_byte += chunk_end
            return raw_data.getvalue()

    def unload_rle_data(self: Self) -> bytes:
        """
        Weirdly named; actually does the RLE encoding

        Takes advantage of numpy parallelization by having it do the heavy lifting and then
        RLE encode it as TGA requires
        """
        flat_data: Final = self.npdata.ravel()

        # Find positions of changes
        changes: Final = flat_data[1:] != flat_data[:-1]

        # Get the indices of these changes and end-of-data index
        idx: Final = np.append(np.nonzero(changes)[0], len(flat_data) - 1)

        # Calculate run lengths by getting the difference between change indices
        run_lengths: Final = np.diff(np.append(-1, idx))
        # Finally use idx as a mask to get the values we need to use
        unique_values: Final = flat_data[idx]

        with BytesIO() as res:
            raw_index: int | None = None  # First raw byte seen (if we're in a run)
            # We can't do a more pythonic "for count, pixel in zip(run_lengths, unique_values, strict=True)"
            # because we need to "look around" too much
            for i in range(len(run_lengths)):
                last_byte: bool = i == (len(run_lengths) - 1)
                if (count := run_lengths[i]) == 1:
                    if raw_index is None:
                        raw_index = i
                    if last_byte:
                        logger.debug("Ending on a raw run...")
                        count = 0  # There is now no non-raw block to write after
                    else:
                        continue
                # Need to check if we just finished a raw run and if so, dump one or more raw segments
                if raw_index is not None:
                    raw_count = i - raw_index
                    if count == 0:  # Ending on raw, so special case +1
                        raw_count += 1
                    while raw_count:
                        size = min(raw_count, 128)  # Max chunk size is 128
                        flag = size - 1
                        res.write(flag.to_bytes(1))
                        for chunk in map(bytes, unique_values[raw_index : raw_index + size]):
                            res.write(chunk)
                        raw_index += size
                        raw_count -= size
                    raw_index = None
                # Need to write out one or more RLE (non-raw) blocks
                while count:
                    size = min(int(count), 128)  # Max chunk size is 128, count was numpy native until now
                    flag = size + 127  # +128 to set high bit, -1 for TGA spec
                    res.write(flag.to_bytes(1))
                    res.write(bytes(unique_values[i]))
                    count -= size
            return res.getvalue()

    def __str__(self: Self) -> str:
        return f"{self.width}x{self.height}/{self.bpp}"


white: Final = TGAColor(255, 255, 255, 255).resize(bpp=3)  # Attention: BGRA order
green: Final = TGAColor(0, 255, 0, 255).resize(bpp=3)
red: Final = TGAColor(0, 0, 255, 255).resize(bpp=3)
blue: Final = TGAColor(255, 128, 64, 255).resize(bpp=3)
yellow: Final = TGAColor(0, 200, 255, 255).resize(bpp=3)
