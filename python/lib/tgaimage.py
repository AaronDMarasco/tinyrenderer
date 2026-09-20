"""##Types, Classes, and Utilities for working with TARGA images"""

from __future__ import annotations

import inspect
import logging
import operator
import threading
from dataclasses import FrozenInstanceError, dataclass, field
from enum import IntEnum
from functools import cache, total_ordering
from io import BytesIO
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypeVar
from warnings import warn

import numba as nb
import numba.extending as nbex
import numpy as np
import numpy.typing as npt
from numba import float64, int64, njit, objmode
from numba import types as nbt
from numba.experimental import jitclass, structref
from numpy import dtype

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import FrameType

try:
    matplotlib_import_err: ImportError | None = None
    logging.getLogger("matplotlib").setLevel("WARNING")
    logging.getLogger("PIL").setLevel("WARNING")
    import matplotlib.pyplot as plt
except ImportError as e:
    matplotlib_import_err = e

# Some C++ cross-referencing for simplicity
uint8_t = np.uint8
uint16_t = np.uint16

# Top-level utilities
rng = np.random.default_rng()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
# _SENTINEL = nb.typed.List.empty_list(int64)  # Stop any direct calls to TGAColor_t
# _SENTINEL = nb.typed.List([True])  # Stop any direct calls to TGAColor_t
_SENTINEL: Final = -4242424242


class Format(IntEnum):
    GRAYSCALE = 1
    RGB = 3
    RGBA = 4


FORMAT_VALS: Final = {x.value for x in Format}

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

_TGA_COLOR_BYTE_CACHE: dict[tuple[int, ...], bytes] = {}
_TGA_COLOR_REPR_CACHE: dict[tuple[int, ...], str] = {}


# @dataclass(slots=True)
# @total_ordering
@jitclass([("_data", int64[:])])
class TGAColor_t:
    """
    Base type for a TGAColor

    Because we have lots of these, they are considerd quasi-immutable and can only be generated
    with a factory function `TGAColor()`. They are then cached so there's only one copy of each
    color combination in memory.
    """

    # _data: tuple[int, ...] = field(init=False)
    # _byte_data: bytes | None = field(default=None, init=False)
    # _repr: str | None = field(default=None, init=False)

    def __init__(
        self: Self,
        # numba cannot handle: *,
        bgra: tuple[int, ...],
        _guard: int | None = None,
    ) -> None:

        if _guard is not _SENTINEL:
            err_msg = "Only call TGAColor() method to get a TGAColor_t"
            raise TypeError(err_msg)

        # All error checking is handled in the helper...
        # numba cannot handle: self._data = np.array(bgra)
        self._data: np.ndarray = np.empty(len(bgra), dtype=np.int64)
        for i, v in enumerate(bgra):
            self._data[i] = v
        # Cached responses:
        # self._byte_data = None  # __bytes__
        # self._repr = None  # __repr__

    @property
    def rgba(self: Self) -> npt.NDArray[np.uint8]:
        """Converts to an array that matplotlib can use for imshow()"""
        # numpy and matplotlib expect 1, 3, or 4 byte data in the order of RGBA
        match self.bytespp:
            case 1:
                data = [uint8_t(self.b)]
            case 3:
                data = [uint8_t(self.r), uint8_t(self.g), uint8_t(self.b)]
            case 4:
                data = [uint8_t(self.r), uint8_t(self.g), uint8_t(self.b), uint8_t(self.a)]
            case _:
                err_msg = f"Cannot convert {self.bytespp}-byte colors to numpy image array"
                raise ValueError(err_msg)
        return np.array(data, dtype=uint8_t)

    @property
    def max_color(self: Self) -> uint8_t:
        return uint8_t(max(self._data[:3]))

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

    # numba can't handle: def resize(self: Self, bpp: int | uint8_t) -> TGAColor_t:
    def resize(self: Self, bpp: int) -> TGAColor_t:
        """Converts to a new pixel with a lower BPP"""
        # TODO: Better algorithm if RGB => Mono? Average maybe?
        # Special case: Assume alpha=255 and allow RGB => RGBA
        if (self.bytespp, bpp) == (3, 4):
            return TGAColor(self.b, self.g, self.r, self.a, 4)
        if bpp > self.bytespp:
            err_msg = f"Asked to increase BPP from {self.bytespp} to {bpp} and don't know how!"
            raise ValueError(err_msg)
        # numba cannot handle: return TGAColor(*(self._data[:bpp]), int(bpp))
        match bpp:
            case 1:
                return TGAColor(self[0], None, None, None, bpp)
            case 2:
                return TGAColor(self[0], self[1], None, None, bpp)
            case 3:
                return TGAColor(self[0], self[1], self[2], None, bpp)
            case _:
                return TGAColor(self[0], self[1], self[2], self[3], bpp)

    def __getitem__(self: Self, idx: int) -> uint8_t:
        """Allow raw indexing like C++ prefers"""
        if not (0 <= idx < len(self._data)):  # numba doesn't seem to implement this itself
            raise IndexError
        return uint8_t(self._data[idx])

    def __setitem__(self: Self, idx: int, val: uint8_t) -> None:
        raise FrozenInstanceError

    def __mul__(self: Self, other: float) -> TGAColor_t:
        """Scaling"""
        if not isinstance(other, (int, float, np.integer, np.floating)):
            return NotImplemented
        res = b"".join(min(round(v * other), 255).to_bytes() for v in self._data)
        return TGAColor_from_raw(res, bpp=self.bytespp, _allow2=True)[0]

    def __rmul__(self: Self, other: float) -> TGAColor_t:
        """Scaling"""
        return self * other

    def __truediv__(self: Self, other: float) -> TGAColor_t:
        """Scaling"""
        if isinstance(other, (int, float, np.integer, np.floating)):
            if other == 0:
                raise ZeroDivisionError
            res = b"".join(round(v / other).to_bytes() for v in self._data)
            return TGAColor_from_raw(res, bpp=self.bytespp, _allow2=True)[0]
        return NotImplemented

    def __le__(self: Self, other: object) -> bool:
        if not isinstance(other, TGAColor_t) or self.bytespp != other.bytespp:
            return NotImplemented
        return self._data <= other._data

    def __eq__(self: Self, other: object) -> bool:
        if not isinstance(other, TGAColor_t):
            return NotImplemented
        # This is almost 7X faster than hashing each and comparing
        return self._data == other._data

    def __hash__(self: Self) -> int:
        return hash(self._data)

    def __str__(self: Self) -> str:
        if True:  # self._data not in _TGA_COLOR_REPR_CACHE:
            try:
                bpp = self.bytespp
                res = [f"b={self[0]}"]
                if bpp >= 2:
                    res.append(f"g={self[1]}")
                if bpp >= 3:
                    res.append(f"r={self[2]}")
                if bpp == 4:
                    res.append(f"a={self[3]}")
                # _TGA_COLOR_REPR_CACHE[self._data] = "TGAColor_t(" + ", ".join(res) + f", bpp={self.bytespp})"
                return "TGAColor_t(" + ", ".join(res) + f", bpp={self.bytespp})"
            except Exception:
                pass
        return "TGAColor_t(NOT FULLY INITIALIZED)"

        # return _TGA_COLOR_REPR_CACHE[self._data]

    def to_bytes(self: Self) -> bytes:
        if self._data not in _TGA_COLOR_BYTE_CACHE:
            if any(not (0 <= c <= 255) for c in self._data):
                with objmode():
                    logger.warning("Out-of-range value for color byte; clipping (%s)", self._data)
            _TGA_COLOR_BYTE_CACHE[self._data] = b"".join(max(0, min(c, 255)).to_bytes() for c in self._data)
        return _TGA_COLOR_BYTE_CACHE[self._data]


_ = """
# https://numba-how-to.readthedocs.io/en/latest/tuples.html
#
# The idea here is to wrap a typed.Dict in another type, the "TupleKeyDictType".
# The purpose of this is so that operations like __getitem__ and __setitem__
# can be proxied through functions that call `hash` on the key. This makes it
# possible to have something that behaves like a dictionary, but supports
# heterogeneous keys (tuples of varying size/type).
#
# Define a new type and register it
@structref.register
class TupleKeyDictType(nbt.StructRef):
    def preprocess_fields(self, fields):
        res = tuple((name, nbt.unliteral(typ)) for name, typ in fields)
        assert None not in res, f"FUCK {res=}"
        return res


# Define the Python side proxy class
class TupleKeyDict(structref.StructRefProxy):
    @property
    def wrapped_dict(self):
        return TupleKeyDict_get_wrapped_dict(self)


# Set up the wiring for it, "wrapped_dict" is the only member in the "struct"
# and it refers to the typed.Dict instance in use
structref.define_proxy(TupleKeyDict, TupleKeyDictType, ["wrapped_dict"])


# Overload operator.getitem for the TupleKeyDictType, note how defers the look
# up to the wrapped_dict member and hashes the key
@nbex.overload(operator.getitem)
def ol_tkd_getitem(inst, key):
    if isinstance(inst, TupleKeyDictType):

        def impl(inst, key):
            return inst.wrapped_dict[hash(key)]

        return impl
    return None


# Overload operator.setitem for the TupleKeyDictType, again, it's hashing the
# key before use.
@nbex.overload(operator.setitem)
def ol_tkd_setitem(inst, key, value):
    if isinstance(inst, TupleKeyDictType):

        def impl(inst, key, value):
            inst.wrapped_dict[hash(key)] = value

        return impl
    return None


# I added this; original didn't have it:
@nbex.overload(operator.contains)
def ol_tkd_contains(inst, key):
    if isinstance(inst, TupleKeyDictType):

        def impl(inst, key):
            # assert None not in key
            return hash(key) in inst.wrapped_dict

        return impl
    return None


# _TGA_COLOR_CACHE_base = nb.typed.Dict.empty(nbt.intp, TGAColor_t.class_type.instance_type)
# _TGA_COLOR_CACHE = TupleKeyDict(_TGA_COLOR_CACHE_base)
# dict[tuple[int, ...], TGAColor_t] = {}
"""
_TGA_COLOR_CACHE_KEY_TYPE = nb.types.Tuple((nb.uint8, nb.int64, nb.int64, nb.int64, nb.int64))
type _TGA_COLOR_CACHE_KEY_TYPE_t = tuple[uint8_t, int, int, int, int]  # Native python
_TGA_COLOR_CACHE_VAL_TYPE = TGAColor_t.class_type.instance_type
# _TGA_COLOR_CACHE = nb.typed.Dict.empty(key_type=_TGA_COLOR_CACHE_KEY_TYPE, value_type=_TGA_COLOR_CACHE_VAL_TYPE)


@njit
def _key_mapper(bgra: list[int]) -> _TGA_COLOR_CACHE_KEY_TYPE_t:
    # numba doesn't "short circuit" so need to explicitly check length each time
    bpp = len(bgra)
    b = bgra[0]
    g = r = a = -1
    if bpp >= 2:
        g = bgra[1]
    if bpp >= 3:
        r = bgra[2]
    if bpp >= 4:
        a = bgra[3]
    return (uint8_t(bpp), b, g, r, a)


cc_spec = [("cache", nbt.DictType(_TGA_COLOR_CACHE_KEY_TYPE, _TGA_COLOR_CACHE_VAL_TYPE))]


@jitclass(cc_spec)
class ColorCache:
    def __init__(self: Self) -> None:
        # Initialize the empty typed dict inside the jitclass constructor
        self.cache = nb.typed.Dict.empty(key_type=_TGA_COLOR_CACHE_KEY_TYPE, value_type=_TGA_COLOR_CACHE_VAL_TYPE)

    def generate(self: Self, bgra: list[int]) -> TGAColor_t:
        key: Final = _key_mapper(bgra)
        with objmode():
            logger.error(f"key {key}")
            logger.error(f"key is in  {key in self.cache}")
        if key not in self.cache:
            with objmode():
                logger.error("Generating")
            self.cache[key] = TGAColor_t(bgra, _SENTINEL)
            with objmode():
                logger.error(f"Generated {key}//{self.cache[key]}")
        with objmode():
            logger.error(f"returning {self.cache[key]} {hex(id(self.cache[key]))}")
        return self.cache[key]

    # def __contains__(self: Self, key: _TGA_COLOR_CACHE_KEY_TYPE_t) -> bool:
    #     return key in self.cache

    # def __getitem__(self: Self, key: _TGA_COLOR_CACHE_KEY_TYPE_t) -> TGAColor_t.class_type.instance_type:
    #     return self.cache[key]

    # def __setitem__(self: Self, key: _TGA_COLOR_CACHE_KEY_TYPE_t, val: TGAColor_t.class_type.instance_type) -> None:
    #     self.cache[key] = val


_TGA_COLOR_CACHE = ColorCache()

_ = '''
# @cache
@njit
def _TGAColor_factory(bgra: list[int], _cache: nb.typed.Dict = _TGA_COLOR_CACHE) -> TGAColor_t:
    """Helper for the TGAColor() factory"""
    assert None not in bgra
    # if _cache is None:
    #     _cache = _TGA_COLOR_CACHE
    # if len(bgra) < 5:
    #     raise ValueError("WTF", bgra)
    # if len(bgra) >= 5:
    #     raise ValueError("WTF2")
    # numba cannot handle: return TGAColor_t(bgra=bgra, _guard=_SENTINEL)
    # key = TupleKeyDictType().preprocess_fields(bgra)
    # return TGAColor_t([255, 255, 255, 255], _SENTINEL)
    key: Final = _key_mapper(bgra)
    if key not in _cache:
        _cache[key] = TGAColor_t(bgra, _SENTINEL)
    return _cache[key]
'''


@njit
def TGAColor(
    b: int | None = None,
    g: int | None = None,
    r: int | None = None,
    a: int | None = None,
    # numba cannot handle: *,
    bpp: int | None = None,
) -> TGAColor_t:
    """
    End user interface to create a TGAColor_t

    Factory helper function that caches since they are "immutable" (but not really)
    """
    # The bpp logic is here to fix caching of specified or not...
    # Special case: Default constructor is (b=0, bpp=1)
    # numba needs me to expand this explicitly... even the expanded tuple fails
    # LLVM error "error: icmp requires integer operands"
    # if (b, g, r, a, bpp) == (None,) * 5:
    if b is None and g is None and r is None and a is None and bpp is None:
        b = 0

    bpp_: uint8_t
    if a is not None:
        bpp_ = uint8_t(4)
    elif r is not None:
        bpp_ = uint8_t(3)
    elif g is not None:
        bpp_ = uint8_t(2)
    else:
        bpp_ = uint8_t(1)

    if bpp is not None and bpp != bpp_:
        with objmode():
            logger.debug("Computed bpp %d but user override %d", bpp_, bpp)
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
    if not (1 <= bpp_ <= 4):
        err_msg = f"Invalid bpp={bpp_} given!"
        raise ValueError(err_msg)

    with objmode(res=_TGA_COLOR_CACHE_VAL_TYPE):
        res = _TGA_COLOR_CACHE.generate(nb.typed.List([b, g, r, a][:bpp_]))
    return res
    # return _TGAColor_factory([b, g, r, a][:bpp_])
    # match bpp_:
    #     case 1:
    #         return _TGAColor_factory((b,))
    #     case 2:
    #         return _TGAColor_factory((b, g))
    #     case 3:
    #         return _TGAColor_factory((b, g, r))
    #     case _:
    #         return _TGAColor_factory((b, g, r, a))


def TGAColor_from_raw(data: bytes | bytearray, *, bpp: int, _allow2: bool = False) -> list[TGAColor_t]:
    """Factory helper - take bytes and get a list of TGAColors"""
    if (ld := len(data)) % bpp:
        warn(f"Possibly bad read of {ld} bytes at {bpp} bpp = remainder {ld % bpp}", stacklevel=2)

    match bpp:
        case Format.GRAYSCALE:
            return [TGAColor(b=v, bpp=1) for v in data]
        case Format.RGB:
            return [TGAColor(b=b, g=g, r=r, bpp=3) for (b, g, r) in batched(data, 3)]
        case Format.RGBA:
            return [TGAColor(b=b, g=g, r=r, a=a, bpp=4) for (b, g, r, a) in batched(data, 4)]
        case _:
            pass
    if _allow2 and bpp == 2:  # Special mode for scaling tests only
        return [TGAColor(b=b, g=g, bpp=2) for (b, g) in batched(data, 2)]
    msg = f"Cannot handle {bpp} BPP"
    raise NotImplementedError(msg)


TI = TypeVar("TI", bound="TGAImage")

_tga_image_spec = [
    ("width", int64),
    ("height", int64),
    ("bpp", nb.uint8),
    ("npdata", nbt.ListType(TGAColor_t.class_type.instance_type)),
    ("was_hflipped", nb.boolean),
    ("was_vflipped", nb.boolean),
    ("was_rle", nb.boolean),
]


@jitclass(_tga_image_spec)
class TGAImage:
    """An in-memory TGAImage"""

    def __init__(self: Self, w: int = 0, h: int = 0, bpp: int = 4, c: TGAColor_t | None = None) -> None:
        self.width = w
        self.height = h
        self.bpp: uint8_t = uint8_t(bpp)
        fill_value = c if c is not None else TGAColor(*([0] * bpp), bpp=bpp)
        # Reminder: numpy array is arr[rows][cols] so maps to arr[y][x] not arr[x][y]
        # Origin is in bottom left corner of image
        self.npdata: np.ndarray = np.full(shape=(h, w), dtype=TGAColor_t, fill_value=fill_value)

        # These are pretty much for testing purposes only:
        self.was_hflipped = False
        self.was_vflipped = False
        self.was_rle = False
        # self._lock = threading.Lock()

    @property
    def rgba(self: Self) -> npt.NDArray[np.uint8]:
        """Converts to an array that matplotlib can use for imshow()"""
        rgba = np.vectorize(lambda px: px.rgba, signature=f"()->({self.bpp})")
        return rgba(self.npdata)

    def __array__(  # ruff: ignore[bad-dunder-method-name]
        self: Self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray:
        # This allows numpy to treat us as "native" data
        if dtype is not None:
            err_msg = f"I don't know how to convert myself to {dtype}!"
            raise ValueError(err_msg)
        return self.rgba

    # @classmethod
    # def read_tga_file(  # ruff: ignore[custom-type-var-for-self]
    #     cls: type[TI],
    #     filename: str | Path,
    # ) -> TI:
    #     """Read a file on disk into memory"""
    #     header = np.fromfile(filename, dtype=TGAHeader, count=1)[0]
    #     w = header["width"]
    #     h = header["height"]
    #     bpp = int(header["bitsperpixel"]) >> 3
    #     data_size: Final = int(w) * int(h) * bpp
    #     dtc = header["datatypecode"]
    #     imgd = header["imagedescriptor"]
    #     assert w > 0, f"Interpreted width {w} is invalid!"
    #     assert h > 0, f"Interpreted height {h} is invalid!"
    #     assert bpp in FORMAT_VALS, f"Interpreted bits-per-pixel {bpp} is invalid!"
    #     assert dtc in {2, 3, 10, 11}, f"Unknown file format '{dtc}'!"
    #     res = cls(w=int(w), h=int(h), bpp=bpp)
    #     # Read the data without the header
    #     raw_data = bytearray(Path(filename).read_bytes())[TGAHeader.itemsize :]
    #     if dtc in {10, 11}:
    #         # RLE data
    #         raw_data = res.load_rle_data(raw_data)
    #         res.was_rle = True
    #     if dtc in {2, 3}:
    #         # Not RLE data
    #         # trailing = raw_data[data_size:]
    #         # Truncate it
    #         del raw_data[data_size:]
    #     res.npdata = np.asarray(TGAColor_from_raw(raw_data, bpp=bpp), dtype=TGAColor_t).reshape(h, w)
    #     if not (imgd & 0x20):
    #         res.flip_vertically()
    #         res.was_vflipped = True
    #     if imgd & 0x10:
    #         res.flip_horizontally()
    #         res.was_hflipped = True
    #     return res

    @property
    def _raw_payload(self: Self) -> bytes:
        self.verify()
        all_bytes: Final = np.vectorize(bytes)(self.npdata)
        return all_bytes.ravel().tobytes()

    def verify(self: Self) -> None:
        """Checks that all pixels are the right size"""
        # Generically named if we want to add other verification later
        # Check pixels have correct BPP:
        all_bytespp: Final = np.vectorize(lambda x: x.bytespp)(self.npdata)
        # Internally addressed as (y, x) so swap in report
        errs: list[str] = [
            f"Pixel (x, y)={bad_bpp[1], bad_bpp[0]} has bytespp={all_bytespp[*bad_bpp]} not {self.bpp}"
            for bad_bpp in np.argwhere(all_bytespp != self.bpp).tolist()
        ]
        if errs:
            err_msg = f"{len(errs)} verification errors!\n" + "\n".join(errs)
            raise ValueError(err_msg)

    # def _variable_name(self: Self, *, frame: FrameType | None = None) -> str | None:
    #     """Try to figure out what user calls us"""
    #     if frame is None:
    #         frame = inspect.currentframe()
    #     if ((this_frame := frame) is None) or ((caller_frame := this_frame.f_back) is None):
    #         return None
    #     # Get all variables in the caller's frame
    #     namespace: Final[dict[str, object]] = {**caller_frame.f_globals, **caller_frame.f_locals}
    #     this: Final[list[str]] = [k for k, v in namespace.items() if v is self]
    #     if this == ["self"]:  # Go back farther
    #         return self._variable_name(frame=caller_frame)
    #     return this[0] if this else None

    # def write_tga_file(self: Self, filename: str | Path, vflip: bool = True, rle: bool = True) -> None:
    #     """Writes image to disk"""
    #     # logger.debug("Writing to file %s...", filename)
    #     self.verify()
    #     developer_area_ref: Final[bytes] = b"\0\0\0\0"
    #     extension_area_ref: Final[bytes] = b"\0\0\0\0"
    #     footer: Final[bytes] = b"TRUEVISION-XFILE.\0"
    #     header = np.zeros(1, dtype=TGAHeader)
    #     header["bitsperpixel"] = self.bpp << 3
    #     header["width"] = self.width
    #     header["height"] = self.height
    #     header["datatypecode"] = (10 if rle else 2) + (1 if self.bpp == Format.GRAYSCALE else 0)
    #     header["imagedescriptor"] = 0 if vflip else 0x20  # top-left or bottom-left origin
    #     with Path(filename).open("wb") as out:
    #         out.write(header.tobytes())
    #         out.write(self.unload_rle_data() if rle else self._raw_payload)
    #         out.write(developer_area_ref)
    #         out.write(extension_area_ref)
    #         out.write(footer)
    #     if name := self._variable_name():
    #         logger.info("Wrote '%s' to file %s: %s", name, filename, self)
    #     else:
    #         logger.info("Wrote to file %s: %s", filename, self)

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
            if (old.bytespp, self.bpp) != (3, 4):  # Don't warn if just adding alpha channel
                warn(f"Pixel write at ({x}, {y}) changed bpp: was {old} now {c}", stacklevel=2)
        if not (0 <= x < self.width) or not (0 <= y < self.height):
            with objmode():
                logger.warning("TGAImage.set(%s, %s) invalid: Image is %d x %d", x, y, self.width, self.height)
                return
        # with self._lock:
        self.npdata[y, x] = c

    def load_rle_data(self: Self, in_: bytes | bytearray) -> bytearray:
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
            return bytearray(raw_data.getvalue())

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
                        # logger.debug("Ending on a raw run...")
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

    # def plot(self: Self, plot: bool = True, *, _test_mode: bool = False) -> None:
    #     """Plot an image using matplotlib"""
    #     if not plot:
    #         return
    #     if matplotlib_import_err is not None:
    #         logger.warning(
    #             "Could not plot %dx%dx%d image - matplotlib failed import: %s",
    #             self.width,
    #             self.height,
    #             self.bpp,
    #             matplotlib_import_errmatplotlib_import_err,
    #         )
    #         return
    #     assert plt is not None  # Keep typing happy
    #     img = plt.imshow(self, origin="lower")
    #     if (
    #         (this := self._variable_name())
    #         and ((fig := img.get_figure()) is not None)
    #         and ((manager := fig.canvas.manager) is not None)
    #     ):
    #         manager.set_window_title(this)
    #         # plt.title(this)
    #     if not _test_mode:
    #         plt.show()

    def set_max(self: Self, max_val: uint8_t | int) -> uint8_t:
        """Helper function to scale up/down all colors"""
        # logger.debug("set_max(%d)", max_val)
        max_color: Final[uint8_t] = max(px.max_color for px in self.npdata.ravel())
        if max_color == 0:
            with objmode():
                logger.warning("Asked to manipulate all-black image!")
            return uint8_t(0)
        scaling: Final = max_val / max_color
        # logger.debug("Scaling factor: %s", scaling)
        scaled: Final = np.vectorize(lambda px: px * scaling)
        self.npdata = scaled(self.npdata)
        # logger.debug("Done scaling")
        return max_color

    def brighten(self: Self) -> None:
        """Helper function to scale up all colors - With Whiter Whites! (TM)"""
        self.set_max(255)

    def __str__(self: Self) -> str:
        return f"{self.width}x{self.height}/{self.bpp}"


# Tools broken out:
@njit
def read_tga_file(filename: str | Path) -> TGAImage:
    """Read a file on disk into memory"""
    header = np.fromfile(filename, dtype=TGAHeader, count=1)[0]
    w = header["width"]
    h = header["height"]
    bpp = int(header["bitsperpixel"]) >> 3
    data_size: Final = int(w) * int(h) * bpp
    dtc = header["datatypecode"]
    imgd = header["imagedescriptor"]
    assert w > 0, f"Interpreted width {w} is invalid!"
    assert h > 0, f"Interpreted height {h} is invalid!"
    assert bpp in FORMAT_VALS, f"Interpreted bits-per-pixel {bpp} is invalid!"
    assert dtc in {2, 3, 10, 11}, f"Unknown file format '{dtc}'!"
    res = TGAImage(w=int(w), h=int(h), bpp=bpp)
    # Read the data without the header
    raw_data = bytearray(Path(filename).read_bytes())[TGAHeader.itemsize :]
    if dtc in {10, 11}:
        # RLE data
        raw_data = res.load_rle_data(raw_data)
        res.was_rle = True
    if dtc in {2, 3}:
        # Not RLE data
        # trailing = raw_data[data_size:]
        # Truncate it
        del raw_data[data_size:]
    res.npdata = np.asarray(TGAColor_from_raw(raw_data, bpp=bpp), dtype=TGAColor_t).reshape(h, w)
    if not (imgd & 0x20):
        res.flip_vertically()
        res.was_vflipped = True
    if imgd & 0x10:
        res.flip_horizontally()
        res.was_hflipped = True
    return res


def write_tga_file(img: TGAImage, filename: str | Path, vflip: bool = True, rle: bool = True) -> None:
    """Writes image to disk"""
    logger.debug("Writing %s to file %s...", img, filename)
    img.verify()
    developer_area_ref: Final[bytes] = b"\0\0\0\0"
    extension_area_ref: Final[bytes] = b"\0\0\0\0"
    footer: Final[bytes] = b"TRUEVISION-XFILE.\0"
    header = np.zeros(1, dtype=TGAHeader)
    header["bitsperpixel"] = img.bpp << 3
    header["width"] = img.width
    header["height"] = img.height
    header["datatypecode"] = (10 if rle else 2) + (1 if img.bpp == Format.GRAYSCALE else 0)
    header["imagedescriptor"] = 0 if vflip else 0x20  # top-left or bottom-left origin
    with Path(filename).open("wb") as out:
        out.write(header.tobytes())
        out.write(img.unload_rle_data() if rle else img._raw_payload)
        out.write(developer_area_ref)
        out.write(extension_area_ref)
        out.write(footer)
    logger.info("Wrote to file %s: %s", filename, img)


def plot(img: TGAImage, plot: bool = True, *, _test_mode: bool = False) -> None:
    """Plot an image using matplotlib"""
    if not plot:
        return
    if matplotlib_import_err is not None:
        logger.warning(
            "Could not plot %dx%dx%d image - matplotlib failed import: %s",
            img.width,
            img.height,
            img.bpp,
            matplotlib_import_err,
        )
        return
    assert plt is not None  # Keep typing happy
    img = plt.imshow(img, origin="lower")
    # if (
    #     (this := img._variable_name())
    #     and ((fig := img.get_figure()) is not None)
    #     and ((manager := fig.canvas.manager) is not None)
    # ):
    #     manager.set_window_title(this)
    #     # plt.title(this)
    if not _test_mode:
        plt.show()


white: Final = TGAColor(255, 255, 255, 255).resize(Format.RGB)  # Attention: BGRA order
green: Final = TGAColor(0, 255, 0, 255).resize(Format.RGB)
red: Final = TGAColor(0, 0, 255, 255).resize(Format.RGB)
blue: Final = TGAColor(255, 128, 64, 255).resize(Format.RGB)
yellow: Final = TGAColor(0, 200, 255, 255).resize(Format.RGB)
black: Final = TGAColor(0, 0, 0)
