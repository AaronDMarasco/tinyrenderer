"""##Custom types used by the Python port of `tinyrenderer`"""

from __future__ import annotations

import copy
import itertools
import math
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast, overload, override

import numpy  # ruff: ignore[unconventional-import-alias]  # Import as np conflicts with property named np
import numpy.typing as npt

from .tgaimage import TGAColor, TGAImage

if TYPE_CHECKING:
    from collections.abc import Sequence

type Matrix2f = numpy.ndarray[tuple[Literal[2], Literal[2]], numpy.dtype[numpy.float64]]
type Matrix3f = numpy.ndarray[tuple[Literal[3], Literal[3]], numpy.dtype[numpy.float64]]
type Matrix4f = numpy.ndarray[tuple[Literal[4], Literal[4]], numpy.dtype[numpy.float64]]

type MatrixLike = Matrix2f | Matrix3f | Matrix4f


@overload
def empty_matrix(rc: Literal[2], /) -> Matrix2f: ...
@overload
def empty_matrix(rc: Literal[3], /) -> Matrix3f: ...
@overload
def empty_matrix(rc: Literal[4], /) -> Matrix4f: ...
def empty_matrix(rc: int, /) -> MatrixLike:
    """Provide a new empty matrix of size rc (rows/columns)"""
    if not (2 <= rc <= 4):
        err_msg = f"Couldn't determine type {rc}!"
        raise ValueError(err_msg)
    return cast("MatrixLike", numpy.zeros(shape=(rc, rc), dtype=float))


@dataclass(slots=True)
class ZBuffer:
    """Raw list of Z values with set function that only allows increasing values to be written"""

    vals: list[list[float]] = field(init=False)
    _lock: threading.Lock = field(init=False)

    def __init__(self: Self, *, width: int, height: int) -> None:
        self.vals = cast(
            "list[list[float]]",
            numpy.full((width, height), numpy.nan, dtype=float).tolist(),
        )
        self._lock = threading.Lock()

    def __deepcopy__(self: Self, memo: dict) -> ZBuffer:
        # Cannot copy lock
        result = ZBuffer(width=len(self.vals), height=len(self.vals[0]))
        result.vals = copy.deepcopy(self.vals, memo)
        return result

    @property
    def array(self: Self) -> list[float]:
        """Dump a straight array of our values"""
        return list(itertools.chain.from_iterable(self.vals))

    def fix_nan(self: Self, val: float) -> None:
        """Set all NaN values to a given value"""

        def nan_to_val(v: float) -> float:
            return val if math.isnan(v) else v

        self.vals = [[nan_to_val(y) for y in x] for x in self.vals]

        # for val_x in self.vals:
        #     for val_y in val_x:
        #         if math.isnan(val_y):
        #             val_y = val

    def try_set(self: Self, x: int, y: int, val: float) -> bool:
        """Atomic-ish set and get if yours was written"""
        with self._lock:
            if val <= self.vals[x][y]:
                return False
            self.vals[x][y] = val
            return True

    def to_tga(self: Self, *, allow_nan: bool = True, nan_val: int = -1000) -> TGAImage:
        """If allow_nan is not set, any unset values will explode"""
        nparray = numpy.array(self.vals, dtype=float)
        if not allow_nan and numpy.isnan(nparray).any():
            err_msg = "ZBuffer had NaN and not told to assume zero!"
            raise ValueError(err_msg)
        nparray[numpy.isnan(nparray)] = nan_val  # We used to use zero, but -1000 is the default in later lessons
        width, height = nparray.shape
        # Want to scale from 0..255
        min_val: Final = nparray.min()
        max_val: Final = nparray.max()
        if min_val == max_val:
            # Just regenerate an empty image
            return TGAImage(w=width, h=height, bpp=1, c=TGAColor(0))
        # Otherwise, create scaled image
        fb = TGAImage(w=width, h=height, bpp=1)
        normalized = (nparray - min_val) * 255 / (max_val - min_val)
        for x in range(width):
            for y in range(height):
                fb.set(x, y, TGAColor(round(normalized[x, y])))
        return fb

    def __getitem__(self: Self, idx: int) -> list[float]:
        """This interface is READ ONLY (due to lock being removed)"""
        with self._lock:
            return deepcopy(self.vals[idx])

    def __setitem__(self: Self, _idx: int, _val: object) -> None:
        err_msg = "Use try_set() to write with locks"
        raise TypeError(err_msg)


@dataclass(frozen=True, slots=True)
class _VectorBase(ABC):
    """vec2 from the C++ code, roughly ported"""

    x: float
    y: float

    @property
    @abstractmethod
    def array(self: Self) -> list[float]:
        """Convert to standard python list"""

    @staticmethod
    @abstractmethod
    def new_zero() -> _VectorBase:
        """Create a one-off with all zeros"""
        # This is just because I am not sure I want the default constructor to exist...

    @staticmethod
    @abstractmethod
    def new_nan() -> _VectorBase:
        """Create a one-off with all NaN"""
        # This is just because I am not sure I want the default constructor to exist...

    @staticmethod
    @abstractmethod
    def new_zeros(count: int) -> Sequence[_VectorBase]:
        """Create a list of one-offs with all zeros"""

    @staticmethod
    @abstractmethod
    def new_nans(count: int) -> Sequence[_VectorBase]:
        """Create a list of one-offs with all NaN"""

    @property
    def norm(self: Self) -> float:
        """Compute norm (from C++)"""
        return float(numpy.sqrt(self.np @ self.np))

    @property
    def normalized(self: Self) -> Self:
        """Make magnitude of a vector 1"""
        denom: Final = numpy.linalg.norm(self.np)
        if denom == 0:
            if all(x == 0 for x in self.array):
                return self
            raise ZeroDivisionError
        return cast("Self", self.from_np(self.np / denom))

    @property
    def np(self: Self) -> npt.NDArray[numpy.float64]:
        """The numpy representation of this 'vector'"""
        return numpy.array(self.array, dtype=float)

    def __array__(  # ruff: ignore[bad-dunder-method-name]
        self: Self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[numpy.float64]:
        # This allows numpy to treat us as "native" data without always calling vec.np
        return numpy.array(self.array, dtype=dtype, copy=copy)

    @staticmethod
    @abstractmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> _VectorBase:
        """Create a vector from a numpy array"""

    @abstractmethod
    def __add__(self: Self, other: Self) -> Self: ...

    @abstractmethod
    def __sub__(self: Self, other: Self) -> Self: ...

    @overload
    def __mul__(self: Self, other: float) -> Self: ...
    @overload
    def __mul__(self: Self, other: Self) -> float: ...

    def __mul__(self: Self, other: object) -> Any:
        """Dot product or scaling"""
        if isinstance(other, (int, float, numpy.integer, numpy.floating)):
            return cast("Self", self.from_np(other * self.np))
        if not isinstance(other, _VectorBase):
            return NotImplemented
        return numpy.dot(self.array, other.array)

    def __truediv__(self: Self, other: float) -> Self:
        """Scaling"""
        if isinstance(other, (int, float, numpy.integer, numpy.floating)):
            if other == 0:
                raise ZeroDivisionError
            return cast("Self", self.from_np(self.np / other))
        return NotImplemented


@dataclass(frozen=True, slots=True)
class vec2(_VectorBase):  # ruff: ignore[invalid-class-name]
    @override
    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y]

    @override
    @staticmethod
    def new_zero() -> vec2:
        return vec2(x=0, y=0)

    @override
    @staticmethod
    def new_nan() -> vec2:
        return vec2(x=math.nan, y=math.nan)

    @override
    @staticmethod
    def new_zeros(count: int) -> Sequence[vec2]:
        return [vec2.new_zero() for _ in range(count)]

    @override
    @staticmethod
    def new_nans(count: int) -> Sequence[vec2]:
        return [vec2.new_nan() for _ in range(count)]

    @override
    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec2:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (2,), f"Invalid array shape {array.shape}!"
        assert array.dtype == float, f"Invalid array type {array.dtype}!"
        return vec2(x=float(array[0]), y=float(array[1]))

    def __add__(self: Self, other: _VectorBase) -> vec2:
        if not isinstance(other, vec2):
            return NotImplemented
        return vec2(self.x + other.x, self.y + other.y)

    def __sub__(self: Self, other: _VectorBase) -> vec2:
        if not isinstance(other, vec2):
            return NotImplemented
        return vec2(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class vec3(_VectorBase):  # ruff: ignore[invalid-class-name]
    z: float

    def cross(self: Self, other: vec3) -> vec3:
        """Cross-product of two vectors"""
        if not isinstance(other, vec3):
            return NotImplemented
        return cast("Self", self.from_np(numpy.cross(self, other)))

    @override
    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y, self.z]

    @override
    @staticmethod
    def new_zero() -> vec3:
        return vec3(x=0, y=0, z=0)

    @override
    @staticmethod
    def new_nan() -> vec3:
        return vec3(x=math.nan, y=math.nan, z=math.nan)

    @override
    @staticmethod
    def new_zeros(count: int) -> Sequence[vec3]:
        return [vec3.new_zero() for _ in range(count)]

    @override
    @staticmethod
    def new_nans(count: int) -> Sequence[vec3]:
        return [vec3.new_nan() for _ in range(count)]

    @override
    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec3:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (3,), f"Invalid array shape {array.shape}!"
        assert array.dtype == float, f"Invalid array type {array.dtype}!"
        return vec3(x=float(array[0]), y=float(array[1]), z=float(array[2]))

    @property
    def xy(self: Self) -> vec2:
        return vec2(x=self.x, y=self.y)

    def __add__(self: Self, other: _VectorBase) -> vec3:
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self: Self, other: _VectorBase) -> vec3:
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True, slots=True)
class vec4(vec3):  # ruff: ignore[invalid-class-name]
    w: float

    @override
    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y, self.z, self.w]

    @override
    @staticmethod
    def new_zero() -> vec4:
        return vec4(x=0, y=0, z=0, w=0)

    @override
    @staticmethod
    def new_nan() -> vec4:
        return vec4(x=math.nan, y=math.nan, z=math.nan, w=math.nan)

    @override
    @staticmethod
    def new_zeros(count: int) -> Sequence[vec4]:
        return [vec4.new_zero() for _ in range(count)]

    @override
    @staticmethod
    def new_nans(count: int) -> Sequence[vec4]:
        return [vec4.new_nan() for _ in range(count)]

    @override
    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec4:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (4,), f"Invalid array shape {array.shape}!"
        assert array.dtype == float, f"Invalid array type {array.dtype}!"
        return vec4(x=float(array[0]), y=float(array[1]), z=float(array[2]), w=float(array[3]))

    @staticmethod
    def from_vec3(v: vec3, *, w: float = 1) -> vec4:
        assert isinstance(v, vec3)
        assert isinstance(w, (int, float))
        return vec4(x=v.x, y=v.y, z=v.z, w=float(w))

    @property
    def xyz(self: Self) -> vec3:
        return vec3(x=self.x, y=self.y, z=self.z)

    def __add__(self: Self, other: _VectorBase) -> vec4:
        if not isinstance(other, vec4):
            return NotImplemented
        return vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self: Self, other: _VectorBase) -> vec4:
        if not isinstance(other, vec4):
            return NotImplemented
        return vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)


type Triangle = tuple[vec4, vec4, vec4]  # Used in Lesson 7 and beyond
