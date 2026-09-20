"""##Custom types used by the Python port of `tinyrenderer`"""

from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import math

# import operator
import threading

# from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal, Self, cast, overload

import numba.extending as nbex
import numpy  # ruff: ignore[unconventional-import-alias]  # Import as np conflicts with property named np
import numpy.typing as npt
from icecream import ic
from numba import float64, njit
from numba import types as nbt
from numba.experimental import jitclass

from .tgaimage import TGAColor, TGAImage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import override
else:

    def override(func):  # ruff: ignore[missing-return-type-undocumented-public-function, missing-type-function-argument]
        return func


top_logger = logging.getLogger(__name__)
zb_logger = logging.getLogger(f"{__name__}.ZBuffer")

type Matrix2f = numpy.ndarray[tuple[Literal[2], Literal[2]], numpy.dtype[numpy.float64]]
type Matrix3f = numpy.ndarray[tuple[Literal[3], Literal[3]], numpy.dtype[numpy.float64]]
type Matrix4f = numpy.ndarray[tuple[Literal[4], Literal[4]], numpy.dtype[numpy.float64]]

type MatrixLike = Matrix2f | Matrix3f | Matrix4f

# The threading didn't seem to help, and adding the deepcopy made the ZBuffer.__getitem__ take 54s vs. 0.01s
THREAD_SAFE: Final[bool] = False


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
    _cm: contextlib.AbstractContextManager = field(init=False)

    def __init__(self: Self, *, width: int, height: int) -> None:
        assert width > 0
        assert height > 0
        self.vals = cast(
            "list[list[float]]",
            numpy.full((width, height), numpy.nan, dtype=float).tolist(),
        )
        if THREAD_SAFE:
            self._cm = threading.Lock()
            zb_logger.debug("%s in thread-safe mode", self)
        else:
            self._cm = contextlib.nullcontext()
            zb_logger.debug("%s NOT in thread-safe mode", self)

    def __deepcopy__(self: Self, memo: dict) -> ZBuffer:
        # Cannot copy lock
        result = ZBuffer(width=len(self.vals), height=len(self.vals[0]))
        result.vals = copy.deepcopy(self.vals, memo)
        return result

    @property
    def array(self: Self) -> list[float]:
        """Dump a straight array of our values"""
        return list(itertools.chain.from_iterable(self.vals))

    @property
    def height(self: Self) -> int:
        return len(self.vals[0])

    @property
    def width(self: Self) -> int:
        return len(self.vals)

    def fix_nan(self: Self, val: float) -> None:
        """Set all NaN values to a given value"""

        def nan_to_val(v: float) -> float:
            return val if math.isnan(v) else v

        self.vals = [[nan_to_val(y) for y in x] for x in self.vals]

    def try_set(self: Self, x: int, y: int, val: float) -> bool:
        """Atomic-ish set and get if yours was written"""
        with self._cm:
            if not (0 <= x < self.width) or not (0 <= y < self.height):
                zb_logger.warning("Ignoring write to (x, y) = (%d, %d) (max %d, %d)", x, y, self.width, self.height)
                return False
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
        if THREAD_SAFE:
            return deepcopy(self.vals[idx])
        return self.vals[idx]

    def __setitem__(self: Self, _idx: int, _val: object) -> None:
        err_msg = "Use try_set() to write with locks"
        raise TypeError(err_msg)

    def __str__(self: Self) -> str:
        return f"ZBuffer@{hex(id(self))} ({self.width}x{self.height})"


# @jitclass
# @dataclass(frozen=True, slots=True)
class _VectorBase:  # (ABC):
    """vec2 from the C++ code, roughly ported"""

    # x: float
    # y: float

    def __init__(self: Self, x: float, y: float) -> None:
        # self._data: list[float] = [x, y]
        self._data = numpy.array([x, y], dtype=numpy.float64)
        # self.x: float = x
        # self.y: float = y
        #

    def __len__(self: Self) -> int:
        return len(self._data)

    def __eq__(self: Self, other: _VectorBase) -> bool:
        return (self._data == other._data).all()

    @property
    def x(self: Self) -> float:
        return self._data[0]

    @property
    def y(self: Self) -> float:
        return self._data[1]

    @property
    # @abstractmethod
    def array(self: Self) -> list[float]:
        """Convert to standard python list"""
        return list(self._data)  # Copy

    # @staticmethod
    # @abstractmethod
    # def new_zero() -> _VectorBase:
    #     """Create a one-off with all zeros"""
    #     # This is just because I am not sure I want the default constructor to exist...
    #     raise NotImplementedError

    # @staticmethod
    # # @abstractmethod
    # def new_nan() -> _VectorBase:
    #     """Create a one-off with all NaN"""
    #     # This is just because I am not sure I want the default constructor to exist...
    #     raise NotImplementedError

    # @staticmethod
    # # @abstractmethod
    # def new_zeros(count: int) -> Sequence[_VectorBase]:
    #     """Create a list of one-offs with all zeros"""
    #     raise NotImplementedError

    # @staticmethod
    # # @abstractmethod
    # def new_nans(count: int) -> Sequence[_VectorBase]:
    #     """Create a list of one-offs with all NaN"""
    #     raise NotImplementedError

    @property
    def norm(self: Self) -> float:
        """Compute norm (from C++)"""
        return float(numpy.sqrt(self.np @ self.np))

    @property
    def normalized(self: Self) -> Self:
        """Make magnitude of a vector 1"""
        denom: Final = numpy.linalg.norm(self.np)
        if denom == 0:
            # raise ValueError(denom, type(denom), self.array, type(self.array), self._data, type(self._data))
            if (self._data == 0).all():  # all(x == 0 for x in self.array):
                return self
            raise ZeroDivisionError
        return self / denom
        # res: Final = self.from_np(self._data / denom)
        # if not TYPE_CHECKING:  # numba can't handle useless cast()
        #     return res
        # return cast("Self", res)

    @property
    def np(self: Self) -> npt.NDArray[numpy.float64]:
        """The numpy representation of this 'vector'"""
        return self._data.copy()  # numpy.array(self._data, dtype=numpy.float64)

    # def __array__(  # ruff: ignore[bad-dunder-method-name]
    #     self: Self,
    #     dtype: npt.DTypeLike | None = None,
    #     copy: bool | None = None,
    # ) -> npt.NDArray[numpy.float64]:
    #     # This allows numpy to treat us as "native" data without always calling vec.np
    #     return numpy.array(self._data, dtype=dtype, copy=copy)

    @staticmethod
    # @abstractmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> _VectorBase:
        """Create a vector from a numpy array"""
        raise NotImplementedError

    # @abstractmethod
    def __add__(self: Self, other: Self) -> Self:
        raise NotImplementedError

    # @abstractmethod
    def __sub__(self: Self, other: Self) -> Self:
        raise NotImplementedError

    @overload
    def __mul__(self: Self, other: float) -> Self: ...
    @overload
    def __mul__(self: Self, other: Self) -> float: ...

    def __mul__(self: Self, other: object) -> Any:
        this: Final[str] = f"vec{len(self)}"
        err_msg: Final[str] = f"Use np.dot({this}.np, {this}.np) or {this}.scale()"
        raise NotImplementedError(err_msg)

    def scale(self: Self, other: nbt.number_domain) -> None:
        self._data = other * self._data

    # XXX NOT ANY MORE Runtime dispatching below with overload_method

    #     """Dot product or scaling"""
    #     if isinstance(other, (int, float, numpy.integer, numpy.floating)):
    #         return cast("Self", self.from_np(other * self.np))
    #     if not isinstance(other, _VectorBase):
    #         return NotImplemented
    #     return numpy.dot(self._data, other._data)

    def __truediv__(self: Self, other: nbt.number_domain) -> Self:
        """Scaling"""
        # if isinstance(other, (int, float, numpy.integer, numpy.floating)):
        if other == 0:
            raise ZeroDivisionError
        res: Final = self.from_np(self._data / other)
        if not TYPE_CHECKING:  # numba can't handle useless cast()
            return res
        return cast("Self", res)


# @dataclass(frozen=True, slots=True)
@jitclass([("_data", float64[:])])
class vec2(_VectorBase):  # ruff: ignore[invalid-class-name]
    # @override
    # @property
    # def array(self: Self) -> list[float]:
    #     return [self.x, self.y]

    # @override
    # @staticmethod
    # def new_zero() -> vec2:
    # return vec2(0, 0)
    # new_zero = vec2_new_zero

    # @override
    # @staticmethod
    # def new_nan() -> vec2:
    #     return vec2(math.nan, math.nan)

    # @override
    # @staticmethod
    # def new_zeros(count: int) -> Sequence[vec2]:
    #     return [vec2.new_zero() for _ in range(count)]

    # @override
    # @staticmethod
    # def new_nans(count: int) -> Sequence[vec2]:
    #     return [vec2.new_nan() for _ in range(count)]

    @override
    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec2:
        # JIT cannot handle these assertions:
        # assert isinstance(array, numpy.ndarray)
        # assert array.shape == (2,), f"Invalid array shape {array.shape}!"
        # assert array.dtype == float, f"Invalid array type {array.dtype}!"
        # This will fail if no shape method or if cannot be coerced into float later...
        if array.shape != (2,):
            err_msg = f"Invalid array shape {array.shape}!"
            raise ValueError(err_msg)
        return vec2(float(array[0]), float(array[1]))

    def __add__(self: Self, other: _VectorBase) -> vec2:
        # if not isinstance(other, vec2):
        # return NotImplemented
        assert len(other) == 2
        return vec2(self.x + other.x, self.y + other.y)

    def __sub__(self: Self, other: _VectorBase) -> vec2:
        # if not isinstance(other, vec2):
        # return NotImplemented
        assert len(other) == 2
        return vec2(self.x - other.x, self.y - other.y)


# @dataclass(frozen=True, slots=True)
@jitclass([("_data", float64[:])])
class vec3(_VectorBase):  # ruff: ignore[invalid-class-name]
    # z: float

    def __init__(self: Self, x: float, y: float, z: float) -> None:
        # numba cannot: super().init(x=x, y=y)
        # self._data.append(z)
        self._data = numpy.array([x, y, z], dtype=numpy.float64)

    def cross(self: Self, other: vec3) -> vec3:
        """Cross-product of two vectors"""
        # if not isinstance(other, vec3):
        #     return NotImplemented
        res: Final = self.from_np(numpy.cross(self.np, other.np))
        if not TYPE_CHECKING:  # numba can't handle useless cast()
            return res
        return cast("Self", res)
        # if not TYPE_CHECKING:  # numba can't handle useless cast()
        #     return self.from_np(numpy.cross(self.np, other.np))
        # return cast("Self", self.from_np(numpy.cross(self.np, other.np)))

    # @override
    # @property
    # def array(self: Self) -> list[float]:
    #     return [self.x, self.y, self.z]

    # @override
    # @staticmethod
    # def new_zero() -> vec3:
    #     return vec3(0, 0, 0)

    # @override
    # @staticmethod
    # def new_nan() -> vec3:
    #     return vec3(math.nan, math.nan, math.nan)

    # @override
    # @staticmethod
    # def new_zeros(count: int) -> Sequence[vec3]:
    #     return [vec3.new_zero() for _ in range(count)]

    # @override
    # @staticmethod
    # def new_nans(count: int) -> Sequence[vec3]:
    #     return [vec3.new_nan() for _ in range(count)]

    @override
    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec3:
        # JIT cannot handle these assertions:
        # assert isinstance(array, numpy.ndarray)
        # assert array.shape == (3,), f"Invalid array shape {array.shape}!"
        # assert array.dtype in {float, numpy.float64}, f"Invalid array type {array.dtype}!"
        # This will fail if no shape method or if cannot be coerced into float later...
        if array.shape != (3,):
            err_msg = f"Invalid array shape {array.shape}!"
            raise ValueError(err_msg)
        return vec3(float(array[0]), float(array[1]), float(array[2]))

    @property
    def z(self: Self) -> float:
        return self._data[2]

    @property
    def xy(self: Self) -> vec2:
        return vec2(self.x, self.y)

    def __add__(self: Self, other: _VectorBase) -> vec3:
        # if not isinstance(other, vec3):
        #     return NotImplemented
        assert len(other) == 3
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self: Self, other: _VectorBase) -> vec3:
        # if not isinstance(other, vec3):
        #     return NotImplemented
        assert len(other) == 3
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)


# @dataclass(frozen=True, slots=True)
@jitclass([("_data", float64[:])])
class vec4(_VectorBase):  # vec3):  # ruff: ignore[invalid-class-name]
    # w: float

    def __init__(self: Self, x: float, y: float, z: float, w: float) -> None:
        # numba cannot: super().init(x=x, y=y)
        # self._data.extend([z, w])
        # self._data: list[float] = [x, y, z, w]
        self._data = numpy.array([x, y, z, w], dtype=numpy.float64)

    # @override
    # @property
    # def array(self: Self) -> list[float]:
    #     return [self.x, self.y, self.z, self.w]

    # @override
    # @staticmethod
    # def new_zero() -> vec4:
    #     return vec4(0, 0, 0, 0)

    # @override
    # @staticmethod
    # def new_nan() -> vec4:
    #     return vec4(math.nan, math.nan, math.nan, math.nan)

    # @override
    # @staticmethod
    # def new_zeros(count: int) -> Sequence[vec4]:
    #     return [vec4.new_zero() for _ in range(count)]

    # @override
    # @staticmethod
    # def new_nans(count: int) -> Sequence[vec4]:
    #     return [vec4.new_nan() for _ in range(count)]

    @override
    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec4:
        # JIT cannot handle these assertions:
        # assert isinstance(array, numpy.ndarray)
        # assert array.shape == (4,), f"Invalid array shape {array.shape}!"
        # assert array.dtype == float, f"Invalid array type {array.dtype}!"
        # This will fail if no shape method or if cannot be coerced into float later...
        if array.shape != (4,):
            err_msg = f"Invalid array shape {array.shape}!"
            raise ValueError(err_msg)
        return vec4(float(array[0]), float(array[1]), float(array[2]), float(array[3]))

    @staticmethod
    def from_vec3(v: vec3, *, w: float = 1) -> vec4:
        # assert isinstance(v, vec3)
        # assert isinstance(w, (int, float))
        return vec4(v.x, v.y, v.z, float(w))

    @property
    def z(self: Self) -> float:
        return self._data[2]

    @property
    def w(self: Self) -> float:
        return self._data[3]

    @property
    def xy(self: Self) -> vec2:
        return vec2(self.x, self.y)

    @property
    def xyz(self: Self) -> vec3:
        return vec3(self.x, self.y, self.z)

    def __add__(self: Self, other: _VectorBase) -> vec4:
        # if not isinstance(other, vec4):
        # return NotImplemented
        assert len(other) == 4
        return vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self: Self, other: _VectorBase) -> vec4:
        # if not isinstance(other, vec4):
        # return NotImplemented
        assert len(other) == 4
        return vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)


_XXX = '''

# @overload
# def overload_VB_mul(self: _VectorBase, other: float) -> _VectorBase: ...
# @overload
# def overload_VB_mul(self: _VectorBase, other: _VectorBase) -> float: ...


# @nbex.overload(operator.mul)
@nbex.overload_method(nbt.misc.ClassInstanceType, "__mul__")
def overload_VB_mul(self: _VectorBase, other: _VectorBase | nbt.number_domain) -> _VectorBase | float:
    """Dot product or scaling"""
    # See https://numba.discourse.group/t/how-to-overload-mul-dunder-method-in-jitclass/2144
    # if
    # if isinstance(other, _VectorBase):
    # if self in {
    #     _VectorBase.class_type.instance_type,
    #     vec2.class_type.instance_type,
    #     vec3.class_type.instance_type,
    #     vec4.class_type.instance_type,
    # }:
    from icecream import ic

    ic(self, other, type(self), type(other), nbt.number_domain, other in nbt.number_domain)
    ic(isinstance(self, nbt.ClassInstanceType))
    ic(self.class_type)
    ic(
        self.class_type
        in {
            # _VectorBase.class_type,
            vec2.class_type,
            vec3.class_type,
            vec4.class_type,
        }
    )
    # raise ValueError("WTF")

    if isinstance(self, nbt.ClassInstanceType) and self.class_type in {
        # _VectorBase.class_type,
        vec2.class_type,
        vec3.class_type,
        vec4.class_type,
    }:
        if isinstance(other, (int, float, numpy.integer, numpy.floating)) or other in nbt.number_domain:
            # def scalar_mul(self: _VectorBase, other: float) -> _VectorBase:
            def scalar_mul(self: _VectorBase, other: _VectorBase | nbt.number_domain) -> _VectorBase | float:
                return self.from_np(other * self._data)

            return scalar_mul  # cast("Self", self.from_np(other * self.np))

        if self is vec2.class_type.instance_type:
            assert other is vec2.class_type.instance_type
        if self is vec3.class_type.instance_type:
            assert other is vec3.class_type.instance_type
        if self is vec4.class_type.instance_type:
            assert other is vec4.class_type.instance_type

        from icecream import ic

        ic(
            self,
            other,
            # self.class_type.instance_type._data,
            # other._data,
            # numpy.dot(self.class_type.instance_type._data, other._data),
            # type(numpy.dot(self.class_type.instance_type._data, other._data)),
        )

        # def dot_mul(self: _VectorBase, other: _VectorBase) -> float:
        def dot_mul(self: _VectorBase, other: _VectorBase | nbt.number_domain) -> _VectorBase | float:
            return numpy.dot(self._data, other._data)

        return dot_mul
    return None


# @overload
# def overload_vec2_mul(self: vec2, other: float) -> _VectorBase: ...
# @overload
# def overload_vec2_mul(self: vec2, other: vec2) -> float: ...


# @numba.extending.overload(operator.mul)
# def overload_vec2_mul(self: vec2, other: object) -> Any:
#     """Dot product or scaling"""
#     if isinstance(other, (int, float, numpy.integer, numpy.floating)):

#         def scalar_mul(self: vec2, other: float) -> vec2:
#             return self.from_np(other * self.np)

#         return scalar_mul  # cast("Self", self.from_np(other * self.np))
#     if isinstance(other, vec2):

#         def dot_mul(self: vec2, other: vec2) -> float:
#             return numpy.dot(self._data, other._data)

#         return dot_mul
#     return None

'''


@njit
def vec2_new_nan() -> vec2:
    return vec2(math.nan, math.nan)


@njit
def vec2_new_nans(count: int) -> Sequence[vec2]:
    return [vec2_new_nan() for _ in range(count)]


@njit
def vec2_new_zero() -> vec2:
    return vec2(0, 0)


@njit
def vec2_new_zeros(count: int) -> Sequence[vec2]:
    return [vec2_new_zero() for _ in range(count)]


@njit
def vec3_new_nan() -> vec3:
    return vec3(math.nan, math.nan, math.nan)


@njit
def vec3_new_nans(count: int) -> Sequence[vec3]:
    return [vec3_new_nan() for _ in range(count)]


@njit
def vec3_new_zero() -> vec3:
    return vec3(0, 0, 0)


@njit
def vec3_new_zeros(count: int) -> Sequence[vec3]:
    return [vec3_new_zero() for _ in range(count)]


@njit
def vec4_new_nan() -> vec4:
    return vec4(math.nan, math.nan, math.nan, math.nan)


@njit
def vec4_new_nans(count: int) -> Sequence[vec4]:
    return [vec4_new_nan() for _ in range(count)]


@njit
def vec4_new_zero() -> vec4:
    return vec4(0, 0, 0, 0)


@njit
def vec4_new_zeros(count: int) -> Sequence[vec4]:
    return [vec4_new_zero() for _ in range(count)]


# @njit(cache=True)
# def _cacher() -> None:
#     _ = vec2(0, 0)
#     _ = vec3(0, 0, 0)
#     _ = vec4(0, 0, 0, 0)


# _cacher()


type Triangle = tuple[vec4, vec4, vec4]  # Used in Lesson 7 and beyond
