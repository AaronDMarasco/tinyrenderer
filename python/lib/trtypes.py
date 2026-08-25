from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Self, cast, overload

import numpy  # Import as np conflicts with property named np
import numpy.typing as npt

from .tgaimage import TGAColor, TGAImage

type Matrix3f = numpy.ndarray[tuple[Literal[3], Literal[3]], numpy.dtype[numpy.float64]]
type Matrix4f = numpy.ndarray[tuple[Literal[4], Literal[4]], numpy.dtype[numpy.float64]]


@dataclass(slots=True)
class ZBuffer:
    vals: list[list[float]] = field(init=False)

    def __init__(self: Self, *, width: int, height: int) -> None:
        self.vals = cast(list[list[float]], numpy.full((width, height), numpy.nan, dtype=float).tolist())

    def to_tga(self: Self, *, nan_zero: bool = False) -> TGAImage:
        """If nan_zero is not set, any unset values will explode"""
        nparray = numpy.array(self.vals, dtype=float)
        if not nan_zero and numpy.isnan(nparray).any():
            err_msg = "ZBuffer had NaN and not told to assume zero!"
            raise ValueError(err_msg)
        nparray[numpy.isnan(nparray)] = 0
        width, height = nparray.shape
        # Want to scale from 0..255
        min_val = nparray.min()
        max_val = nparray.max()
        if min_val == max_val:
            # Just regenerate an empty image
            return TGAImage(w=width, h=height, bpp=1, c=TGAColor(0))
        # Otherwise, create scaled image
        fb = TGAImage(w=width, h=height, bpp=1)
        normalized = (nparray - min_val) * 255 / (max_val - min_val)
        for x in range(0, width):
            for y in range(0, height):
                fb.set(x, y, TGAColor(round(normalized[x, y])))
        return fb


@dataclass(frozen=True, slots=True)
class _VectorBase(ABC):
    x: float
    y: float

    @property
    @abstractmethod
    def array(self: Self) -> list[float]: ...

    @property
    def normalized(self: Self) -> Self:
        # Not sure which algorithm is expected... if we want the linear algebra one,
        # then testing is unknown... so I'll leave that here...
        # return cast(Self, self.from_np(self.np / numpy.linalg.norm(self.np)))
        offset = self.np - numpy.min(self.np)
        if offset.max() == 0:  # Avoid potential divide-by-zero
            return cast(Self, self.from_np(offset))
        return cast(Self, self.from_np(offset / numpy.max(offset)))

    @property
    def np(self: Self) -> npt.NDArray[numpy.float64]:
        return numpy.array(self.array, dtype=float)

    @staticmethod
    @abstractmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> _VectorBase: ...

    def cross(self: Self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return cast(Self, self.from_np(numpy.cross(self.np, other.np)))

    @abstractmethod
    def __add__(self: Self, other: Self) -> Self: ...

    @abstractmethod
    def __sub__(self: Self, other: Self) -> Self: ...

    @overload
    def __mul__(self: Self, other: int | float) -> Self: ...
    @overload
    def __mul__(self: Self, other: Self) -> float: ...

    def __mul__(self: Self, other: Any) -> Any:
        """Dot product or scaling"""
        if isinstance(other, (int, float, numpy.integer, numpy.floating)):
            return cast(Self, self.from_np(other * self.np))
        if not isinstance(other, _VectorBase):
            return NotImplemented
        return numpy.dot(self.array, other.array)


@dataclass(frozen=True, slots=True)
class vec2(_VectorBase):
    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y]

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
class vec3(_VectorBase):
    z: float

    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y, self.z]

    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec3:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (3,), f"Invalid array shape {array.shape}!"
        assert array.dtype == float, f"Invalid array type {array.dtype}!"
        return vec3(x=float(array[0]), y=float(array[1]), z=float(array[2]))

    def __add__(self: Self, other: _VectorBase) -> vec3:
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self: Self, other: _VectorBase) -> vec3:
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)


@dataclass(frozen=True, slots=True)
class vec4(_VectorBase):
    z: float
    w: float

    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y, self.z, self.w]

    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec4:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (4,), f"Invalid array shape {array.shape}!"
        assert array.dtype == float, f"Invalid array type {array.dtype}!"
        return vec4(x=float(array[0]), y=float(array[1]), z=float(array[2]), w=float(array[3]))

    @staticmethod
    def from_vec3(v: vec3, *, w: float) -> vec4:
        assert isinstance(v, vec3)
        assert isinstance(w, (int, float))
        return vec4(x=v.x, y=v.y, z=v.z, w=float(w))

    @property
    def xy(self: Self) -> vec2:
        return vec2(x=self.x, y=self.y)

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


def norm(v: _VectorBase) -> float:
    return float(numpy.sqrt(v * v))
