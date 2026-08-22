from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self  # , TypeVar

import numpy  # Import as np conflicts with property named np
import numpy.typing as npt

# VB = TypeVar("VB", bound="_VectorBase")


@dataclass(frozen=True, slots=True)
class _VectorBase(ABC):
    x: float
    y: float

    @property
    @abstractmethod
    def array(self: Self) -> list[float]:
        raise NotImplementedError

    @property
    def np(self: Self) -> npt.NDArray[numpy.float64]:
        return numpy.array(self.array, dtype=float)

    @staticmethod
    @abstractmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> _VectorBase: ...

    @abstractmethod
    def __add__(self: Self, other: _VectorBase) -> _VectorBase: ...

    @abstractmethod
    def __sub__(self: Self, other: _VectorBase) -> _VectorBase: ...

    def __mul__(self: Self, other: _VectorBase) -> float:
        """Dot product"""
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
        assert array.shape == (2,)
        assert array.dtype == float
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
        assert array.shape == (3,)
        assert array.dtype == float
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
    a: float

    @property
    def array(self: Self) -> list[float]:
        return [self.x, self.y, self.z, self.a]

    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec4:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (4,)
        assert array.dtype == float
        return vec4(x=float(array[0]), y=float(array[1]), z=float(array[2]), a=float(array[3]))

    def __add__(self: Self, other: _VectorBase) -> vec4:
        if not isinstance(other, vec4):
            return NotImplemented
        return vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.a + other.a)

    def __sub__(self: Self, other: _VectorBase) -> vec4:
        if not isinstance(other, vec4):
            return NotImplemented
        return vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.a - other.a)
