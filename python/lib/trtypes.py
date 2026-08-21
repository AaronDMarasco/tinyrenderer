from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy  # Import as np conflicts with property named np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class vec3:
    x: float
    y: float
    z: float

    @property
    def np(self: Self) -> npt.NDArray[numpy.float64]:
        return numpy.array([self.x, self.y, self.z], dtype=float)

    @staticmethod
    def from_np(array: npt.NDArray[numpy.float64]) -> vec3:
        assert isinstance(array, numpy.ndarray)
        assert array.shape == (3,)
        assert array.dtype == float
        return vec3(x=float(array[0]), y=float(array[1]), z=float(array[2]))

    def __add__(self: Self, other: vec3) -> vec3:
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self: Self, other: vec3) -> vec3:
        if not isinstance(other, vec3):
            return NotImplemented
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self: Self, other: vec3) -> float:
        """Dot product"""
        if not isinstance(other, vec3):
            return NotImplemented
        return self.x * other.x + self.y * other.y + self.z * other.z
