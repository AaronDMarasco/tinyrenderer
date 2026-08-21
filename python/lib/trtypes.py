from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class vec3:
    x: float
    y: float
    z: float
