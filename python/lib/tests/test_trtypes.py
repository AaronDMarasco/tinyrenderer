#!/bin/env python
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Final, NamedTuple, Self

import numpy as np

# import numpy.typing as npt
import pytest
# from hypothesis import assume, given
# from hypothesis import strategies as st

from ..trtypes import vec3


class TestVec3:
    def test_to_from_numpy(self: Self) -> None:
        uut = vec3(1, 2, 3)
        from icecream import ic

        ic(uut.np)
        uut2 = uut.from_np(uut.np)
        assert isinstance(uut.np, np.ndarray)
        assert uut.np.shape == (3,)
        assert uut.np.dtype == float
        assert uut == uut
        assert isinstance(uut2.x, float)  # Not an np type
        assert isinstance(uut2.y, float)  # Not an np type
        assert isinstance(uut2.z, float)  # Not an np type
        assert uut2.y == 2
