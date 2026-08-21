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
from hypothesis import assume, given
from hypothesis import strategies as st

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
        assert isinstance(uut2.y, float)
        assert isinstance(uut2.z, float)
        assert uut2.y == 2

    def test_from_numpy_bad_type(self: Self) -> None:
        with pytest.raises(AssertionError):
            vec3.from_np([1, 2, 3])  # type: ignore[arg-type]

    def test_from_numpy_wrong_size(self: Self) -> None:
        with pytest.raises(AssertionError):
            vec3.from_np(np.array([]))
        with pytest.raises(AssertionError):
            vec3.from_np(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))

    def test_from_numpy_wrong_type(self: Self) -> None:
        with pytest.raises(AssertionError):
            vec3.from_np(np.array([1, 2, 3], dtype=int))

    @given(vec_in=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=6, max_size=6))
    def test_add(self: Self, vec_in: list[float]) -> None:
        uut1 = vec3(*vec_in[0:3])
        uut2 = vec3(*vec_in[3:6])
        res = uut1 + uut2
        assert res.x == pytest.approx(vec_in[0] + vec_in[3])
        assert res.y == pytest.approx(vec_in[1] + vec_in[4])
        assert res.z == pytest.approx(vec_in[2] + vec_in[5])

    @given(vec_in=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=6, max_size=6))
    def test_sub(self: Self, vec_in: list[float]) -> None:
        uut1 = vec3(*vec_in[0:3])
        uut2 = vec3(*vec_in[3:6])
        res = uut1 - uut2
        assert res.x == pytest.approx(vec_in[0] - vec_in[3])
        assert res.y == pytest.approx(vec_in[1] - vec_in[4])
        assert res.z == pytest.approx(vec_in[2] - vec_in[5])

    @given(vec_in=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=6, max_size=6))
    def test_dot_product(self: Self, vec_in: list[float]) -> None:
        uut1 = vec3(*vec_in[0:3])
        uut2 = vec3(*vec_in[3:6])
        res = uut1 * uut2
        assert res == pytest.approx(np.dot(uut1.np, uut2.np))
