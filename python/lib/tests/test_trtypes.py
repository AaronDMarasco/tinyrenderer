#!/bin/env python
from __future__ import annotations

from typing import Self, TypeAlias

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ..trtypes import _VectorBase, vec2, vec3, vec4

reasonable_integers = st.integers(min_value=-(2**31), max_value=2**31 - 1)

VecParam: TypeAlias = tuple[int, type[_VectorBase]]


@pytest.mark.parametrize("vec_param", [(2, vec2), (3, vec3), (4, vec4)], ids=["vec2", "vec3", "vec4"])
class TestVector:
    @given(vec_in=st.lists(reasonable_integers, min_size=4, max_size=4))
    def test_to_from_numpy(self: Self, vec_in: list[int], vec_param: VecParam) -> None:
        width, class_ = vec_param
        vec_in = vec_in[:width]

        uut = class_(*vec_in)
        uut2 = uut.from_np(uut.np)
        assert isinstance(uut.np, np.ndarray)
        assert uut.np.shape == (width,)
        assert uut.np.dtype == float
        assert uut == uut
        assert isinstance(uut2.x, float)  # Not an np type
        assert isinstance(uut2.y, float)
        if width >= 3:
            assert isinstance(uut2, (vec3, vec4))
            assert isinstance(uut2.z, float)
        if width >= 4:
            assert isinstance(uut2, vec4)
            assert isinstance(uut2.a, float)

        assert uut2.y == vec_in[1]

    def test_from_numpy_bad_type(self: Self, vec_param: VecParam) -> None:
        width, class_ = vec_param

        with pytest.raises(AssertionError):
            class_.from_np([0] * width)  # type: ignore[arg-type]

    @given(vec_in=st.lists(reasonable_integers, min_size=4, max_size=4))
    def test_from_numpy_wrong_size(self: Self, vec_in: list[int], vec_param: VecParam) -> None:
        width, class_ = vec_param
        vec_in = vec_in[:width]

        with pytest.raises(AssertionError):
            class_.from_np(np.array([]))
        with pytest.raises(AssertionError):
            class_.from_np(np.array([vec_in, vec_in], dtype=float))

    @given(vec_in=st.lists(reasonable_integers, min_size=4, max_size=4))
    def test_from_numpy_wrong_type(self: Self, vec_in: list[int], vec_param: VecParam) -> None:
        width, class_ = vec_param

        with pytest.raises(AssertionError):
            class_.from_np(np.array(vec_in[:width], dtype=int))

    @given(vec_in=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_add(self: Self, vec_in: list[float], vec_param: VecParam) -> None:
        width, class_ = vec_param
        # vec_in = vec_in[: (width * 2)]

        uut1 = class_(*vec_in[0:width])
        uut2 = class_(*vec_in[width : 2 * width])
        res = uut1 + uut2
        assert res.x == pytest.approx(vec_in[0] + vec_in[width])
        assert res.y == pytest.approx(vec_in[1] + vec_in[width + 1])
        if width >= 3:
            assert isinstance(res, (vec3, vec4))
            assert res.z == pytest.approx(vec_in[2] + vec_in[width + 2])
        if width >= 4:
            assert isinstance(res, vec4)
            assert res.a == pytest.approx(vec_in[3] + vec_in[width + 3])

    @given(vec_in=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_sub(self: Self, vec_in: list[float], vec_param: VecParam) -> None:
        width, class_ = vec_param

        uut1 = class_(*vec_in[0:width])
        uut2 = class_(*vec_in[width : 2 * width])
        res = uut1 - uut2
        assert res.x == pytest.approx(vec_in[0] - vec_in[width])
        assert res.y == pytest.approx(vec_in[1] - vec_in[width + 1])
        if width >= 3:
            assert isinstance(res, (vec3, vec4))
            assert res.z == pytest.approx(vec_in[2] - vec_in[width + 2])
        if width >= 4:
            assert isinstance(res, vec4)
            assert res.a == pytest.approx(vec_in[3] - vec_in[width + 3])

    @given(vec_in=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=8, max_size=8))
    def test_dot_product(self: Self, vec_in: list[float], vec_param: VecParam) -> None:
        width, class_ = vec_param

        uut1 = class_(*vec_in[0:width])
        uut2 = class_(*vec_in[width : 2 * width])
        res = uut1 * uut2
        expected = uut1.x * uut2.x + uut1.y * uut2.y
        if width >= 3:
            assert isinstance(uut1, (vec3, vec4))
            assert isinstance(uut2, (vec3, vec4))
            expected += uut1.z * uut2.z
        if width >= 4:
            assert isinstance(uut1, vec4)
            assert isinstance(uut2, vec4)
            expected += uut1.a * uut2.a
        assert res == pytest.approx(expected)
