#!/bin/env python
from __future__ import annotations

from math import isnan
from typing import Self, TypeAlias

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ..tgaimage import TGAColor
from ..trtypes import Matrix2f, Matrix3f, Matrix4f, MatrixLike, ZBuffer, _VectorBase, vec2, vec3, vec4

# positive_integers  = st.integers(min_value=0, max_value=2**31 - 1)
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
            assert isinstance(uut2.w, float)

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

    @given(in_data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_add(self: Self, in_data: list[float], vec_param: VecParam) -> None:
        width, class_ = vec_param
        uut1 = class_(*in_data[0:width])
        uut2 = class_(*in_data[width : 2 * width])

        res = uut1 + uut2
        assert res.x == pytest.approx(in_data[0] + in_data[width])
        assert res.y == pytest.approx(in_data[1] + in_data[width + 1])
        if width >= 3:
            assert isinstance(res, (vec3, vec4))
            assert res.z == pytest.approx(in_data[2] + in_data[width + 2])
        if width >= 4:
            assert isinstance(res, vec4)
            assert res.w == pytest.approx(in_data[3] + in_data[width + 3])

    @given(in_data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=8, max_size=8))
    def test_sub(self: Self, in_data: list[float], vec_param: VecParam) -> None:
        width, class_ = vec_param
        uut1 = class_(*in_data[0:width])
        uut2 = class_(*in_data[width : 2 * width])

        res = uut1 - uut2
        assert res.x == pytest.approx(in_data[0] - in_data[width])
        assert res.y == pytest.approx(in_data[1] - in_data[width + 1])
        if width >= 3:
            assert isinstance(res, (vec3, vec4))
            assert res.z == pytest.approx(in_data[2] - in_data[width + 2])
        if width >= 4:
            assert isinstance(res, vec4)
            assert res.w == pytest.approx(in_data[3] - in_data[width + 3])

    @given(in_data=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=8, max_size=8))
    def test_dot_product(self: Self, in_data: list[float], vec_param: VecParam) -> None:
        width, class_ = vec_param
        uut1 = class_(*in_data[0:width])
        uut2 = class_(*in_data[width : 2 * width])

        res = uut1 * uut2
        expected = uut1.x * uut2.x + uut1.y * uut2.y
        if width >= 3:
            assert isinstance(uut1, (vec3, vec4))
            assert isinstance(uut2, (vec3, vec4))
            expected += uut1.z * uut2.z
        if width >= 4:
            assert isinstance(uut1, vec4)
            assert isinstance(uut2, vec4)
            expected += uut1.w * uut2.w
        assert res == pytest.approx(expected)

    @pytest.mark.skip("Bad test")
    @given(in_min=reasonable_integers, in_max=reasonable_integers)
    def test_normalize(self: Self, in_min: int, in_max: int, vec_param: VecParam) -> None:
        assume(in_min < in_max)
        width, class_ = vec_param
        # Create values that are all min-value except the first one is max
        vals = [in_min] * width
        vals[0] = in_max
        uut = class_(*vals)
        res = uut.normalized
        assert res.x == pytest.approx(1)
        assert res.y == pytest.approx(0)
        if width >= 3:
            assert isinstance(res, (vec3, vec4))
            assert res.z == pytest.approx(0)
        if width >= 4:
            assert isinstance(res, vec4)
            assert res.w == pytest.approx(0)

    # vec * vec is tested elsewhere in test_dot_product

    # With vec4, 1 UUT would be 4 data values plus a 4x4 matrix of 16 = 20 values to choose from
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(in_data=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=20, max_size=20))
    def test_mult_cpp_vec_matrix(
        self: Self, in_data: list[float], vec_param: VecParam, request: pytest.FixtureRequest
    ) -> None:
        """Compare to C++ multiplication operations geometry.h"""
        this_test = request.node.callspec.id
        width, class_ = vec_param
        uut = class_(*in_data[0:width])

        _CPP = """
return (mat<1,nrows>{{lhs}}*rhs)[0]
            which calls matrix operator*:
            R1 = 1 (fixed)
            C1 = vector size
            C2 = vector size
template<int R1,int C1,int C2>mat<R1,C2> operator*(const mat<R1,C1>& lhs, const mat<C1,C2>& rhs) {
    mat<R1,C2> result;
    for (int i=R1; i--; )
        for (int j=C2; j--; )
            for (int k=C1; k--; result[i][j]+=lhs[i][k]*rhs[k][j]);
    return result;
"""
        rhs2: Matrix2f = np.array([in_data[0 * width : 1 * width], in_data[1 * width : 2 * width]])
        rhs3: Matrix3f = np.array([
            in_data[0 * width : 1 * width],
            in_data[1 * width : 2 * width],
            in_data[2 * width : 3 * width],
        ])
        rhs4: Matrix4f = np.array([
            in_data[0 * width : 1 * width],
            in_data[1 * width : 2 * width],
            in_data[2 * width : 3 * width],
            in_data[3 * width : 4 * width],
        ])
        rhs: MatrixLike

        match this_test:
            case "vec2":
                rhs = rhs2
            case "vec3":
                rhs = rhs3
            case "vec4":
                rhs = rhs4
            case _:
                err_msg = f"Invalid test id '{this_test}'"
                raise ValueError(err_msg)

        expected: list[float] = [0] * width
        for j in range(width):
            for k in range(width):
                expected[j] += uut.array[k] * rhs[k][j]

        res = uut.np @ rhs

        for w in range(width):
            assert expected[w] == pytest.approx(res[w])

    # With vec4, two 4x4 matrices of 16 = 32 values to choose from
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(in_data=st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=32, max_size=32))
    def test_mult_cpp_matrix_matrix(
        self: Self, in_data: list[float], vec_param: VecParam, request: pytest.FixtureRequest
    ) -> None:
        """Compare to C++ multiplication operations geometry.h"""
        this_test = request.node.callspec.id
        width, _ = vec_param

        _CPP = """
            R1 = C1 = C2 = vector size (should we test non-uniform???)
template<int R1,int C1,int C2>mat<R1,C2> operator*(const mat<R1,C1>& lhs, const mat<C1,C2>& rhs) {
    mat<R1,C2> result;
    for (int i=R1; i--; )
        for (int j=C2; j--; )
            for (int k=C1; k--; result[i][j]+=lhs[i][k]*rhs[k][j]);
    return result;
"""
        lhs2: Matrix2f = np.array([in_data[0 * width : 1 * width], in_data[1 * width : 2 * width]])
        lhs3: Matrix3f = np.array([
            in_data[0 * width : 1 * width],
            in_data[1 * width : 2 * width],
            in_data[2 * width : 3 * width],
        ])
        lhs4: Matrix4f = np.array([
            in_data[0 * width : 1 * width],
            in_data[1 * width : 2 * width],
            in_data[2 * width : 3 * width],
            in_data[3 * width : 4 * width],
        ])
        lhs: MatrixLike

        rhs2: Matrix2f = np.array([in_data[4 * width : 5 * width], in_data[5 * width : 6 * width]])
        rhs3: Matrix3f = np.array([
            in_data[4 * width : 5 * width],
            in_data[5 * width : 6 * width],
            in_data[6 * width : 7 * width],
        ])
        rhs4: Matrix4f = np.array([
            in_data[4 * width : 5 * width],
            in_data[5 * width : 6 * width],
            in_data[6 * width : 7 * width],
            in_data[7 * width : 8 * width],
        ])
        rhs: MatrixLike

        match this_test:
            case "vec2":
                lhs = lhs2
                rhs = rhs2
            case "vec3":
                lhs = lhs3
                rhs = rhs3
            case "vec4":
                lhs = lhs4
                rhs = rhs4
            case _:
                err_msg = f"Invalid test id '{this_test}'"
                raise ValueError(err_msg)

        expected = np.zeros(shape=(width, width))
        # Follow the C++ alg:
        for i in range(width):
            for j in range(width):
                for k in range(width):
                    expected[i][j] += lhs[i][k] * rhs[k][j]

        # Native numpy
        res = lhs @ rhs

        for r in range(width):
            for c in range(width):
                assert expected[r][c] == pytest.approx(res[r][c])


class TestZBuffer:
    def test_default_size(self: Self) -> None:
        with pytest.raises(TypeError):
            _ = ZBuffer()  # type: ignore[call-arg]

    def test_dimensions(self: Self) -> None:
        uut = ZBuffer(width=20, height=30)
        assert isnan(uut.vals[19][29])

    @pytest.mark.parametrize("nan_zero", [True, False], ids=["nan_zero=True", "nan_zero=False"])
    def test_to_tga_1x1(self: Self, nan_zero: bool) -> None:
        uut = ZBuffer(width=1, height=1)
        if not nan_zero:
            with pytest.raises(ValueError):
                fb = uut.to_tga(nan_zero=False)
        else:
            fb = uut.to_tga(nan_zero=True)
            assert fb.get(0, 0) == TGAColor(0)

    @pytest.mark.parametrize("nan_zero", [True, False], ids=["nan_zero=True", "nan_zero=False"])
    def test_to_tga_2x3(self: Self, nan_zero: bool) -> None:
        uut = ZBuffer(width=2, height=3)
        if not nan_zero:
            with pytest.raises(ValueError):
                fb = uut.to_tga(nan_zero=False)
        else:
            uut.vals[0][0] = 100
            uut.vals[1][2] = 100
            fb = uut.to_tga(nan_zero=True)
            # At this point, 0..100 should be 0..255
            for x in range(2):
                for y in range(3):
                    if (x, y) not in {(0, 0), (1, 2)}:
                        assert fb.get(x, y) == TGAColor(0)
                    else:
                        assert fb.get(x, y) == TGAColor(255)

    def test_to_tga_scaling(self: Self) -> None:
        uut = ZBuffer(width=20, height=1)
        for i in range(20):
            uut.vals[i][0] = -2000 + (i * 2000)  # -2000 to 36000 normalized should be about 13.42
        fb = uut.to_tga(nan_zero=True)
        for i in range(20):
            assert fb.get(i, 0) == TGAColor(round(i * 13.42))
