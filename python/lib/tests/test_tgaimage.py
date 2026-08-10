#!/bin/env python
from __future__ import annotations
import pytest
from typing import Self

from ..tgaimage import TGAColor, TGAImage, uint8_t
from hypothesis import assume, given, strategies as st
import sys

valid_uint8_t = st.integers(0, 255)


class TestTGAColor:
    def test_bad_index(self: Self) -> None:
        uut = TGAColor()
        with pytest.raises(IndexError):
            uut[-5]
        with pytest.raises(IndexError):
            uut[4]

    @given(st.integers())
    def test_bad_index_hypothesis(self: Self, v: int) -> None:
        assume(v < -4 or v > 3)
        assume(v < sys.maxsize)
        uut = TGAColor()
        with pytest.raises(IndexError):
            uut[v]

    @given(st.integers())
    def test_bad_index_write_hypothesis(self: Self, v: int) -> None:
        assume(v < -4 or v > 3)
        assume(v < sys.maxsize)
        uut = TGAColor()
        with pytest.raises(IndexError):
            uut[v] = uint8_t(42)

    @given(idx=st.integers(0, 3), b=valid_uint8_t, g=valid_uint8_t, r=valid_uint8_t, a=valid_uint8_t)
    def test_write(self: Self, idx: int, b: int, g: int, r: int, a: int) -> None:
        uut = TGAColor(b, g, r, a)
        assert uut[0] == b
        assert uut[1] == g
        assert uut[2] == r
        assert uut[3] == a
        uut[idx] = uint8_t(42)
        assert uut[idx] == 42

    @given(b=valid_uint8_t, g=valid_uint8_t, r=valid_uint8_t, a=valid_uint8_t)
    def test_bgra_hypothesis(self: Self, b: int, g: int, r: int, a: int) -> None:
        uut = TGAColor(b, g, r, a)
        assert uut.bgra.tolist() == [(b, g, r, a)]
        assert uut[0] == b
        assert uut[1] == g
        assert uut[2] == r
        assert uut[3] == a

    def test_bgra(self: Self) -> None:
        for b in range(20):
            for g in range(20):
                for r in range(20):
                    for a in range(20):
                        uut = TGAColor(b, g, r, a)
                        assert uut[0] == b
                        assert uut[1] == g
                        assert uut[2] == r
                        assert uut[3] == a

    def test_equal(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) == TGAColor(1, 2, 3, 4)

    def test_not_equal(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) != TGAColor(1, 2, 3, 3)


class TestTGAImage:
    def test_raw_empty(self: Self) -> None:
        uut = TGAImage(h=5, w=3)
        assert uut.npdata.tolist() == [[TGAColor()] * 3] * 5

    def test_raw_fill(self: Self) -> None:
        uut = TGAImage(h=5, w=3, c=TGAColor(1, 2, 3, 4))
        assert uut.npdata.tolist() == [[TGAColor(1, 2, 3, 4)] * 3] * 5
