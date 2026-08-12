#!/bin/env python
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple, Self

import friendly
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

# from icecream import ic
from ..tgaimage import TGAColor, TGAImage, uint8_t

valid_uint8_t = st.integers(0, 255)
friendly.install()


class GoldenFile(NamedTuple):
    path: Path
    color_type: str
    RLE: bool
    colors: int
    height: int
    width: int

    def __str__(self: Self) -> str:
        return f"{self.path.name}: {self.width}x{self.height} {self.color_type} RLE={self.RLE}"


TEST_FILES: Final[tuple[GoldenFile, ...]] = (
    GoldenFile(
        Path("~/.steam/debian-installation/public/c1.tga").expanduser().resolve(),
        "RGB",
        RLE=False,
        colors=275,
        height=46,
        width=70,
    ),
    # For icon_close_hover, XnView MP says 2 colors, but really 27 because there are 26 blacks
    # with different alpha values:
    GoldenFile(
        Path("~/.steam/debian-installation/graphics/broadcast/icon_close_hover.tga").expanduser().resolve(),
        "RGBA",
        RLE=False,
        colors=27,
        height=17,
        width=17,
    ),
    GoldenFile(
        Path(Path(__file__).parent / "../.." / "../obj/floor_spec.tga").resolve(),
        "RGB",
        RLE=True,
        colors=1,
        height=1,
        width=1,
    ),
    GoldenFile(
        Path(Path(__file__).parent / "../.." / "../obj/boggie/body_spec.tga").resolve(),
        "RGB",
        RLE=True,
        colors=4035,
        height=2048,
        width=2048,
    ),
    GoldenFile(
        Path(Path(__file__).parent / "../.." / "../obj/african_head/african_head_eye_inner_spec.tga").resolve(),
        "RGBA",
        RLE=True,
        colors=5651,
        height=256,
        width=256,
    ),
    GoldenFile(
        Path(Path(__file__).parent / "../.." / "../obj/african_head/african_head_spec.tga").resolve(),
        "Mono",
        RLE=True,
        colors=166,
        height=1024,
        width=1024,
    ),
)


class TestTGAColor:
    def test_bad_index(self: Self, subtests: pytest.Subtests) -> None:
        with pytest.raises(AssertionError):
            uut = TGAColor(bpp=uint8_t(0))
        with pytest.raises(AssertionError):
            uut = TGAColor(bpp=uint8_t(5))
        for bpp in range(1, 5):
            with subtests.test(bpp=bpp):
                uut = TGAColor(bpp=uint8_t(bpp))
                with pytest.raises(IndexError):
                    uut[-(bpp + 1)]
                with pytest.raises(IndexError):
                    uut[bpp]

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
        assert uut.data == [b.to_bytes(1), g.to_bytes(1), r.to_bytes(1), a.to_bytes(1)]
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


@st.composite
def limited_xy(draw: Callable, min_: int = 0) -> tuple[int, int]:
    """
    Gives a tuple where the max (first value) is random and the second will be less

    This is used to randomize tests where we want an (x, y) and know it is an index within (w, h)
    """
    # The TGA spec is 2^16-1, but that ends up being really slow and doesn't really "prove" much
    MAX_VAL: Final[int] = pow(2, 10) - 1
    max_ = draw(st.integers(min_value=min_, max_value=MAX_VAL))
    sub = draw(st.integers(min_value=min_, max_value=max_ - 1)) if max_ else 0
    return (max_, sub)


GRADIENT: Final[list[list[TGAColor]]] = [
    [
        TGAColor(0, 0, 0, 0),
        TGAColor(10, 10, 10, 10),
        TGAColor(20, 20, 20, 20),
        TGAColor(30, 30, 30, 30),
    ],
    [
        TGAColor(1, 1, 1, 1),
        TGAColor(11, 11, 11, 11),
        TGAColor(21, 21, 21, 21),
        TGAColor(31, 31, 31, 31),
    ],
    [
        TGAColor(2, 2, 2, 2),
        TGAColor(12, 12, 12, 12),
        TGAColor(22, 22, 22, 22),
        TGAColor(32, 32, 32, 32),
    ],
    [
        TGAColor(3, 3, 3, 3),
        TGAColor(13, 13, 13, 13),
        TGAColor(23, 23, 23, 23),
        TGAColor(33, 33, 33, 33),
    ],
]


class TestTGAImage:
    def test_raw_empty(self: Self) -> None:
        uut = TGAImage(h=5, w=3)
        assert uut.npdata.tolist() == [[TGAColor()] * 3] * 5

    def test_raw_fill(self: Self) -> None:
        uut = TGAImage(h=5, w=3, c=TGAColor(1, 2, 3, 4))
        assert uut.npdata.tolist() == [[TGAColor(1, 2, 3, 4)] * 3] * 5

    @given(h_y=limited_xy(), w_x=limited_xy())
    def test_set_get(self: Self, h_y: tuple[int, int], w_x: tuple[int, int]) -> None:
        (h, y) = h_y
        (w, x) = w_x
        # Minimum 1x1 image:
        h = h if h else 1
        w = w if w else 1
        uut = TGAImage(w=w, h=h)
        color = TGAColor(1, 2, 3, 4)
        assert uut.get(0, 0) == TGAColor()
        uut.set(x, y, color)
        assert uut.get(x, y) != TGAColor()
        assert uut.get(x, y) == color

    def gradient_fill(self: Self) -> TGAImage:
        uut = TGAImage(w=4, h=4)
        for row in range(4):
            for col in range(4):
                v = 10 * row + col
                c = TGAColor(v, v, v, v)
                uut.set(row, col, c)
        return uut

    def test_gradient_fill(self: Self) -> None:
        uut = self.gradient_fill()
        assert uut.npdata.tolist() == np.array(GRADIENT).tolist()
        assert np.array_equal(uut.npdata, np.array(GRADIENT))

    def test_double_flips(self: Self) -> None:
        uut = self.gradient_fill()
        uut.flip_horizontally()
        uut.flip_horizontally()
        assert np.array_equal(uut.npdata, np.array(GRADIENT))
        uut.flip_vertically()
        uut.flip_vertically()
        assert np.array_equal(uut.npdata, np.array(GRADIENT))
        uut.flip_vertically()
        uut.flip_horizontally()
        uut.flip_vertically()
        uut.flip_horizontally()
        assert np.array_equal(uut.npdata, np.array(GRADIENT))

    def test_horizontal_flip(self: Self) -> None:
        uut = self.gradient_fill()
        uut.flip_horizontally()
        golden = np.full(shape=(4, 4), fill_value=TGAColor())
        for row in range(4):
            for col in range(4):
                v = 30 - (10 * row) + col
                c = TGAColor(v, v, v, v)
                golden[col, row] = c
        assert np.array_equal(uut.npdata, golden)

    def test_vertical_flip(self: Self) -> None:
        uut = self.gradient_fill()
        uut.flip_vertically()
        golden = np.full(shape=(4, 4), fill_value=TGAColor())
        for row in range(4):
            for col in range(4):
                v = 10 * row + (3 - col)
                c = TGAColor(v, v, v, v)
                golden[col, row] = c
        assert np.array_equal(uut.npdata, golden)

    def test_bad_file(self: Self) -> None:
        uut = TGAImage()
        with pytest.raises(OSError):
            uut.read_tga_file("/abc.tga")

    @staticmethod
    def skip_missing(tga_file: Path) -> None:
        if not tga_file.is_file():
            pytest.skip(f"The required file ({tga_file}) was not found - have you installed Steam?")

    @staticmethod
    def count_unique_colors(uut: TGAImage) -> int:
        uniques = set(uut.npdata.flat)
        count = len(uniques)
        return count

    @staticmethod
    def check_file(test_file: GoldenFile) -> None:
        TestTGAImage.skip_missing(test_file.path)
        uut = TGAImage.read_tga_file(test_file.path)
        assert uut.width == test_file.width, f"Expected width={test_file.width}, got {uut.width}"
        assert uut.height == test_file.height, f"Expected height={test_file.height}, got {uut.height}"
        assert (count := TestTGAImage.count_unique_colors(uut)) == test_file.colors, (
            f"Expected {test_file.colors} colors, got {count}"
        )

    # @pytest.mark.skip()
    def test_good_file_matrix(self: Self, subtests: pytest.Subtests) -> None:
        for f in TEST_FILES:
            with subtests.test(msg=str(f)):
                self.check_file(f)

    def test_rle_unrle(self: Self, subtests: pytest.Subtests) -> None:
        # Sometimes the raw RLE didn't match, but the re-expanded matches...
        for test_file in (f for f in TEST_FILES if f.RLE):
            with subtests.test(msg=str(test_file)):
                TestTGAImage.skip_missing(test_file.path)
                # Un-RLE the data...
                uut = TGAImage.read_tga_file(test_file.path)
                # Get the original data back out
                golden_data = uut._raw_payload
                # Re-RLE the data...
                re_compressed = uut.unload_rle_data()
                # Expand that
                test_data = uut.load_rle_data(re_compressed)
                assert test_data == golden_data
