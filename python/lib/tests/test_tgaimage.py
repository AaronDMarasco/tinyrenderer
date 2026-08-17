#!/bin/env python
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Final, NamedTuple, Self

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ..tgaimage import TGAColor, TGAColor_t, TGAImage, uint8_t

valid_uint8_t = st.integers(0, 255)


class GoldenFile(NamedTuple):
    """Helper to describe a golden file for testing fixture"""

    path: Path  # These are written weird to allow paste relative from python subdir
    color_type: str
    RLE: bool
    colors: int
    height: int
    width: int

    def __str__(self: Self) -> str:
        return f"{self.path.name}: {self.width}x{self.height} {self.color_type} ({'' if self.RLE else 'no '}RLE)"


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


@pytest.fixture(params=TEST_FILES, ids=[str(g) for g in TEST_FILES])
def file_suite(request: pytest.FixtureRequest) -> Generator[GoldenFile]:
    """Converts TEST_FILES into pytest-native input iterators"""
    yield request.param


class TestTGAColor:
    def test_auto_bpp(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4).bytespp == 4
        assert TGAColor(1, 2, 3).bytespp == 3
        assert TGAColor(1, 2).bytespp == 2
        assert TGAColor(1).bytespp == 1

    def test_bad_index(self: Self, subtests: pytest.Subtests) -> None:
        with pytest.raises(ValueError):
            uut = TGAColor(bpp=0)
        with pytest.raises(ValueError):
            uut = TGAColor(bpp=5)
        for bpp in range(1, 5):
            with subtests.test(bpp=bpp):
                vals = [0] * bpp
                uut = TGAColor(*vals, bpp=bpp)
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

    def test_bad_kwargs(self: Self) -> None:
        with pytest.raises(ValueError):
            TGAColor(g=1)
        with pytest.raises(ValueError):
            TGAColor(r=1)
        with pytest.raises(ValueError):
            TGAColor(a=1)
        with pytest.raises(ValueError):
            TGAColor(b=None, g=1, r=1, a=1)
        with pytest.raises(ValueError):
            TGAColor(b=1, g=None, r=1, a=1)
        with pytest.raises(ValueError):
            TGAColor(b=1, g=1, r=None, a=1)

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

    @given(b=valid_uint8_t, g=valid_uint8_t, r=valid_uint8_t, a=valid_uint8_t)
    def test_bgra_hypothesis(self: Self, b: int, g: int, r: int, a: int) -> None:
        uut = TGAColor(b, g, r, a)
        assert uut[0] == b
        assert uut[1] == g
        assert uut[2] == r
        assert uut[3] == a

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    @given(b=valid_uint8_t, g=valid_uint8_t, r=valid_uint8_t, a=valid_uint8_t)
    def test_bgra_hypothesis_bpp(self: Self, b: int, g: int, r: int, a: int, bpp: int) -> None:
        uut = TGAColor(b, g, r, a, bpp=bpp)
        assert uut[0] == b
        if bpp >= 2:
            assert uut[1] == g
        else:
            with pytest.raises(IndexError):
                uut[1]
        if bpp >= 3:
            assert uut[2] == r
        else:
            with pytest.raises(IndexError):
                uut[2]
        if bpp == 4:
            assert uut[3] == a
        else:
            with pytest.raises(IndexError):
                uut[3]

    def test_caching(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) is TGAColor(1, 2, 3, 4)
        assert TGAColor(1, 2, 3) is TGAColor(1, 2, 3)
        assert TGAColor(1, 2) is TGAColor(1, 2)
        assert TGAColor(1) is TGAColor(1)
        assert TGAColor() is TGAColor()
        assert all(map(lambda c: c is TGAColor(4, 3, 2, 1), (TGAColor(4, 3, 2, 1) for _ in range(1_000_000))))

    def test_caching_auto_bpp(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) is TGAColor(1, 2, 3, 4, bpp=4)
        assert TGAColor(1, 2, 3) is TGAColor(1, 2, 3, bpp=3)
        assert TGAColor(1, 2) is TGAColor(1, 2, bpp=2)
        assert TGAColor(1) is TGAColor(1, bpp=1)
        assert TGAColor() is TGAColor(0, bpp=1)

    def test_equal(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) == TGAColor(1, 2, 3, 4)

    def test_not_equal(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) != TGAColor(1, 2, 3, 3)

    def test_frozen(self: Self) -> None:
        uut = TGAColor(1, 2, 3, 4)
        assert uut[1] == 2
        with pytest.raises(AttributeError):
            uut[1] = uint8_t(2)

    def test_string(self: Self, subtests: pytest.Subtests) -> None:
        with subtests.test("Full Constructor"):
            assert repr(TGAColor(1, 2, 3, 4, bpp=4)) == "TGAColor_t(b=1, g=2, r=3, a=4, bpp=4)"
            assert repr(TGAColor(1, 2, 3, bpp=3)) == "TGAColor_t(b=1, g=2, r=3, bpp=3)"
            assert repr(TGAColor(1, 2, bpp=2)) == "TGAColor_t(b=1, g=2, bpp=2)"
            assert repr(TGAColor(1, bpp=1)) == "TGAColor_t(b=1, bpp=1)"
        with subtests.test("Automatic BPP"):
            assert repr(TGAColor(1, 2, 3, 4)) == "TGAColor_t(b=1, g=2, r=3, a=4, bpp=4)"
            assert repr(TGAColor(1, 2, 3)) == "TGAColor_t(b=1, g=2, r=3, bpp=3)"
            assert repr(TGAColor(1, 2)) == "TGAColor_t(b=1, g=2, bpp=2)"
            assert repr(TGAColor(1)) == "TGAColor_t(b=1, bpp=1)"


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


class TestTGAImage:
    GRADIENT: list[list[TGAColor_t]] | None = None
    GRADIENT_SIZE: Final = 4  # Max 4 or exceeds uint8_t

    # Helpers:
    @staticmethod
    def check_image(uut: TGAImage, metadata: GoldenFile) -> None:
        # TODO: Sample a random set of pixels or something?
        assert uut.width == metadata.width, f"Expected width={metadata.width}, got {uut.width}"
        assert uut.height == metadata.height, f"Expected height={metadata.height}, got {uut.height}"
        assert (count := TestTGAImage.count_unique_colors(uut)) == metadata.colors, (  # noqa: RUF018
            f"Expected {metadata.colors} colors, got {count}"
        )

    @staticmethod
    def count_unique_colors(uut: TGAImage) -> int:
        uniques = set(uut.npdata.flat)
        count = len(uniques)
        return count

    @staticmethod
    def skip_missing(tga_file: Path) -> None:
        if not tga_file.is_file():
            pytest.skip(f"The required file ({tga_file}) was not found - have you installed Steam?")

    @classmethod
    def gradient_fill(cls) -> TGAImage:
        uut = TGAImage(w=cls.GRADIENT_SIZE, h=cls.GRADIENT_SIZE)
        first_run = cls.GRADIENT is None
        if first_run:
            cls.GRADIENT = [[] for _ in range(cls.GRADIENT_SIZE)]
        for row in range(cls.GRADIENT_SIZE):
            for col in range(cls.GRADIENT_SIZE):
                v = 10 * row + col
                c = TGAColor(v, v, v, v)
                uut.set(row, col, c)
                if first_run:
                    assert cls.GRADIENT is not None, "Useless but keeping mypy happy"
                    cls.GRADIENT[col].append(c)
        return uut

    # Test methods:
    def test_bad_file(self: Self) -> None:
        uut = TGAImage()
        with pytest.raises(OSError):
            uut.read_tga_file("/abc.tga")

    def test_double_flips(self: Self) -> None:
        uut = self.gradient_fill()
        uut.flip_horizontally()
        uut.flip_horizontally()
        assert np.array_equal(uut.npdata, np.array(self.GRADIENT))
        uut.flip_vertically()
        uut.flip_vertically()
        assert np.array_equal(uut.npdata, np.array(self.GRADIENT))
        uut.flip_vertically()
        uut.flip_horizontally()
        uut.flip_vertically()
        uut.flip_horizontally()
        assert np.array_equal(uut.npdata, np.array(self.GRADIENT))

    @pytest.mark.skipif(bool(os.getenv("QUICK_CHECK")), reason="QUICK_CHECK")
    def test_good_files(self: Self, file_suite: GoldenFile) -> None:
        TestTGAImage.skip_missing(file_suite.path)
        uut = TGAImage.read_tga_file(file_suite.path)
        self.check_image(uut, file_suite)

    def test_gradient_fill(self: Self) -> None:
        uut = self.gradient_fill()
        assert uut.npdata.tolist() == np.array(self.GRADIENT).tolist()
        assert np.array_equal(uut.npdata, np.array(self.GRADIENT))

    def test_horizontal_flip(self: Self) -> None:
        uut = self.gradient_fill()
        uut.flip_horizontally()
        golden = np.full(shape=(self.GRADIENT_SIZE, self.GRADIENT_SIZE), fill_value=TGAColor())
        for row in range(self.GRADIENT_SIZE):
            for col in range(self.GRADIENT_SIZE):
                v = ((self.GRADIENT_SIZE - 1) * 10) - (10 * row) + col
                c = TGAColor(v, v, v, v)
                golden[col, row] = c
        assert np.array_equal(uut.npdata, golden)

    def test_raw_empty(self: Self) -> None:
        uut = TGAImage(h=5, w=3)
        assert uut.npdata.tolist() == [[uut.fill_value] * 3] * 5

    def test_raw_fill(self: Self) -> None:
        uut = TGAImage(h=5, w=3, c=TGAColor(1, 2, 3, 4))
        assert uut.npdata.tolist() == [[TGAColor(1, 2, 3, 4)] * 3] * 5

    @pytest.mark.skipif(bool(os.getenv("QUICK_CHECK")), reason="QUICK_CHECK")
    def test_rle_unrle(self: Self, subtests: pytest.Subtests, file_suite: GoldenFile) -> None:
        # Sometimes the raw RLE didn't match, but the re-expanded matches...
        TestTGAImage.skip_missing(file_suite.path)
        # Un-RLE the data...
        uut = TGAImage.read_tga_file(file_suite.path)
        # Get the original data back out
        golden_data = uut._raw_payload
        # Re-RLE the data...
        re_compressed = uut.unload_rle_data()
        # Expand that
        test_data = uut.load_rle_data(re_compressed)
        assert test_data == golden_data

    @given(h_y=limited_xy(), w_x=limited_xy())
    def test_set_get(self: Self, h_y: tuple[int, int], w_x: tuple[int, int]) -> None:
        (h, y) = h_y
        (w, x) = w_x
        # Minimum 1x1 image:
        h = h if h else 1
        w = w if w else 1
        uut = TGAImage(w=w, h=h)
        color = TGAColor(1, 2, 3, 4)
        assert uut.get(0, 0) == uut.fill_value
        uut.set(x, y, color)
        assert uut.get(x, y) != uut.fill_value
        assert uut.get(x, y) == color

    def test_vertical_flip(self: Self) -> None:
        uut = self.gradient_fill()
        uut.flip_vertically()
        golden = np.full(shape=(self.GRADIENT_SIZE, self.GRADIENT_SIZE), fill_value=TGAColor())
        for row in range(self.GRADIENT_SIZE):
            for col in range(self.GRADIENT_SIZE):
                v = 10 * row + (self.GRADIENT_SIZE - 1 - col)
                c = TGAColor(v, v, v, v)
                golden[col, row] = c
        assert np.array_equal(uut.npdata, golden)

    @pytest.mark.skipif(bool(os.getenv("QUICK_CHECK")), reason="QUICK_CHECK")
    @pytest.mark.parametrize("vflip", [False, True], ids=["no_vflip", "vflip"])
    @pytest.mark.parametrize("rle", [False, True], ids=["no_rle", "rle"])
    def test_write_file(
        self: Self, tmp_path_factory: pytest.TempPathFactory, file_suite: GoldenFile, vflip: bool, rle: bool
    ) -> None:
        tmpdir = tmp_path_factory.mktemp("test_write_file")
        ofile = f"{file_suite.path.stem}{'_flip' if vflip else ''}{'_rle' if rle else ''}.tga"

        writer = TGAImage.read_tga_file(file_suite.path)
        writer.write_tga_file(tmpdir / ofile, vflip=vflip, rle=rle)

        uut = TGAImage.read_tga_file(tmpdir / ofile)
        self.check_image(uut, file_suite)
        assert uut.was_vflipped == vflip, f"Expected {vflip=} but got {uut.was_vflipped}"
        assert uut.was_rle == rle, f"Expected {rle=} but got {uut.was_rle}"
