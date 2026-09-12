from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Self

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from lib.tgaimage import TGAColor, TGAColor_t, TGAImage, uint8_t
from lib.tgaimage import err as plot_err

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

valid_uint8_t = st.integers(0, 255)
valid_uint8_t_div2 = st.integers(0, 127)


@cache
def _read_tga_file(tfile: str | Path) -> TGAImage:
    """Helper to cache the files - full test 44s => 28s"""
    return TGAImage.read_tga_file(tfile)


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
        Path(Path(__file__).parent / "../.." / "../obj/floor_spec.tga").resolve(),
        "RGB",
        RLE=True,
        colors=1,
        height=1,
        width=1,
    ),
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


def color_to_bpp(color_type: str) -> int:
    match color_type:
        case "Mono":
            return 1
        case "RGB":
            return 3
        case "RGBA":
            return 4
    err_msg = f"Invalid color string '{color_type}'"
    raise ValueError(err_msg)


@pytest.fixture(params=TEST_FILES, ids=[str(g) for g in TEST_FILES])
def file_suite(request: pytest.FixtureRequest) -> Generator[GoldenFile]:
    """Converts TEST_FILES into pytest-native input iterators"""
    if os.getenv("QUICK_CHECK") and request.param.width >= 16:
        pytest.skip("QUICK_CHECK set")
    return request.param


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

    def test_bad_init(self: Self, subtests: pytest.Subtests) -> None:
        with subtests.test("Require TGAColor() helper"), pytest.raises(TypeError, match="call TGAColor"):
            TGAColor_t(b=1, g=1, r=1, a=1)
        with subtests.test("Bad BPP (5..255)"):
            for bpp in range(5, 256):
                with pytest.raises(ValueError, match="Invalid value given"):
                    TGAColor(bpp=bpp)
        with subtests.test("BPP >= 256"), pytest.raises(OverflowError):
            TGAColor(bpp=256)
        with subtests.test("No BPP, bad values"):
            with pytest.raises(ValueError):
                TGAColor(b=None, g=1, r=1)
            with pytest.raises(ValueError):
                TGAColor(b=1, g=None, r=1)
            with pytest.raises(ValueError):
                TGAColor(b=1, g=1, r=None, a=1)
        # Bad values
        for test_value in {-1, 256}:
            with subtests.test(f"Invalid color value {test_value}"):
                with pytest.raises(ValueError):
                    TGAColor(b=test_value)
                with pytest.raises(ValueError):
                    TGAColor(b=1, g=test_value)
                with pytest.raises(ValueError):
                    TGAColor(b=1, g=1, r=test_value)
                with pytest.raises(ValueError):
                    TGAColor(b=1, g=1, r=1, a=test_value)

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

    @pytest.mark.parametrize("bpp", range(3, 5), ids=[f"bpp={b}" for b in range(3, 5)])
    @given(b=valid_uint8_t, g=valid_uint8_t, r=valid_uint8_t, a=valid_uint8_t)
    def test_rgba(self: Self, *, b: int, g: int, r: int, a: int, bpp: int) -> None:
        uut = TGAColor(b, g, r, a, bpp=bpp)
        assert uut.rgba[0] == r
        assert uut.rgba[1] == g
        assert uut.rgba[2] == b
        if bpp == 4:
            assert uut.rgba[3] == a
        else:
            with pytest.raises(IndexError):
                uut.rgba[3]

    @given(b=valid_uint8_t, g=valid_uint8_t, r=valid_uint8_t, a=valid_uint8_t)
    def test_caching(self: Self, *, b: int, g: int, r: int, a: int) -> None:
        assert TGAColor(1, 2, 3, 4) is TGAColor(1, 2, 3, 4)
        assert TGAColor(1, 2, 3) is TGAColor(1, 2, 3)
        assert TGAColor(1, 2) is TGAColor(1, 2)
        assert TGAColor(1) is TGAColor(1)
        assert TGAColor() is TGAColor()
        assert TGAColor(b=b, g=g, r=r, a=a) is TGAColor(b=b, g=g, r=r, a=a)
        assert TGAColor(b=b, g=g, r=r) is TGAColor(b=b, g=g, r=r)
        assert TGAColor(b=b, g=g) is TGAColor(b=b, g=g)
        assert TGAColor(b) is TGAColor(b)

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

    def test_sorting(self: Self) -> None:
        assert TGAColor(1, 2, 3, 4) <= TGAColor(1, 2, 3, 4, bpp=4)
        assert TGAColor(1, 2, 3) <= TGAColor(1, 2, 3, bpp=3)
        assert TGAColor(1, 2) <= TGAColor(1, 2, bpp=2)
        assert TGAColor(1) <= TGAColor(1, bpp=1)
        assert TGAColor() <= TGAColor(0, bpp=1)

        assert TGAColor(1, 2, 3, 4) <= TGAColor(1, 2, 3, 5, bpp=4)
        assert TGAColor(1, 2, 3) <= TGAColor(1, 2, 4, bpp=3)
        assert TGAColor(1, 2) <= TGAColor(1, 3, bpp=2)
        assert TGAColor(1) <= TGAColor(2, bpp=1)
        assert TGAColor() <= TGAColor(1, bpp=1)

        assert TGAColor(1, 2, 3, 4) >= TGAColor(1, 2, 3, 4, bpp=4)
        assert TGAColor(1, 2, 3) >= TGAColor(1, 2, 3, bpp=3)
        assert TGAColor(1, 2) >= TGAColor(1, 2, bpp=2)
        assert TGAColor(1) >= TGAColor(1, bpp=1)
        assert TGAColor() >= TGAColor(0, bpp=1)

        assert TGAColor(1, 2, 3, 5) >= TGAColor(1, 2, 3, 4, bpp=4)
        assert TGAColor(1, 2, 4) >= TGAColor(1, 2, 3, bpp=3)
        assert TGAColor(1, 3) >= TGAColor(1, 2, bpp=2)
        assert TGAColor(2) >= TGAColor(1, bpp=1)
        assert TGAColor(1) >= TGAColor()

    @given(b=valid_uint8_t_div2, g=valid_uint8_t_div2, r=valid_uint8_t_div2, a=valid_uint8_t)
    def test_max_color(self: Self, b: int, g: int, r: int, a: int) -> None:
        uut = TGAColor(b, g, r, a)
        max_val = max((b, g, r))
        assume(a >= max_val)  # Want it to be independent of alpha
        assert uut.max_color == max_val

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    @given(bgra_in=st.lists(valid_uint8_t, min_size=4, max_size=4))
    def test_max_color_parameterized(self: Self, bgra_in: list[int], bpp: int) -> None:
        uut = TGAColor(*bgra_in, bpp=bpp)
        max_color_idx = min(bpp, 3)  # Want to ignore alpha (if present)
        max_val = max(bgra_in[:max_color_idx])
        assert uut.max_color == max_val

    def test_set_invalid_bpp(self: Self, *, subtests: pytest.Subtests) -> None:
        with subtests.test("Downsizing"):
            uut = TGAImage(w=1, h=1, bpp=1)
            for bpp in range(2, 5):
                with pytest.warns(UserWarning, match="changed bpp"):
                    uut.set(0, 0, TGAColor_t.random(bpp=uint8_t(bpp)))
        with subtests.test("Upsizing"):
            uut = TGAImage(w=1, h=1, bpp=4)
            for bpp in range(1, 4):
                if bpp == 3:
                    # pytest.skip("BPP 3=>4 allowed")
                    uut.set(0, 0, TGAColor_t.random(bpp=uint8_t(bpp)))
                else:
                    with pytest.raises(ValueError):
                        uut.set(0, 0, TGAColor_t.random(bpp=uint8_t(bpp)))

    @given(box=st.integers(min_value=1, max_value=32), new_max=valid_uint8_t)
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_set_max(
        self: Self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests, box: int, new_max: int
    ) -> None:
        uut = TGAImage(w=box, h=box, bpp=3)

        def _helper(text: str, start: TGAColor_t) -> None:
            with subtests.test(f"{text}/{box}x{box}/{new_max}"):
                uut.set(box // 2, box // 2, start)
                uut.set_max(new_max)
                assert uut.get(box // 2, box // 2).max_color == new_max

        with subtests.test(f"Black/{box}x{box}/{new_max}"), caplog.at_level("DEBUG"):
            uut.set(box, box, TGAColor(0, 0, 0))
            uut.set_max(new_max)
            assert "all-black" in caplog.text

        _helper("Tiny Red", TGAColor(r=2, g=0, b=0))
        _helper("Half Red", TGAColor(r=128, g=0, b=0))
        _helper("Full Red", TGAColor(r=255, g=0, b=0))
        _helper("Tiny Green", TGAColor(r=0, g=2, b=0))
        _helper("Half Green", TGAColor(r=0, g=128, b=0))
        _helper("Full Green", TGAColor(r=0, g=255, b=0))
        _helper("Tiny Blue", TGAColor(r=0, g=0, b=2))
        _helper("Half Blue", TGAColor(r=0, g=0, b=128))
        _helper("Full Blue", TGAColor(r=0, g=0, b=255))
        _helper("Tiny Gray", TGAColor(r=2, g=2, b=2))
        _helper("Half Gray", TGAColor(r=128, g=128, b=128))
        _helper("Middling Gray", TGAColor(r=127, g=128, b=127))

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    @given(bgra_in=st.lists(valid_uint8_t, min_size=8, max_size=8))
    def test_sorting_hypothesis_le(self: Self, bgra_in: list[int], bpp: int) -> None:
        b1, b2, g1, g2, r1, r2, a1, a2 = sorted(bgra_in)
        uut = TGAColor(b1, g1, r1, a1, bpp=bpp)
        uut2 = TGAColor(b2, g2, r2, a2, bpp=bpp)
        assert uut <= uut2
        assert not (uut > uut2)

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    @given(bgra_in=st.lists(valid_uint8_t, min_size=8, max_size=8, unique=True))
    def test_sorting_hypothesis_lt(self: Self, bgra_in: list[int], bpp: int) -> None:
        b1, b2, g1, g2, r1, r2, a1, a2 = sorted(bgra_in)
        uut = TGAColor(b1, g1, r1, a1, bpp=bpp)
        uut2 = TGAColor(b2, g2, r2, a2, bpp=bpp)
        assert uut < uut2
        assert not (uut > uut2)
        assert uut != uut2

    def test_frozen(self: Self) -> None:
        uut = TGAColor(1, 2, 3, 4)
        assert uut[1] == 2
        with pytest.raises(AttributeError):
            uut[1] = uint8_t(2)

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    def test_random(self: Self, *, bpp: int) -> None:
        uut = TGAColor_t.random(bpp=uint8_t(bpp))
        assert uut.bytespp == bpp
        assert all(0 <= x <= 255 for x in uut._data)

    @pytest.mark.parametrize("bpp", range(2, 5), ids=[f"bpp={b}" for b in range(2, 5)])
    def test_resize(self: Self, *, bpp: int, subtests: pytest.Subtests) -> None:
        uut = TGAColor(*[1, 2, 3, 4][:bpp])
        for test_bpp in range(1, 4):
            if test_bpp == bpp:
                continue
            if test_bpp > bpp:
                with pytest.raises(ValueError):
                    uut.resize(test_bpp)
            else:
                with subtests.test(f"Resize down to {test_bpp}"):
                    uut2 = uut.resize(test_bpp)
                    assert all(uut._data[i] == uut2._data[i] for i in range(test_bpp))

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

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    @given(bgra_in=st.lists(valid_uint8_t, min_size=4, max_size=4, unique=True))
    def test_shortcuts(self: Self, bgra_in: list[int], bpp: int) -> None:
        uut = TGAColor(*bgra_in, bpp=bpp)
        assert uut.b == bgra_in[0]
        if bpp >= 2:
            assert uut.g == bgra_in[1]
        else:
            with pytest.raises(ValueError):
                _ = uut.g
        if bpp >= 3:
            assert uut.r == bgra_in[2]
        else:
            with pytest.raises(ValueError):
                _ = uut.r
        if bpp >= 4:
            assert uut.a == bgra_in[3]
        else:
            with pytest.raises(ValueError):
                _ = uut.a

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    @given(bgra_in=st.lists(valid_uint8_t_div2, min_size=4, max_size=4, unique=True))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_scaling(self: Self, bgra_in: list[int], bpp: int, subtests: pytest.Subtests) -> None:
        with subtests.test("Multiplication"):
            uut = 2 * TGAColor(*bgra_in, bpp=bpp)  # __rmul__
            assert uut.b == 2 * bgra_in[0]
            if bpp >= 2:
                assert uut.g == 2 * bgra_in[1]
            if bpp >= 3:
                assert uut.r == 2 * bgra_in[2]
            if bpp >= 4:
                assert uut.a == 2 * bgra_in[3]
            uut = TGAColor(*bgra_in, bpp=bpp) * 1.5  # __mul__
            assert uut.b == round(1.5 * bgra_in[0])
            if bpp >= 2:
                assert uut.g == round(1.5 * bgra_in[1])
            if bpp >= 3:
                assert uut.r == round(1.5 * bgra_in[2])
            if bpp >= 4:
                assert uut.a == round(1.5 * bgra_in[3])
        with subtests.test("Division"):
            uut = TGAColor(*bgra_in, bpp=bpp) / 2
            assert uut.b == round(bgra_in[0] / 2)
            if bpp >= 2:
                assert uut.g == round(bgra_in[1] / 2)
            if bpp >= 3:
                assert uut.r == round(bgra_in[2] / 2)
            if bpp >= 4:
                assert uut.a == round(bgra_in[3] / 2)
            uut = TGAColor(*bgra_in, bpp=bpp) / 3.2
            assert uut.b == round(bgra_in[0] / 3.2)
            if bpp >= 2:
                assert uut.g == round(bgra_in[1] / 3.2)
            if bpp >= 3:
                assert uut.r == round(bgra_in[2] / 3.2)
            if bpp >= 4:
                assert uut.a == round(bgra_in[3] / 3.2)
        with subtests.test("Saturation"):
            uut = TGAColor(*[130, 150, 180, 220][:bpp], bpp=bpp)
            uut2 = 2 * uut
            assert list(uut2._data) == [255] * bpp  # _data is a tuple

    @pytest.mark.parametrize("bpp", range(1, 5), ids=[f"bpp={b}" for b in range(1, 5)])
    def test_scaling_zero(self: Self, *, bpp: int) -> None:
        with pytest.raises(ZeroDivisionError):
            TGAColor_t.random(bpp=uint8_t(bpp)) / 0  # pyrefly: ignore[division-by-zero]


@st.composite
def limited_xy(draw: Callable, min_: int = 0) -> tuple[int, int]:
    """
    Gives a tuple where the max (first value) is random and the second will be less

    This is used to randomize tests where we want an (x, y) and know it is an index within (w, h)
    """
    # The TGA spec is 2^16-1, but that ends up being really slow and doesn't really "prove" much
    MAX_VAL: Final[int] = pow(2, 10) - 1  # ruff: ignore[non-lowercase-variable-in-function]
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
        assert (count := TestTGAImage.count_unique_colors(uut)) == metadata.colors, (
            f"Expected {metadata.colors} colors, got {count}"
        )

    @staticmethod
    def count_unique_colors(uut: TGAImage) -> int:
        uniques = set(uut.npdata.flat)
        return len(uniques)

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
                    assert cls.GRADIENT is not None, "Useless but keeping typecheckers happy"
                    cls.GRADIENT[col].append(c)
        return uut

    # Test methods:
    def test_bad_file(self: Self) -> None:
        uut = TGAImage()
        with pytest.raises(OSError):
            uut.read_tga_file("/abc.tga")

    @given(box=st.integers(min_value=1, max_value=32))
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_brighten(self: Self, caplog: pytest.LogCaptureFixture, subtests: pytest.Subtests, box: int) -> None:
        uut = TGAImage(w=box, h=box, bpp=3)

        def _helper(text: str, start: TGAColor_t, expected: TGAColor_t) -> None:
            with subtests.test(f"{text}/{box}x{box}"):
                uut.set(box // 2, box // 2, start)
                uut.brighten()
                assert uut.get(box // 2, box // 2) == expected

        with subtests.test(f"Black/{box}x{box}"), caplog.at_level("DEBUG"):
            uut.set(box, box, TGAColor(0, 0, 0))
            uut.brighten()
            assert "all-black" in caplog.text

        _helper("Tiny Red", TGAColor(r=2, g=0, b=0), TGAColor(r=255, g=0, b=0))
        _helper("Half Red", TGAColor(r=128, g=0, b=0), TGAColor(r=255, g=0, b=0))
        _helper("Full Red", TGAColor(r=255, g=0, b=0), TGAColor(r=255, g=0, b=0))
        _helper("Tiny Green", TGAColor(r=0, g=2, b=0), TGAColor(r=0, g=255, b=0))
        _helper("Half Green", TGAColor(r=0, g=128, b=0), TGAColor(r=0, g=255, b=0))
        _helper("Full Green", TGAColor(r=0, g=255, b=0), TGAColor(r=0, g=255, b=0))
        _helper("Tiny Blue", TGAColor(r=0, g=0, b=2), TGAColor(r=0, g=0, b=255))
        _helper("Half Blue", TGAColor(r=0, g=0, b=128), TGAColor(r=0, g=0, b=255))
        _helper("Full Blue", TGAColor(r=0, g=0, b=255), TGAColor(r=0, g=0, b=255))
        _helper("Tiny Gray", TGAColor(r=2, g=2, b=2), TGAColor(r=255, g=255, b=255))
        _helper("Half Gray", TGAColor(r=128, g=128, b=128), TGAColor(r=255, g=255, b=255))
        _helper("Middling Gray", TGAColor(r=127, g=128, b=127), TGAColor(r=253, g=255, b=253))

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

    def test_good_files(self: Self, file_suite: GoldenFile) -> None:
        TestTGAImage.skip_missing(file_suite.path)
        uut = _read_tga_file(file_suite.path)
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

    def test_rle_unrle(self: Self, subtests: pytest.Subtests, file_suite: GoldenFile) -> None:
        # Sometimes the raw RLE didn't match, but the re-expanded matches...
        TestTGAImage.skip_missing(file_suite.path)
        with subtests.test("Reading"):
            # Un-RLE the data...
            uut = _read_tga_file(file_suite.path)
            assert str(uut) == f"{file_suite.width}x{file_suite.height}/{color_to_bpp(file_suite.color_type)}"
        with subtests.test("Extracting"):
            # Get the original data back out
            golden_data = uut._raw_payload
        with subtests.test("Recompressing"):
            # Re-RLE the data...
            re_compressed = uut.unload_rle_data()
        with subtests.test("Re-expanding"):
            # Expand that
            test_data = uut.load_rle_data(re_compressed)
        with subtests.test("Comparing"):
            # These are broken out to stop pytest from being "helpful" and dumping all the data to the screen...
            len_test = len(test_data)
            len_golden = len(golden_data)
            assert len_test == len_golden
            CHUNK_SIZE = 32  # ruff: ignore[non-lowercase-variable-in-function]
            for chunk in range(0, len(golden_data), CHUNK_SIZE):
                # The first half is to let you see where the difference is in absolute terms
                abs_loc = chunk * CHUNK_SIZE
                assert abs_loc >= 0
                assert golden_data[chunk : chunk + CHUNK_SIZE] == test_data[chunk : chunk + CHUNK_SIZE]

    @given(h_y=limited_xy(), w_x=limited_xy())
    def test_set_get(self: Self, h_y: tuple[int, int], w_x: tuple[int, int]) -> None:
        (h, y) = h_y
        (w, x) = w_x
        # Minimum 1x1 image:
        h = h or 1
        w = w or 1
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

    @pytest.mark.parametrize("vflip", [False, True], ids=["no_vflip", "vflip"])
    @pytest.mark.parametrize("rle", [False, True], ids=["no_rle", "rle"])
    def test_write_file(
        self: Self, tmp_path_factory: pytest.TempPathFactory, file_suite: GoldenFile, vflip: bool, rle: bool
    ) -> None:
        tmpdir = tmp_path_factory.mktemp("test_write_file")
        ofile = f"{file_suite.path.stem}{'_flip' if vflip else ''}{'_rle' if rle else ''}.tga"

        writer = _read_tga_file(file_suite.path)
        writer.write_tga_file(tmpdir / ofile, vflip=vflip, rle=rle)

        uut = TGAImage.read_tga_file(tmpdir / ofile)
        self.check_image(uut, file_suite)
        assert uut.was_vflipped == vflip, f"Expected {vflip=} but got {uut.was_vflipped}"
        assert uut.was_rle == rle, f"Expected {rle=} but got {uut.was_rle}"

    @pytest.mark.skipif(plot_err is not None, reason="matplotlib wasn't imported")
    def test_plot(self: Self, file_suite: GoldenFile) -> None:
        TestTGAImage.skip_missing(file_suite.path)
        uut = _read_tga_file(file_suite.path)
        uut.plot(_test_mode=True)
