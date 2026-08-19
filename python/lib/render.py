from __future__ import annotations

from typing import Final

from .tgaimage import TGAColor_t, TGAImage


def line(
    ax: int, ay: int, bx: int, by: int, framebuffer: TGAImage, color: TGAColor_t, *, draw: bool = True
) -> list[tuple[int, int]]:
    """Draw a line (Lesson 1)"""
    steep: Final = abs(ax - bx) < abs(ay - by)
    if steep:
        ax, ay = ay, ax
        bx, by = by, bx
    if ax > bx:  # make it left-to-right
        ax, bx = bx, ax
        ay, by = by, ay
    y: int = ay
    error: int = 0
    error_inc: Final[int] = 2 * (bx - ax)
    points = []
    for x in range(ax, min(bx + 1, framebuffer.width)):
        if steep:  # if transposed, de-transpose
            if draw:
                framebuffer.set(y, x, color)
            points.append((y, x))
        else:
            if draw:
                framebuffer.set(x, y, color)
            points.append((x, y))
        error += 2 * abs(by - ay)
        if error > (bx - ax):
            y += 1 if by > ay else -1
            error -= error_inc
    return points


def triangle_scanlines(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    framebuffer: TGAImage,
    color: TGAColor_t,
    *,
    fill: bool = True,
) -> None:
    """Draw an unfilled triangle"""
    points = line(*a, *b, framebuffer, color, draw=not fill)
    points.extend(line(*b, *c, framebuffer, color, draw=not fill))
    points.extend(line(*c, *a, framebuffer, color, draw=not fill))

    if fill:
        # Scan-line algorithm: no need to sort the Y values just throw them in a set
        for y in {pt[1] for pt in points}:
            # Get all the points drawn at that Y
            this_line = [(px, py) for px, py in points if py == y]
            min_x = min(pt[0] for pt in this_line)
            max_x = max(pt[0] for pt in this_line)
            line(min_x, y, max_x, y, framebuffer, color)


def _signed_triangle_area(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> float:
    return 0.5 * ((by - ay) * (bx + ax) + (cy - by) * (cx + bx) + (ay - cy) * (ax + cx))


def triangle_barycentric(
    a: tuple[int, int],
    b: tuple[int, int],
    c: tuple[int, int],
    framebuffer: TGAImage,
    color: TGAColor_t,
    *,
    fill: bool = True,
) -> None:
    assert fill, "This version can't do non-filled!"
    ax, ay = a
    bx, by = b
    cx, cy = c
    bb_min_x: Final[int] = min(ax, bx, cx)
    bb_max_x: Final[int] = max(ax, bx, cx)
    bb_min_y: Final[int] = min(ay, by, cy)
    bb_max_y: Final[int] = max(ay, by, cy)
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel
    ta_s = -1 if total_area < 0 else 1  # We only seem to care about the sign, so why do real division?

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) * ta_s  # / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) * ta_s  # / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) * ta_s  # / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            framebuffer.set(x, y, color)


triangle = triangle_barycentric
