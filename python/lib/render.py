from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np

from .tgaimage import TGAColor, TGAColor_t, TGAImage
from .trtypes import Matrix3f, Matrix4f, ZBuffer, vec2, vec3, vec4


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


def triangle_barycentric_lesson_3(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    framebuffer: TGAImage,
) -> None:
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    bb_min_x: Final[int] = min(ax, bx, cx)
    bb_max_x: Final[int] = max(ax, bx, cx)
    bb_min_y: Final[int] = min(ay, by, cy)
    bb_max_y: Final[int] = max(ay, by, cy)
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z = round(alpha * az + beta * bz + gamma * cz)
            assert z <= 255, f"Bad {z=}"

            framebuffer.set(x, y, TGAColor(z))


def triangle_barycentric_lesson_3_homework(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    framebuffer: TGAImage,
) -> None:
    ax, ay, _az = a
    bx, by, _bz = b
    cx, cy, _cz = c
    bb_min_x: Final[int] = min(ax, bx, cx)
    bb_max_x: Final[int] = max(ax, bx, cx)
    bb_min_y: Final[int] = min(ay, by, cy)
    bb_max_y: Final[int] = max(ay, by, cy)
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            color1 = round(alpha * 255)
            color2 = round(beta * 255)
            color3 = round(gamma * 255)
            assert color1 <= 255 and color2 <= 255 and color3 <= 255, f"Bad {color1=} {color2=} {color3=}"
            if alpha <= 0.1 or beta <= 0.1 or gamma <= 0.1:
                framebuffer.set(x, y, TGAColor(color1, color2, color3))


def triangle_barycentric_lesson_4(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    z_buffer: TGAImage,
    framebuffer: TGAImage,
    color: TGAColor_t,
) -> None:
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    bb_min_x: Final[int] = min(ax, bx, cx)  # bounding box for the triangle
    bb_max_x: Final[int] = max(ax, bx, cx)  # defined by its top left and bottom right corners
    bb_min_y: Final[int] = min(ay, by, cy)
    bb_max_y: Final[int] = max(ay, by, cy)
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z = round(alpha * az + beta * bz + gamma * cz)
            z_color = TGAColor(z)
            if z_color <= z_buffer.get(x, y):  # Behind what we've already drawn
                continue
            z_buffer.set(x, y, z_color)
            framebuffer.set(x, y, color)


def triangle_barycentric_lesson_5(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    c: tuple[int, int, int],
    z_buffer: ZBuffer,
    framebuffer: TGAImage,
    color: TGAColor_t,
) -> None:
    assert isinstance(z_buffer, ZBuffer), "Wrong version of triangle_barycentric called!"
    ax, ay, az = a
    bx, by, bz = b
    cx, cy, cz = c
    total_area: Final = _signed_triangle_area(ax, ay, bx, by, cx, cy)
    if total_area < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    bb_min_x: Final[int] = max(0, min(ax, bx, cx))  # bounding box for the triangle
    bb_max_x: Final[int] = min(framebuffer.width, max(ax, bx, cx))  # defined by its top left and bottom right corners
    bb_min_y: Final[int] = max(0, min(ay, by, cy))  # but bound by canvas size
    bb_max_y: Final[int] = min(framebuffer.height, max(ay, by, cy))

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            alpha = _signed_triangle_area(x, y, bx, by, cx, cy) / total_area
            beta = _signed_triangle_area(x, y, cx, cy, ax, ay) / total_area
            gamma = _signed_triangle_area(x, y, ax, ay, bx, by) / total_area
            if alpha < 0 or beta < 0 or gamma < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z = alpha * az + beta * bz + gamma * cz
            if z <= z_buffer.vals[x][y]:  # Behind what we've already drawn
                continue
            z_buffer.vals[x][y] = z
            framebuffer.set(x, y, color)


def rasterize_lesson_6(
    clip: Sequence[vec4],
    viewport: Matrix4f,
    z_buffer: ZBuffer,
    framebuffer: TGAImage,
    color: TGAColor_t,
) -> None:
    ndc: Final[list[vec4]] = [
        clip[0] * (1 / clip[0].w),
        clip[1] * (1 / clip[1].w),
        clip[2] * (1 / clip[2].w),
    ]  # normalized device coordinates
    screen: Final[list[vec2]] = [
        vec4.from_np(viewport @ ndc[0].np).xy,
        vec4.from_np(viewport @ ndc[1].np).xy,
        vec4.from_np(viewport @ ndc[2].np).xy,
    ]  # screen coordinates
    ABC: Final[Matrix3f] = np.array([
        [screen[0].x, screen[0].y, 1.0],
        [screen[1].x, screen[1].y, 1.0],
        [screen[2].x, screen[2].y, 1.0],
    ])
    if np.linalg.det(ABC) < 1:
        return  # Early return for backface culling + discarding triangles that cover less than a pixel

    # bounding box for the triangle defined by its top left and bottom right corners but bound by canvas size
    bb_min_x: Final[int] = int(max(0, min(screen[0].x, screen[1].x, screen[2].x)))
    bb_max_x: Final[int] = int(min(framebuffer.width - 1, max(screen[0].x, screen[1].x, screen[2].x)))
    bb_min_y: Final[int] = int(max(0, min(screen[0].y, screen[1].y, screen[2].y)))
    bb_max_y: Final[int] = int(min(framebuffer.height - 1, max(screen[0].y, screen[1].y, screen[2].y)))

    for x in range(bb_min_x, bb_max_x + 1):
        for y in range(bb_min_y, bb_max_y + 1):
            ABC_invert_transpose = np.linalg.inv(ABC.T)
            # bc = barycentric coordinates of {x,y} w.r.t the triangle
            bc: vec3 = vec3.from_np(ABC_invert_transpose @ vec3(x, y, 1).np)
            if bc.x < 0 or bc.y < 0 or bc.z < 0:
                continue  # negative barycentric coordinate => the pixel is outside the triangle
            z = bc * vec3(ndc[0].z, ndc[1].z, ndc[2].z)
            if z <= z_buffer.vals[x][y]:  # Behind what we've already drawn
                continue
            z_buffer.vals[x][y] = z
            framebuffer.set(x, y, color)


triangle = triangle_barycentric
