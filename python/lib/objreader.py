# TODO: Some testing would be nice...

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Self

from .render import line as render_line
from .tgaimage import TGAColor, TGAColor_t, TGAImage
from .trtypes import vec3 as OBJ_Vertex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Internal representation of an OBJ file... will be expanding as needed...
@dataclass(frozen=True, kw_only=True, slots=True)
class OBJ_Face_entry:
    vertex: int
    texture: int
    normal: int


@dataclass(frozen=True, slots=True)
class OBJ_Face:
    data: tuple[OBJ_Face_entry, OBJ_Face_entry, OBJ_Face_entry]

    def __getitem__(self: Self, idx: int):
        return self.data[idx]


# NOTE: At this time, the 6-entry version of the vertex is unknown (but seems to be a default gray)
# and the 2-vertex version of the texture assumes Z=0
FLOATING_RE: Final = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
VERTEX_RE: Final = re.compile(rf"^v\s+(?P<x>{FLOATING_RE})\s+(?P<y>{FLOATING_RE})\s+(?P<z>{FLOATING_RE})\s*$")
VERTEX_RE_6: Final = re.compile(
    rf"^v\s+(?P<x>{FLOATING_RE})\s+(?P<y>{FLOATING_RE})\s+(?P<z>{FLOATING_RE})\s+(?P<unk4>{FLOATING_RE})\s+(?P<unk5>{FLOATING_RE})\s+(?P<unk6>{FLOATING_RE})\s*$"
)
SINGLE_FACE_RE: Final = r"(?P<vertex_XXX>\d+)/(?P<texture_XXX>\d+)/(?P<normal_XXX>\d+)"
FACE_RE: Final = re.compile(
    r"^f\s+"
    rf"{SINGLE_FACE_RE.replace('XXX', '0')}\s+"
    rf"{SINGLE_FACE_RE.replace('XXX', '1')}\s+"
    rf"{SINGLE_FACE_RE.replace('XXX', '2')}\s*$"
)
V_NORMAL_RE: Final = re.compile(VERTEX_RE.pattern.replace("^v", "^vn"))
TEXTURE_V_RE: Final = re.compile(VERTEX_RE.pattern.replace("^v", "^vt"))
TEXTURE_V_RE_2: Final = re.compile(rf"^vt\s+(?P<x>{FLOATING_RE})\s+(?P<y>{FLOATING_RE})\s*$")
_red3: Final[TGAColor_t] = TGAColor(0, 0, 255)


@dataclass(init=True, slots=True)
class OBJ_Data:
    comments: list[str] = field(init=False, default_factory=list)
    faces: list[OBJ_Face] = field(init=False, default_factory=list)
    texture_vs: list[OBJ_Vertex] = field(init=False, default_factory=list)
    unknowns: list[str] = field(init=False, default_factory=list)
    # Insert a dummy vertex point to make everything 1-based to match face's vertex index:
    vertices: list[OBJ_Vertex] = field(init=False, default_factory=lambda: [OBJ_Vertex(0, 0, 0)])
    v_normals: list[OBJ_Vertex] = field(init=False, default_factory=list)

    def add_comment(self: Self, comment: str) -> None:
        for c in comment[1:].lstrip().split(","):  # Some are doubled up; remove # and split on ,
            self.comments.append(c.strip())

    def add_face(self: Self, face: OBJ_Face) -> None:
        self.faces.append(face)

    def add_texture_v(self: Self, vertex: OBJ_Vertex) -> None:
        self.texture_vs.append(vertex)

    def add_unknown(self: Self, string: str) -> None:
        self.unknowns.append(string)

    def add_vertex(self: Self, vertex: OBJ_Vertex) -> None:
        self.vertices.append(vertex)

    def add_v_normal(self: Self, vertex: OBJ_Vertex) -> None:
        self.v_normals.append(vertex)

    def verify(self: Self) -> None:
        if not __debug__:
            logger.warning("Assertions disabled; obj file cannot be verified")
        for comment in self.comments:
            match comment.split(maxsplit=1):
                case (count, "faces"):
                    icount = int(count)
                    assert (have := len(self.faces)) == icount, (  # noqa: RUF018
                        f"Incorrect number of faces ({have=} expect={icount})"
                    )
                    logger.debug("Faces count confirmed (%d)", icount)
                case (count, "texture vertices") | (count, "coords texture"):  # Seems to be an alias?
                    icount = int(count)
                    assert (have := len(self.texture_vs)) == icount, (  # noqa: RUF018
                        f"Incorrect number of texture vertices ({have=} expect={icount})"
                    )
                    logger.debug("Texture vertex count confirmed (%d)", icount)
                case (count, "vertex normals"):
                    icount = int(count)
                    assert (have := len(self.v_normals)) == icount, (  # noqa: RUF018
                        f"Incorrect number of vertex normals ({have=} expect={icount})"
                    )
                    logger.debug("Vertex normal count confirmed (%d)", icount)
                case (count, "vertices"):  # We insert a blank one
                    icount = int(count)
                    assert (have := len(self.vertices) - 1) == icount, (  # noqa: RUF018
                        f"Incorrect number of vertices ({have=} expect={icount})"
                    )
                    logger.debug("Vertex count confirmed (%d)", icount)
                case _:
                    logger.info("Could not parse comment: %s", comment)

    @staticmethod
    def from_file(infile: str | Path) -> OBJ_Data:
        res = OBJ_Data()
        with open(infile, encoding="utf-8") as ifile:
            for line_no, line in enumerate(ifile, 1):
                match line:
                    case _ if line.strip() == "":
                        # Blank line
                        pass
                    case _ if line.startswith("#"):
                        res.add_comment(line.rstrip())
                    case _ if m := FACE_RE.match(line):
                        f0 = OBJ_Face_entry(
                            vertex=int(m["vertex_0"]), texture=int(m["texture_0"]), normal=int(m["normal_0"])
                        )
                        f1 = OBJ_Face_entry(
                            vertex=int(m["vertex_1"]), texture=int(m["texture_1"]), normal=int(m["normal_1"])
                        )
                        f2 = OBJ_Face_entry(
                            vertex=int(m["vertex_2"]), texture=int(m["texture_2"]), normal=int(m["normal_2"])
                        )
                        res.add_face(OBJ_Face((f0, f1, f2)))
                    case _ if m := TEXTURE_V_RE.match(line):
                        res.add_texture_v(OBJ_Vertex(float(m["x"]), float(m["y"]), float(m["z"])))
                    case _ if m := TEXTURE_V_RE_2.match(line):
                        res.add_texture_v(OBJ_Vertex(float(m["x"]), float(m["y"]), 0))
                    case _ if m := VERTEX_RE.match(line):
                        res.add_vertex(OBJ_Vertex(float(m["x"]), float(m["y"]), float(m["z"])))
                    case _ if m := VERTEX_RE_6.match(line):
                        if m["unk4"] == m["unk5"] == m["unk6"] == "0.752941":
                            res.add_vertex(OBJ_Vertex(float(m["x"]), float(m["y"]), float(m["z"])))
                        else:
                            logger.debug("Unknown 6-entry vertex where unknown values didn't match: %s", line.rstrip())
                    case _ if m := V_NORMAL_RE.match(line):
                        res.add_v_normal(OBJ_Vertex(float(m["x"]), float(m["y"]), float(m["z"])))
                    case _:
                        logger.debug("Could not interpret line %d: %s", line_no, line.rstrip())
                        res.add_unknown(line.rstrip())
        logger.debug("Read OBJ file %s: %s", Path(infile).name, res)
        res.verify()
        return res

    def orthogonal_projection(
        self: Self, *, width: int = 2048, height: int = 2048, color: TGAColor_t = _red3
    ) -> TGAImage:
        # Basically, homework 1
        # The numbers go from -1..1 so we need to map them from the center of the image...
        width_center: Final[int] = width // 2
        height_center: Final[int] = height // 2

        framebuffer = TGAImage(width, height, TGAImage.Format.RGB)

        for face in self.faces:
            # Get the indices of the vertices
            idx = (face[0].vertex, face[1].vertex, face[2].vertex)
            # Read those out
            points = (self.vertices[idx[0]], self.vertices[idx[1]], self.vertices[idx[2]])
            # Draw the lines
            for i in range(3):
                this = i % 3
                that = (i + 1) % 3
                # The extra "-1" is to shift the image into the correct quadrant; e.g.:
                # DC => AB
                # BA    CD
                # Because our origin is the bottom left corner of C but the OBJ is at center of image
                # (bottom left corner of C)
                render_line(
                    round((points[this].x - 1) * (width_center)),
                    round((points[this].y - 1) * (height_center)),
                    round((points[that].x - 1) * (width_center)),
                    round((points[that].y - 1) * (height_center)),
                    framebuffer,
                    color,
                )
        return framebuffer

    def __str__(self: Self) -> str:
        return (
            f"OBJ_Data({hex(id(self))}) with "
            f"{len(self.faces)} face(s), "
            f"{len(self.vertices) - 1} vert(ex|ices), "
            f"{len(self.v_normals)} vertex normal(s), "
            f"{len(self.texture_vs)} texture vert(ex|ices)"
        )
