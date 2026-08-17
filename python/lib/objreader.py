# TODO: Some testing would be nice...
# The OBJ file seems to have comments with the counts of each type it should see?

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

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


@dataclass(frozen=True, slots=True)
class OBJ_Vertex:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class OBJ_Data:
    faces: list[OBJ_Face]
    vertices: list[OBJ_Vertex]

    def __init__(self: Self) -> None:
        # Insert a dummy vertex point to make everything 1-based to match face's vertex index
        # TODO: Consider better interfaces like to pre-size the list, etc?
        self.faces = []
        self.vertices = [OBJ_Vertex(0, 0, 0)]

    def add_face(self: Self, face: OBJ_Face) -> None:
        self.faces.append(face)

    def add_vertex(self: Self, vertex: OBJ_Vertex) -> None:
        self.vertices.append(vertex)

    def __str__(self: Self) -> str:
        return f"OBJ_Data({hex(id(self))}) with {len(self.faces)} face(s) and {len(self.vertices) - 1} vert(ex|ices)"


FLOATING_RE: Final = r"-?\d+(?:\.\d+)?"
VERTEX_RE: Final = re.compile(rf"^v\s+(?P<x>{FLOATING_RE})\s+(?P<y>{FLOATING_RE})\s+(?P<z>{FLOATING_RE})\s*$")
SINGLE_FACE_RE: Final = r"(?P<vertex_XXX>\d+)/(?P<texture_XXX>\d+)/(?P<normal_XXX>\d+)"
FACE_RE: Final = re.compile(
    r"^f\s+"
    rf"{SINGLE_FACE_RE.replace('XXX', '0')}\s+"
    rf"{SINGLE_FACE_RE.replace('XXX', '1')}\s+"
    rf"{SINGLE_FACE_RE.replace('XXX', '2')}\s*$"
)


def read_obj_file(infile: str | Path) -> OBJ_Data:
    res = OBJ_Data()
    with open(infile, encoding="utf-8") as ifile:
        for line_no, line in enumerate(ifile, 1):
            if m := VERTEX_RE.match(line):
                res.add_vertex(OBJ_Vertex(float(m["x"]), float(m["y"]), float(m["z"])))
            elif m := FACE_RE.match(line):
                f0 = OBJ_Face_entry(vertex=int(m["vertex_0"]), texture=int(m["texture_0"]), normal=int(m["normal_0"]))
                f1 = OBJ_Face_entry(vertex=int(m["vertex_1"]), texture=int(m["texture_1"]), normal=int(m["normal_1"]))
                f2 = OBJ_Face_entry(vertex=int(m["vertex_2"]), texture=int(m["texture_2"]), normal=int(m["normal_2"]))
                res.add_face(OBJ_Face((f0, f1, f2)))
            elif line.strip() == "" or line.startswith("#"):
                # Empty / comment
                pass
            else:
                logger.debug("Could not interpret line %d: %s", line_no, line.rstrip())
    return res
