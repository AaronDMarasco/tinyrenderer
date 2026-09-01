from __future__ import annotations

import logging
import re
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Self, get_args

from .tgaimage import TGAImage
from .trtypes import vec2, vec3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


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
GROUP_RE: Final = re.compile(r"^g\s*(\w*)$")
SMOOTH_RE: Final = re.compile(r"^s\s+(\d+|off)$")


_SENTINEL = object()

SupportFiles = Literal["diffuse", "glow", "gloss", "nm", "nm_tangent", "spec"]
SUPPORT_FILES: Final = sorted(get_args(SupportFiles))


@dataclass(init=True, slots=True)
class Model:
    @dataclass(frozen=True, kw_only=True, slots=True)
    class FaceEntry:
        """Single Entry for a Face"""

        vertex: int
        texture: int
        normal: int

    @dataclass(frozen=True, slots=True)
    class Face:
        """A Face"""

        data: tuple[Model.FaceEntry, Model.FaceEntry, Model.FaceEntry]
        group: str | None = field(default=None, kw_only=True)
        smooth: int | None = field(default=None, kw_only=True)

        def __getitem__(self: Self, idx: int) -> Model.FaceEntry:
            return self.data[idx]

    comments: list[str] = field(init=False, default_factory=list)
    faces: list[Model.Face] = field(init=False, default_factory=list)
    # Insert a dummy vertex point to make everything 1-based to match face's vertex index:
    texture_vs: list[vec3] = field(init=False, default_factory=lambda: [vec3(0, 0, 0)])
    vertices: list[vec3] = field(init=False, default_factory=lambda: [vec3(0, 0, 0)])
    v_normals: list[vec3] = field(init=False, default_factory=lambda: [vec3(0, 0, 0)])
    # unknowns: list[str] = field(init=False, default_factory=list)
    # Various support files that may be present:
    ext: dict[SupportFiles, TGAImage] = field(default_factory=dict)
    # Don't allow anybody else to (easily) create this class:
    _guard: InitVar[Any] = field(default=None)

    def __post_init__(self: Self, _guard: Any) -> None:
        if _guard is not _SENTINEL:
            err_msg = "Only call Model.from_file() method to get a Model"
            raise TypeError(err_msg)

    @property
    def groups(self: Self) -> set[str | None]:
        groups = {f.group for f in self.faces}
        return groups

    def add_comment(self: Self, comment: str) -> None:
        for c in comment[1:].lstrip().split(","):  # Some are doubled up; remove # and split on ,
            self.comments.append(c.strip())

    def add_face(self: Self, face: Face) -> None:
        self.faces.append(face)

    def add_texture_v(self: Self, vertex: vec3) -> None:
        self.texture_vs.append(vertex)

    # # def add_unknown(self: Self, string: str) -> None:
    # #     self.unknowns.append(string)

    def add_vertex(self: Self, vertex: vec3) -> None:
        self.vertices.append(vertex)

    def add_v_normal(self: Self, vertex: vec3) -> None:
        self.v_normals.append(vertex)

    def verify(self: Self) -> None:
        if not __debug__:
            logger.warning("Assertions disabled; obj file cannot be verified")
        for comment in self.comments:
            match comment.split(maxsplit=1):
                case (count, "faces"):
                    icount = int(count)
                    assert (have := len(self.faces)) == icount, f"Incorrect number of faces ({have=} expect={icount})"
                    logger.debug("Faces count confirmed (%d)", icount)
                case (count, "texture vertices") | (count, "coords texture"):  # Seems to be an alias?
                    icount = int(count)
                    assert (have := len(self.texture_vs) - 1) == icount, (
                        f"Incorrect number of texture vertices ({have=} expect={icount})"
                    )
                    logger.debug("Texture vertex count confirmed (%d)", icount)
                case (count, "vertex normals"):
                    icount = int(count)
                    assert (have := len(self.v_normals) - 1) == icount, (
                        f"Incorrect number of vertex normals ({have=} expect={icount})"
                    )
                    logger.debug("Vertex normal count confirmed (%d)", icount)
                case (count, "vertices"):  # We insert a blank one
                    icount = int(count)
                    assert (have := len(self.vertices) - 1) == icount, (
                        f"Incorrect number of vertices ({have=} expect={icount})"
                    )
                    logger.debug("Vertex count confirmed (%d)", icount)
                case _:
                    logger.info("Could not parse comment: %s", comment)

    def normal(self: Self, iface: int, nthvert: int) -> vec3:
        """Helper ported from C++"""
        face: Final = self.faces[iface]
        n_vertex: Final = face[nthvert].normal
        return self.v_normals[n_vertex]

    def vert(self: Self, iface: int, nthvert: int) -> vec3:
        """Helper ported from C++"""
        face: Final = self.faces[iface]
        vertex: Final = face[nthvert].vertex
        return self.vertices[vertex]

    def vert_texture(self: Self, iface: int, nthvert: int) -> vec2:
        face: Final = self.faces[iface]
        vertex: Final = face[nthvert].texture
        return self.texture_vs[vertex].xy

    @staticmethod
    def from_file(infile: str | Path) -> Model:
        res = Model(_guard=_SENTINEL)
        current_group = None
        current_smooth = None
        with open(infile, encoding="utf-8") as ifile:
            for line_no, line in enumerate(ifile, 1):
                match line:
                    case _ if line.strip() == "":
                        # Blank line
                        pass
                    case _ if line.startswith("#"):
                        res.add_comment(line.rstrip())
                    case _ if m := GROUP_RE.match(line):
                        # A blank seems to clear it (assumed)
                        if m[1].strip() == "":
                            current_group = None
                            current_smooth = None
                        else:
                            current_group = m[1]
                    case _ if m := SMOOTH_RE.match(line):
                        current_smooth = None if m[1] == "off" else int(m[1])
                    case _ if m := FACE_RE.match(line):
                        f0 = Model.FaceEntry(
                            vertex=int(m["vertex_0"]), texture=int(m["texture_0"]), normal=int(m["normal_0"])
                        )
                        f1 = Model.FaceEntry(
                            vertex=int(m["vertex_1"]), texture=int(m["texture_1"]), normal=int(m["normal_1"])
                        )
                        f2 = Model.FaceEntry(
                            vertex=int(m["vertex_2"]), texture=int(m["texture_2"]), normal=int(m["normal_2"])
                        )
                        res.add_face(Model.Face((f0, f1, f2), group=current_group, smooth=current_smooth))
                    case _ if m := TEXTURE_V_RE.match(line):
                        res.add_texture_v(vec3(float(m["x"]), float(m["y"]), float(m["z"])))
                    case _ if m := TEXTURE_V_RE_2.match(line):
                        res.add_texture_v(vec3(float(m["x"]), float(m["y"]), 0))
                    case _ if m := VERTEX_RE.match(line):
                        res.add_vertex(vec3(float(m["x"]), float(m["y"]), float(m["z"])))
                    case _ if m := VERTEX_RE_6.match(line):
                        if m["unk4"] == m["unk5"] == m["unk6"] == "0.752941":
                            res.add_vertex(vec3(float(m["x"]), float(m["y"]), float(m["z"])))
                        else:
                            logger.debug("Unknown 6-entry vertex where unknown values didn't match: %s", line.rstrip())
                    case _ if m := V_NORMAL_RE.match(line):
                        res.add_v_normal(vec3(float(m["x"]), float(m["y"]), float(m["z"])))
                    case _:
                        logger.debug("Could not interpret line %d: %s", line_no, line.rstrip())
                        # res.add_unknown(line.rstrip())
                        raise ValueError(line.rstrip())
        base_file: Final = str(infile)[:-4]
        for support_file in SUPPORT_FILES:
            if (sfile := Path(f"{base_file}_{support_file}.tga")).is_file():
                res.ext[support_file] = TGAImage.read_tga_file(sfile)

        logger.debug("Read OBJ file %s: %s", Path(infile).name, res)
        res.verify()
        return res

    def __str__(self: Self) -> str:
        res = (
            f"Model({hex(id(self))}) with "
            f"{len(self.faces)} face(s), "
            f"{len(self.vertices) - 1} vert(ex|ices), "
            f"{len(self.v_normals)} vertex normal(s), "
            f"{len(self.texture_vs)} texture vert(ex|ices)"
        )
        for support_file in (s for s in SUPPORT_FILES if s in self.ext):
            res += f", {self.ext[support_file].width}x{self.ext[support_file].height} {support_file} image"
        return res
