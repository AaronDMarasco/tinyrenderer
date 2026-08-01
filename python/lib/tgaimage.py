from __future__ import annotations

from typing import TypeAlias, Final

from numpy import dtype
import numpy as np
# import numpy.typing as npt

# Some C++ cross-referencing for simplicty
uint8_t: TypeAlias = np.uint8
uint16_t: TypeAlias = np.uint16

TGAHeader: Final[dtype] = dtype(
    [
        ("idlength", uint8_t),
        ("colormaptype", uint8_t),
        ("datatypecode", uint8_t),
        ("colormaporigin", uint16_t),
        ("colormaplength", uint16_t),
        ("colormapdepth", uint8_t),
        ("x_origin", uint16_t),
        ("y_origin", uint16_t),
        ("width", uint16_t),
        ("height", uint16_t),
        ("bitsperpixel", uint8_t),
        ("imagedescriptor", uint8_t),
    ]
)


class TGAColor:
    # TODO: Still working. Binary dump representation. Tests!
    bgra_t: dtype = dtype(
        [
            ("b", uint8_t),
            ("g", uint8_t),
            ("r", uint8_t),
            ("a", uint8_t),
        ]
    )

    def __init__(self, b: int = 0, g: int = 0, r: int = 0, a: int = 0) -> None:
        self.bgra = np.array([(b, g, r, a)], dtype=TGAColor.bgra_t)
        self.bytespp: Final[uint8_t] = uint8_t(4)

    def __getitem__(self, idx: int) -> uint8_t:
        match idx:
            case 0:
                return self.bgra[0]["b"]
            case 1:
                return self.bgra[0]["g"]
            case 2:
                return self.bgra[0]["r"]
            case 3:
                return self.bgra[0]["a"]
            case _:
                raise ValueError("Only allow 0-3")

    def __setitem__(self, idx: int, val: uint8_t) -> None:
        match idx:
            case 0:
                self.bgra[0]["b"] = val
            case 1:
                self.bgra[0]["g"] = val
            case 2:
                self.bgra[0]["r"] = val
            case 3:
                self.bgra[0]["a"] = val
            case _:
                raise ValueError("Only allow 0-3")


if __name__ == "__main__":
    print(TGAHeader)
    for b in range(20):
        for g in range(20):
            for r in range(20):
                for a in range(20):
                    # print(TGAColor(b, g, r, a).bgra)
                    pass
    print(TGAColor(1, 2, 3, 4)[0])
    print(type(TGAColor(1, 2, 3, 4)[0]))
    print(TGAColor(1, 2, 3, 4)[2])
    print(type(TGAColor(1, 2, 3, 4)[2]))
