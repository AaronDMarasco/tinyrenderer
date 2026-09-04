from __future__ import annotations

import logging
from types import ModuleType

from .tgaimage import TGAImage

plt: ModuleType | None

try:
    err: ImportError | None = None
    logging.getLogger("matplotlib").setLevel("WARNING")
    logging.getLogger("PIL").setLevel("WARNING")
    import matplotlib.pyplot as plt
except ImportError as e:
    err = e
    plt = None  # Test script wants it

logger = logging.getLogger(__name__)


def plot(framebuffer: TGAImage, plot: bool = True, *, _test_mode: bool = False) -> None:
    if not plot:
        return
    if err is not None:
        logger.warning(
            "Could not plot %dx%dx%d image - matplotlib failed import: %s",
            framebuffer.width,
            framebuffer.height,
            framebuffer.bpp,
            err,
        )
        return
    assert plt is not None  # Keep typing happy
    plt.imshow(framebuffer, origin="lower")
    if not _test_mode:
        plt.show()
