"""Minimal PNG writer (RGB8) using zlib — no PIL dependency."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
import numpy as np


def write_png(path, rgb: np.ndarray) -> Path:
    path = Path(path)
    img = np.clip(rgb, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    return path
