"""Generate synthetic radiometric-style thermal frames of a solar array (clean vs anomalous) so the
vision pipeline and evals are reproducible without proprietary drone footage."""
import os
import sys

import numpy as np
from PIL import Image, ImageFilter

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "imagery")


def palette(v):
    """Ironbow-like mapping from 0..1 to RGB."""
    r = np.clip(1.6 * v - 0.1, 0, 1)
    g = np.clip(2.0 * v - 1.0, 0, 1) * 0.9
    b = np.clip(0.9 - 1.8 * abs(v - 0.25), 0, 1) * 0.8
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def array_frame(hot_cells, seed, size=512):
    rng = np.random.default_rng(seed)
    v = rng.normal(0.22, 0.02, (size, size)).astype(np.float32)  # ground
    rows, cols, ph, pw, gap = 6, 8, 52, 44, 12
    x0, y0 = (size - cols * (pw + gap)) // 2, (size - rows * (ph + gap)) // 2
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * (pw + gap), y0 + r * (ph + gap)
            v[y:y + ph, x:x + pw] = rng.normal(0.42, 0.015, (ph, pw))
    for (r, c, frac) in hot_cells:
        x, y = x0 + c * (pw + gap), y0 + r * (ph + gap)
        hy, hx = int(ph * frac), int(pw * frac)
        v[y:y + hy, x:x + hx] = rng.normal(0.95, 0.02, (hy, hx))
    return Image.fromarray(palette(np.clip(v, 0, 1))).filter(ImageFilter.GaussianBlur(0.8))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    array_frame([], 7).save(os.path.join(OUT, "solar_thermal_clean.png"))
    array_frame([(2, 5, 0.5)], 11).save(os.path.join(OUT, "solar_thermal_anomaly.png"))
    array_frame([(1, 2, 1.0), (1, 3, 1.0), (4, 6, 0.6)], 23).save(os.path.join(OUT, "solar_thermal_hot_string.png"))
    print("wrote 3 frames to", os.path.abspath(OUT))
    sys.exit(0)
