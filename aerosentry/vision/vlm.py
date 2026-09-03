"""Aerial / thermal imagery analysis. A numpy hotspot detector always runs (deterministic, offline);
when a live provider is configured the frame is also sent to a Vision-Language Model and the two
views are fused. If the VLM call fails the heuristic result is returned - never an empty answer."""
import base64
import io
import os
from collections import deque

import numpy as np
from PIL import Image

from ..config import settings
from ..llm import LLMError, get_llm, resilient_chat
from ..observability import trace_event

VLM_PROMPT = ("You are an aerial inspection analyst for autonomous drone operations at industrial sites. "
              "Describe visible anomalies (thermal hotspots, damaged panels, intruders, vegetation encroachment, leaks), "
              "estimate severity (low/medium/high/critical) and recommend an operator action in under 120 words.")


def heuristic_hotspots(img: Image.Image, grid: int = 96, delta: float = 0.35, min_cells: int = 3) -> list[dict]:
    """Cells whose luminance exceeds the scene median by `delta` (robust to global brightness), then
    4-connected flood fill on a coarse grid -> bounding boxes with area fraction and intensity."""
    small = img.convert("L").resize((grid, grid), Image.BILINEAR)
    norm = np.asarray(small, dtype=np.float32) / 255.0
    mask = norm >= float(np.median(norm)) + delta
    seen, boxes = np.zeros_like(mask, dtype=bool), []
    for y in range(grid):
        for x in range(grid):
            if not mask[y, x] or seen[y, x]:
                continue
            q, cells = deque([(y, x)]), []
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < grid and 0 <= nx < grid and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(cells) < min_cells:
                continue
            ys, xs = zip(*cells)
            boxes.append({"x": round(min(xs) / grid, 3), "y": round(min(ys) / grid, 3),
                          "w": round((max(xs) - min(xs) + 1) / grid, 3), "h": round((max(ys) - min(ys) + 1) / grid, 3),
                          "area_frac": round(len(cells) / (grid * grid), 4),
                          "mean_intensity": round(float(np.mean([norm[c] for c in cells])), 3)})
    return sorted(boxes, key=lambda b: -b["area_frac"])


def severity_from(boxes: list[dict]) -> str:
    """Calibrated on the SOP classes: isolated hotspot -> medium, hot string / multiple -> high, large area -> critical."""
    if not boxes:
        return "low"
    total = sum(b["area_frac"] for b in boxes)
    if total > 0.03 or (len(boxes) >= 3 and total > 0.015):
        return "critical"
    if total > 0.005 or len(boxes) >= 2:
        return "high"
    return "medium"


def analyze(image_path: str, question: str = VLM_PROMPT) -> dict:
    path = image_path if os.path.isabs(image_path) else os.path.join(settings.data_dir, "..", image_path)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return {"error": f"image not found: {image_path}"}
    img = Image.open(path).convert("RGB")
    boxes = heuristic_hotspots(img)
    result = {"image": image_path, "detector": "heuristic-hotspot", "hotspots": boxes[:5],
              "hotspot_count": len(boxes), "severity": severity_from(boxes)}
    if settings.live:
        buf = io.BytesIO()
        img.resize((512, 512)).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        messages = [{"role": "system", "content": VLM_PROMPT},
                    {"role": "user", "content": [{"type": "text", "text": f"{question}\nHeuristic detector found {len(boxes)} hotspot(s): {boxes[:3]}"},
                                                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
        try:
            out = resilient_chat(get_llm(), messages, model=settings.vision_model)
            result["vlm_summary"], result["detector"] = out["content"], "vlm+heuristic"
        except LLMError as e:
            result["vlm_error"], result["fallback"] = str(e)[:160], "heuristic-only"
    trace_event("vision.analyze", {"image": image_path, "detector": result["detector"], "severity": result["severity"], "hotspots": len(boxes)})
    return result
