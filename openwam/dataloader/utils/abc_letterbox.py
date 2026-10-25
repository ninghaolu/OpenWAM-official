"""Per-episode letterbox crops for the ABC-130k LeRobot v3 export.

The export pads every camera stream into a 224x224 square with black bars
above and below. The content aspect follows the recording station's camera
(4:3, 16:10, 16:9, or 5:4) and varies between episodes even inside one packed
mp4, while all three cameras of an episode share it. This module scans one
head-camera frame per episode, snaps the visible row extent to the nearest
canonical layout, and writes an ``episode_index -> layout`` table the ABC
reader uses to crop the bars before resizing frames into their canvas slots.

CLI:
    python -m openwam.dataloader.utils.abc_letterbox \\
        --dataset-dir /path/to/abc_130k_v3_train --output /path/to/abc_letterbox_layouts.npy --workers 32
"""

from __future__ import annotations

import argparse
import os
import tempfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

FRAME_SIZE = 224
HEAD_CAMERA = "observation.images.top"
# Visible content rows (first, last) inside the 224x224 square, per layout.
LAYOUT_ROWS: Dict[str, Tuple[int, int]] = {
    "4:3": (28, 195),
    "16:10": (42, 181),
    "16:9": (49, 174),
    "5:4": (22, 200),
}
LAYOUT_CODES = {name: code for code, name in enumerate(LAYOUT_ROWS)}
UNKNOWN_CODE = -1
# Rows trimmed inside each edge so compression bleed from the bars never enters the crop.
EDGE_INSET = 1
CONTENT_THRESHOLD = 24
SNAP_TOLERANCE = 4


def layout_crop_box(code: int) -> Optional[Tuple[int, int, int, int]]:
    """PIL ``(left, upper, right, lower)`` crop box for a layout code, or None (unknown -> full frame)."""
    if code == UNKNOWN_CODE:
        return None
    first, last = LAYOUT_ROWS[list(LAYOUT_ROWS)[code]]
    return (0, first + EDGE_INSET, FRAME_SIZE, last + 1 - EDGE_INSET)


def classify_rows(frames: np.ndarray) -> int:
    """Layout code for (N, H, W, 3) uint8 frames, or ``UNKNOWN_CODE`` if no layout is within tolerance."""
    rows = np.flatnonzero(frames.max(axis=(0, 2, 3)) > CONTENT_THRESHOLD)
    if rows.size == 0:
        return UNKNOWN_CODE
    first, last = int(rows.min()), int(rows.max())
    dist = {name: abs(r[0] - first) + abs(r[1] - last) for name, r in LAYOUT_ROWS.items()}
    best = min(dist, key=dist.get)
    return LAYOUT_CODES[best] if dist[best] <= SNAP_TOLERANCE else UNKNOWN_CODE


def _scan_episode(rec: tuple) -> tuple:
    from openwam.dataloader.utils.video_io import decode_video_frames

    episode_index, path, start, length = rec
    probes = [start + length // 2, start + length // 4, start + (3 * length) // 4]
    frames = []
    for i, frame_idx in enumerate(probes):
        frames.append(np.asarray(decode_video_frames(path, [frame_idx], FRAME_SIZE, FRAME_SIZE)[0]))
        code = classify_rows(np.stack(frames))
        if code != UNKNOWN_CODE:
            return episode_index, code, i + 1
    return episode_index, UNKNOWN_CODE, len(probes)


def scan_layouts(dataset_dir: str | Path, *, workers: int) -> dict:
    from openwam.dataloader.utils.lerobotv3 import load_episodes_parquet, parse_info_json

    dataset_dir = Path(dataset_dir)
    info = parse_info_json(dataset_dir)
    fps = float(info["fps"])
    eps = load_episodes_parquet(dataset_dir)
    cols = {c: i for i, c in enumerate(eps.columns)}
    recs = [
        (
            int(r[cols["episode_index"]]),
            str(
                dataset_dir
                / info["video_path"].format(
                    video_key=HEAD_CAMERA,
                    chunk_index=int(r[cols[f"videos/{HEAD_CAMERA}/chunk_index"]]),
                    file_index=int(r[cols[f"videos/{HEAD_CAMERA}/file_index"]]),
                )
            ),
            int(round(float(r[cols[f"videos/{HEAD_CAMERA}/from_timestamp"]]) * fps)),
            int(r[cols["length"]]),
        )
        for r in eps.itertuples(index=False, name=None)
    ]
    results = []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_episode, recs, chunksize=16), 1):
            results.append(res)
            if i % 5000 == 0 or i == len(recs):
                print(f"  scanned {i}/{len(recs)} episodes", flush=True)
    results.sort()
    return {
        "episode_index": np.array([r[0] for r in results], dtype=np.int64),
        "layout_code": np.array([r[1] for r in results], dtype=np.int8),
        "probes_used": np.array([r[2] for r in results], dtype=np.int8),
        "layouts": dict(LAYOUT_ROWS),
        "edge_inset": EDGE_INSET,
        "camera": HEAD_CAMERA,
        "dataset_dir": str(dataset_dir),
    }


def load_layout_table(path: str | Path) -> Dict[int, int]:
    payload = np.load(path, allow_pickle=True).item()
    if payload.get("layouts") != LAYOUT_ROWS or payload.get("edge_inset") != EDGE_INSET:
        raise ValueError(f"{path}: layout table was built with different canonical layouts; rebuild it")
    return dict(zip(payload["episode_index"].tolist(), payload["layout_code"].tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    payload = scan_layouts(args.dataset_dir, workers=args.workers)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    names = {code: name for name, code in LAYOUT_CODES.items()} | {UNKNOWN_CODE: "unknown"}
    counts = Counter(names[c] for c in payload["layout_code"].tolist())
    print(f"wrote {out}: {len(payload['episode_index'])} episodes, layouts {dict(counts)}")
    print(f"  probes used: {dict(Counter(payload['probes_used'].tolist()))}")


if __name__ == "__main__":
    main()
