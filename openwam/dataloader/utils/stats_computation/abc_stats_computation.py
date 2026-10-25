"""EEF20 normalization statistics for the ABC-130k LeRobot v3 export.

Scans every ``data/**/*.parquet`` file, maps the 14-D ``action`` /
``observation.state`` joint vectors through YAM forward kinematics, and writes
separate ``eef`` (action) and ``eef_state`` (proprio) blocks in the schema the
ABC reader and the deploy normalizer consume.

* min / max / mean / std are exact over all rows (streamed per file).
* q01 / q10 / q50 / q90 / q99 come from a deterministic every-``--quantile-stride``
  row subsample (the corpus is ~382M rows).
* rot6d dims are pinned to identity; gripper dims (9, 19) are pinned to the
  physical aperture range min=0 / max=1 so min-max maps closed -> -1 and
  open -> +1 exactly for both action and state (measured state slightly
  overshoots [0, 1] and is clipped by the normalizer).

CLI:
    python -m openwam.dataloader.utils.stats_computation.abc_stats_computation \\
        --dataset-dir /path/to/abc_130k_v3_train \\
        --fk-mjcf-path /path/to/yam_mujoco/put_bottle.xml \\
        --output /path/to/abc_eef20_normalization_stats.npy --workers 32
"""

from __future__ import annotations

import argparse
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.abc import ACTION_STATS_KEY, GRIPPER_CONVENTION, POSE_FRAME, STATE_STATS_KEY
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity
from openwam.dataloader.utils.yam_kinematics import EEF20_DIM, JOINT14_DIM, TOOL_OFFSET_Z, YamBimanualFK

COLUMNS = {ACTION_STATS_KEY: "action", STATE_STATS_KEY: "observation.state"}
GRIPPER_DIMS = (9, 19)
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)

_FK: YamBimanualFK | None = None
_STRIDE = 1


def _init_worker(mjcf_path: str, stride: int) -> None:
    global _FK, _STRIDE
    _FK = YamBimanualFK.from_mjcf(mjcf_path)
    _STRIDE = stride


def _joint_matrix(table, column: str) -> np.ndarray:
    values = table.column(column).combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return values.reshape(-1, JOINT14_DIM).astype(np.float64)


def _scan_file(path: str) -> dict:
    try:
        table = pq.read_table(path, columns=list(COLUMNS.values()))
    except pa.ArrowInvalid as exc:
        # Some pyarrow versions read dotted LeRobot names as nested-field paths.
        if "Dot path" not in str(exc):
            raise
        table = pq.read_table(path)
    out = {}
    for key, column in COLUMNS.items():
        joints = _joint_matrix(table, column)
        if not np.isfinite(joints).all():
            raise ValueError(f"{path}:{column} contains NaN or infinity")
        eef = _FK.joint14_to_eef20(joints).astype(np.float64)
        out[key] = {
            "count": eef.shape[0],
            "sum": eef.sum(0),
            "sumsq": np.square(eef).sum(0),
            "min": eef.min(0),
            "max": eef.max(0),
            "sample": eef[::_STRIDE].astype(np.float32),
        }
    return out


def _finalize(parts: list[dict]) -> dict:
    count = sum(p["count"] for p in parts)
    total = np.sum([p["sum"] for p in parts], axis=0)
    total_sq = np.sum([p["sumsq"] for p in parts], axis=0)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0))
    sample = np.concatenate([p["sample"] for p in parts], axis=0)
    quantiles = np.quantile(sample, QUANTILES, axis=0)
    stats = {
        "min": np.min([p["min"] for p in parts], axis=0).tolist(),
        "max": np.max([p["max"] for p in parts], axis=0).tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [int(count)],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }
    # Measured gripper extremes are kept for reference before pinning.
    stats["gripper_observed_min"] = [stats["min"][d] for d in GRIPPER_DIMS]
    stats["gripper_observed_max"] = [stats["max"][d] for d in GRIPPER_DIMS]
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    for d in GRIPPER_DIMS:
        stats["min"][d], stats["max"][d] = 0.0, 1.0
        stats["q01"][d], stats["q99"][d] = 0.0, 1.0
    stats["num_timesteps"] = int(count)
    stats["quantile_sample_rows"] = int(sample.shape[0])
    return stats


def compute_abc_stats(dataset_dir: str | Path, fk_mjcf_path: str, *, workers: int, quantile_stride: int) -> dict:
    files = sorted(str(p) for p in (Path(dataset_dir) / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {Path(dataset_dir) / 'data'}")
    parts = {key: [] for key in COLUMNS}
    with Pool(workers, initializer=_init_worker, initargs=(fk_mjcf_path, quantile_stride)) as pool:
        for i, result in enumerate(pool.imap_unordered(_scan_file, files, chunksize=1), 1):
            for key in COLUMNS:
                parts[key].append(result[key])
            if i % 25 == 0 or i == len(files):
                print(f"  scanned {i}/{len(files)} files", flush=True)
    payload = {}
    for key, semantics in ((ACTION_STATS_KEY, "commanded tool pose"), (STATE_STATS_KEY, "measured tool pose")):
        stats = _finalize(parts[key])
        stats.update(
            {
                "pool": "action_only" if key == ACTION_STATS_KEY else "state_only",
                "gripper_convention": GRIPPER_CONVENTION,
                "pose_frame": POSE_FRAME,
                "representation": "yam_fk_eef20",
                "semantics": semantics,
                "fk_mjcf_path": str(fk_mjcf_path),
                "tool_offset_z": TOOL_OFFSET_Z,
                "dataset_dir": str(dataset_dir),
            }
        )
        payload[key] = stats
    return payload


def build_and_save_abc_stats(
    dataset_dir: str | Path,
    fk_mjcf_path: str,
    output: str | Path,
    *,
    workers: int = 8,
    quantile_stride: int = 100,
) -> Path:
    """Compute and atomically write the stats payload; returns the output path."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = compute_abc_stats(dataset_dir, fk_mjcf_path, workers=workers, quantile_stride=quantile_stride)
    fd, tmp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, payload, allow_pickle=True)
        os.replace(tmp_name, out_path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, help="ABC LeRobot v3 dataset root (contains meta/info.json)")
    parser.add_argument("--fk-mjcf-path", required=True, help="YAM bimanual MJCF (put_bottle.xml)")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quantile-stride", type=int, default=100, help="Keep every Nth row for quantiles")
    args = parser.parse_args()
    out_path = build_and_save_abc_stats(
        args.dataset_dir,
        args.fk_mjcf_path,
        args.output,
        workers=args.workers,
        quantile_stride=args.quantile_stride,
    )
    payload = np.load(out_path, allow_pickle=True).item()
    print(f"wrote {out_path}")
    for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
        block = payload[key]
        print(f"  {key}: num_timesteps={block['num_timesteps']}")
        for name in ("min", "max", "mean", "std"):
            print(f"    {name:4s} {np.round(np.asarray(block[name], dtype=np.float64), 4).tolist()}")
    assert len(payload[ACTION_STATS_KEY]["min"]) == EEF20_DIM


if __name__ == "__main__":
    main()
