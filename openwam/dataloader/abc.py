"""ABC-130k (YAM bimanual) LeRobot v3 reader for OpenWAM-α fine-tuning.

The export stores 14-D joint vectors for both ``action`` (commanded) and
``observation.state`` (measured)::

    [L_joint1..6 (rad), L_grip, R_joint1..6 (rad), R_grip]   grip: 0 = closed, 1 = open

OpenWAM-α expects the bimanual EEF20 contract, so every window is mapped
through YAM forward kinematics (:mod:`openwam.dataloader.utils.yam_kinematics`)
into ``[L_pos, L_rot6d, L_grip, R_pos, R_rot6d, R_grip]`` with each arm in its
own base frame, then min-max normalized and scattered into the unified 80-D
slots ``0-9`` / ``34-43`` by the base reader.

The 224x224 videos are letterboxed; with ``letterbox_layouts_path`` set
(built by :mod:`openwam.dataloader.utils.abc_letterbox`) the black bars are
cropped per episode before frames are resized into the canvas slots.

Normalization stats are computed over the EEF20 representation (not the joint
stats in ``meta/stats.json``) by
:mod:`openwam.dataloader.utils.stats_computation.abc_stats_computation` and
read from ``normalization_stats_path``. The file holds separate ``eef``
(action) and ``eef_state`` (proprio) blocks; the deploy normalizer consumes the
same file, so the reader never writes into the (shared) dataset directory.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from openwam.dataloader.bases.lerobot_v3_reader import LeRobotV3Reader
from openwam.dataloader.utils.abc_letterbox import layout_crop_box, load_layout_table
from openwam.dataloader.utils.eef import EEF_DIM
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.yam_kinematics import JOINT14_DIM, YamBimanualFK

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=64)
def _open_parquet(path: str) -> Tuple[pq.ParquetFile, np.ndarray]:
    """Per-process cached ParquetFile plus the first row of every row group."""
    pf = pq.ParquetFile(path, memory_map=True)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)]
    return pf, np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


class _RowGroupTable:
    """Duck-typed stand-in for the base reader's whole-file table.

    ``slice(start, n)`` reads only the row groups overlapping the window
    (ABC shards hold ~380k rows in ~3k-row groups), instead of materializing
    the full 50 MB shard per sample.
    """

    def __init__(self, path: str, columns: Tuple[str, ...]):
        self._pf, self._starts = _open_parquet(path)
        self._columns = list(columns)

    def slice(self, start: int, length: int):
        first = int(np.searchsorted(self._starts, start, side="right") - 1)
        last = int(np.searchsorted(self._starts, start + length - 1, side="right") - 1)
        table = self._pf.read_row_groups(list(range(first, last + 1)), columns=self._columns)
        return table.slice(start - int(self._starts[first]), length)


ACTION_MODE = "eef"
ACTION_STATS_KEY = ACTION_MODE
STATE_STATS_KEY = f"{ACTION_MODE}_state"
GRIPPER_CONVENTION = "minus1_closed_plus1_open"
POSE_FRAME = "per_arm_base"


class ABCDataset(LeRobotV3Reader):
    """Single-bucket ABC-130k reader (joint14 -> EEF20 via YAM FK)."""

    DATASET_NAME: ClassVar[str] = "ABC"
    NEEDED_COLS: ClassVar[tuple] = ("observation.state", "action", "task_index")
    HEAD_CAMERA: ClassVar[Optional[str]] = "observation.images.top"
    LEFT_WRIST_CAMERA: ClassVar[Optional[str]] = "observation.images.left_wrist"
    RIGHT_WRIST_CAMERA: ClassVar[Optional[str]] = "observation.images.right_wrist"
    ACTION_DIM: ClassVar[int] = EEF_DIM
    PROMPT_SOURCE: ClassVar[str] = "task_index"
    DEPLOY_ACTION_MODE: ClassVar[Optional[str]] = ACTION_MODE
    DEFAULT_NORMALIZE_MODE: ClassVar[Optional[str]] = "min-max"
    CONFIG_KEYS: ClassVar[tuple] = LeRobotV3Reader.CONFIG_KEYS + (
        "normalization_stats_path",
        "fk_mjcf_path",
        "letterbox_layouts_path",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        fk_mjcf_path: str,
        normalization_stats_path: Optional[str] = None,
        letterbox_layouts_path: Optional[str] = None,
        **kwargs: Any,
    ):
        self._fk_mjcf_path = str(fk_mjcf_path)
        self._source_stats_path = str(normalization_stats_path) if normalization_stats_path else None
        self._state_normalization_stats: Optional[dict] = None
        self._fk = YamBimanualFK.from_mjcf(self._fk_mjcf_path)
        # episode_index -> letterbox layout code (all cameras of an episode share it); None = full frames.
        self._letterbox_layouts = load_layout_table(letterbox_layouts_path) if letterbox_layouts_path else None
        super().__init__(dataset_dir=dataset_dir, **kwargs)

    def _post_init(self, info: dict) -> None:
        features = info.get("features", {}) or {}
        for column in ("observation.state", "action"):
            shape = tuple(features.get(column, {}).get("shape", ()))
            if shape != (JOINT14_DIM,):
                raise ValueError(f"ABC {column} feature must have shape [{JOINT14_DIM}], got {shape}")
        names = features["action"].get("names") or []
        if len(names) == JOINT14_DIM and not (names[6].startswith("left_gripper") and names[13].startswith("right_gripper")):
            raise ValueError(f"ABC action gripper columns must sit at indices 6 / 13, got names={names}")
        if self._letterbox_layouts is not None:
            missing = set(self._eps_df["episode_index"].tolist()) - self._letterbox_layouts.keys()
            if missing:
                raise ValueError(
                    f"ABC letterbox layout table lacks {len(missing)} episodes of {self._dataset_dir} "
                    f"(e.g. {sorted(missing)[:5]}); rebuild it with python -m openwam.dataloader.utils.abc_letterbox"
                )

    def _load_data_table(self, chunk_idx: int, file_idx: int) -> _RowGroupTable:
        path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        return _RowGroupTable(str(path), self.NEEDED_COLS)

    def _camera_crop_box(self, camera: str, row):
        if self._letterbox_layouts is None:
            return None
        return layout_crop_box(self._letterbox_layouts[int(row["episode_index"])])

    def _load_stats(self, info: dict) -> Optional[dict]:
        if not self._normalize_mode or self._normalize_mode in ("none", "null"):
            return None
        if self._source_stats_path is None:
            raise ValueError(
                "ABC requires normalization_stats_path (EEF20 stats). Build it with "
                "python -m openwam.dataloader.utils.stats_computation.abc_stats_computation"
            )
        stats_path = Path(self._source_stats_path)
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"ABC normalization stats {stats_path} not found. Build them with "
                "python -m openwam.dataloader.utils.stats_computation.abc_stats_computation "
                f"--dataset-dir {self._dataset_dir} --fk-mjcf-path {self._fk_mjcf_path} --output {stats_path}"
            )
        raw = np.load(stats_path, allow_pickle=True).item()
        if ACTION_STATS_KEY not in raw or STATE_STATS_KEY not in raw:
            raise KeyError(f"{stats_path} must contain {ACTION_STATS_KEY!r} and {STATE_STATS_KEY!r} blocks")
        for key in (ACTION_STATS_KEY, STATE_STATS_KEY):
            block = raw[key]
            if block.get("gripper_convention") != GRIPPER_CONVENTION or block.get("pose_frame") != POSE_FRAME:
                raise ValueError(
                    f"{stats_path}:{key} declares gripper_convention={block.get('gripper_convention')!r}, "
                    f"pose_frame={block.get('pose_frame')!r}; expected {GRIPPER_CONVENTION!r}, {POSE_FRAME!r}"
                )
        action_stats = materialize_eef_stats(
            dict(raw[ACTION_STATS_KEY]),
            self._normalize_mode,
            dim=EEF_DIM,
            strict_minmax=True,
            source_hint=f"{stats_path}:{ACTION_STATS_KEY}",
            force_rot6d_identity=True,
        )
        self._state_normalization_stats = materialize_eef_stats(
            dict(raw[STATE_STATS_KEY]),
            self._normalize_mode,
            dim=EEF_DIM,
            strict_minmax=True,
            source_hint=f"{stats_path}:{STATE_STATS_KEY}",
            force_rot6d_identity=True,
        )
        self.normalization_stats_path = str(stats_path)
        return action_stats

    def _normalize_array(self, arr: np.ndarray, stats: Optional[dict]) -> np.ndarray:
        """Normalize pos / gripper dims; rot6d passes through exactly."""
        raw = np.asarray(arr, dtype=np.float32)
        output = np.array(apply_normalization(raw, stats, self._normalize_mode), dtype=np.float32, copy=True)
        output[..., ROT6D_DIMS_EEF20] = raw[..., ROT6D_DIMS_EEF20]
        return output

    def _eef20(self, win, column: str, n_rows: Optional[int] = None) -> np.ndarray:
        values = np.stack(win[column].values[:n_rows]).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"ABC {column} window contains NaN or infinity")
        return self._fk.joint14_to_eef20(values)

    def _action_20d(self, win) -> np.ndarray:
        return self._normalize_array(self._eef20(win, "action"), self._normalization_stats)

    def _proprio_20d(self, win) -> np.ndarray:
        return self._normalize_array(self._eef20(win, "observation.state", 1), self._state_normalization_stats)


__all__ = [
    "ABCDataset",
    "ACTION_MODE",
    "ACTION_STATS_KEY",
    "GRIPPER_CONVENTION",
    "POSE_FRAME",
    "STATE_STATS_KEY",
]
