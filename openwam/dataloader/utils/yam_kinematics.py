"""YAM bimanual forward kinematics for the ABC-130k LeRobot export.

ABC stores 14-D joint vectors ``[L_joint1..6, L_grip, R_joint1..6, R_grip]``
(radians, gripper aperture 0 = closed / 1 = open). OpenWAM-α consumes the
bimanual EEF20 layout ``[L_pos, L_rot6d, L_grip, R_pos, R_rot6d, R_grip]``, so
this module maps joints to the tool pose of each arm.

Frame contract (validated against the ``RobotState.pose`` field of raw ABC
MCAP episodes, 0.000 mm / 0.000 deg residual):

* The recorded tool point is the MJCF ``<side>_tcp_site`` (the link_6 flange)
  translated ``TOOL_OFFSET_Z`` along the site's local z axis; orientation is the
  site orientation.
* Poses are expressed in each arm's own base frame (the ``<side>_arm`` body),
  the bimanual convention OpenWAM uses for RoboDojo. This also removes the
  1 cm arm-spacing difference between the MJCF and the recorded stations.

The chain constants are read from the MJCF once (MuJoCo is imported lazily in
:meth:`YamBimanualFK.from_mjcf`); evaluation is batched NumPy with no MuJoCo
call per row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

JOINT14_DIM = 14
EEF20_DIM = 20
ARM_JOINT_SLICES = {"left": slice(0, 6), "right": slice(7, 13)}
GRIPPER_INDICES = {"left": 6, "right": 13}
SIDES = ("left", "right")

# Distance from tcp_site (flange) to the recorded tool point, along site z.
TOOL_OFFSET_Z = 0.1347


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle_matrices(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Batched Rodrigues: unit ``axis`` (3,), ``angle`` (N,) -> (N, 3, 3)."""
    k = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
    )
    s = np.sin(angle)[:, None, None]
    c = np.cos(angle)[:, None, None]
    return np.eye(3)[None] + s * k[None] + (1.0 - c) * (k @ k)[None]


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two columns, ``[col0, col1]`` (OpenWAM rot6d)."""
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


@dataclass(frozen=True)
class _ArmChain:
    """Per-arm chain from the ``<side>_arm`` base body to the tool point."""

    body_pos: np.ndarray  # (6, 3) link_i offset in its parent frame
    body_rot: np.ndarray  # (6, 3, 3) link_i orientation in its parent frame
    joint_pos: np.ndarray  # (6, 3) hinge anchor in the link frame
    joint_axis: np.ndarray  # (6, 3) unit hinge axis in the link frame
    joint_ref: np.ndarray  # (6,) qpos0 (MuJoCo hinge angle = qpos - qpos0)
    tool_pos: np.ndarray  # (3,) tool point in the link_6 frame
    tool_rot: np.ndarray  # (3, 3) tool orientation in the link_6 frame

    def forward(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``joints`` (N, 6) -> base-frame tool position (N, 3) and rotation (N, 3, 3)."""
        n = joints.shape[0]
        pos = np.zeros((n, 3))
        rot = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        for i in range(6):
            pos = pos + rot @ self.body_pos[i]
            rot = rot @ self.body_rot[i]
            hinge = _axis_angle_matrices(self.joint_axis[i], joints[:, i] - self.joint_ref[i])
            # Rotation about the anchor: p' = p + R (a - H a), R' = R H.
            anchor = self.joint_pos[i]
            pos = pos + np.einsum("nij,j->ni", rot, anchor) - np.einsum("nij,njk,k->ni", rot, hinge, anchor)
            rot = rot @ hinge
        pos = pos + rot @ self.tool_pos
        rot = rot @ self.tool_rot
        return pos, rot


class YamBimanualFK:
    """Joint14 -> EEF20 for the YAM bimanual station (see module docstring)."""

    def __init__(self, chains: Dict[str, _ArmChain]):
        self._chains = chains

    @classmethod
    def from_mjcf(cls, mjcf_path: str) -> "YamBimanualFK":
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(mjcf_path))

        def _id(kind, name: str) -> int:
            idx = mujoco.mj_name2id(model, kind, name)
            if idx < 0:
                raise KeyError(f"{mjcf_path}: missing {kind.name} {name!r}")
            return idx

        chains = {}
        for side in SIDES:
            base = _id(mujoco.mjtObj.mjOBJ_BODY, f"{side}_arm")
            links = [_id(mujoco.mjtObj.mjOBJ_BODY, f"{side}_link_{i}") for i in range(1, 7)]
            joints = [_id(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{i}") for i in range(1, 7)]
            site = _id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp_site")
            parents = [base] + links[:-1]
            for link, parent, joint in zip(links, parents, joints):
                if model.body_parentid[link] != parent:
                    raise ValueError(f"{mjcf_path}: {side} chain is not base -> link_1 -> ... -> link_6")
                if model.jnt_bodyid[joint] != link or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"{mjcf_path}: {side} joint {joint} is not a hinge on its link")
            if model.site_bodyid[site] != links[-1]:
                raise ValueError(f"{mjcf_path}: {side}_tcp_site is not attached to {side}_link_6")
            site_rot = _quat_wxyz_to_matrix(model.site_quat[site])
            chains[side] = _ArmChain(
                body_pos=np.array([model.body_pos[b] for b in links], dtype=np.float64),
                body_rot=np.array([_quat_wxyz_to_matrix(model.body_quat[b]) for b in links]),
                joint_pos=np.array([model.jnt_pos[j] for j in joints], dtype=np.float64),
                joint_axis=np.array(
                    [model.jnt_axis[j] / np.linalg.norm(model.jnt_axis[j]) for j in joints], dtype=np.float64
                ),
                joint_ref=np.array([model.qpos0[model.jnt_qposadr[j]] for j in joints], dtype=np.float64),
                tool_pos=np.asarray(model.site_pos[site], dtype=np.float64) + site_rot @ [0.0, 0.0, TOOL_OFFSET_Z],
                tool_rot=site_rot,
            )
        return cls(chains)

    def arm_pose(self, side: str, joints6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Base-frame tool pose of one arm: (N, 6) joints -> (N, 3) pos, (N, 3, 3) rot."""
        return self._chains[side].forward(np.asarray(joints6, dtype=np.float64))

    def joint14_to_eef20(self, joint14: np.ndarray) -> np.ndarray:
        """(N, 14) ABC joint/gripper vector -> (N, 20) EEF20 (gripper passed through)."""
        joint14 = np.asarray(joint14, dtype=np.float64)
        if joint14.ndim != 2 or joint14.shape[1] != JOINT14_DIM:
            raise ValueError(f"expected (N, {JOINT14_DIM}) joint vector, got {joint14.shape}")
        parts = []
        for side in SIDES:
            pos, rot = self.arm_pose(side, joint14[:, ARM_JOINT_SLICES[side]])
            grip = joint14[:, GRIPPER_INDICES[side] : GRIPPER_INDICES[side] + 1]
            parts += [pos, matrix_to_rot6d(rot), grip]
        return np.concatenate(parts, axis=-1).astype(np.float32)


__all__ = [
    "EEF20_DIM",
    "JOINT14_DIM",
    "TOOL_OFFSET_Z",
    "YamBimanualFK",
    "matrix_to_rot6d",
]
