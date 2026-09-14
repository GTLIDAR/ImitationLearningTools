"""MANO-to-Sharpa retargeting with the video-to-data Pink task layout.

This is a small dependency-isolated port of the Apache-2.0 video-to-data hand
kinematics code.  It uses the same eleven frame tasks, DAQP solver, sequential
warm start, 200 Hz integration rate, and convergence rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from iltools.core import sha256_file


MANO_JOINT_NAMES = (
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb4",
    "index1",
    "index2",
    "index3",
    "index4",
    "middle1",
    "middle2",
    "middle3",
    "middle4",
    "ring1",
    "ring2",
    "ring3",
    "ring4",
    "pinky1",
    "pinky2",
    "pinky3",
    "pinky4",
)

SHARPA_TASKS: dict[str, tuple[str, float, float]] = {
    ".*_hand_C_MC": ("wrist", 0.2, 0.2),
    ".*_thumb_MCP_VL_site": ("thumb2", 0.1, 0.0),
    ".*_thumb_tip_site": ("thumb4", 1.0, 0.05),
    ".*_index_MP_site": ("index1", 0.1, 0.0),
    ".*_index_tip_site": ("index4", 1.0, 0.1),
    ".*_middle_MP_site": ("middle1", 0.1, 0.0),
    ".*_middle_tip_site": ("middle4", 1.0, 0.1),
    ".*_ring_MP_site": ("ring1", 0.1, 0.0),
    ".*_ring_tip_site": ("ring4", 1.0, 0.1),
    ".*_pinky_MP_site": ("pinky1", 0.1, 0.0),
    ".*_pinky_tip_site": ("pinky4", 0.5, 0.1),
}


def _imports() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import pinocchio as pin
        import pink
        from pink import solve_ik
        from pink.limits import ConfigurationLimit, VelocityLimit
        from pink.tasks import FrameTask
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "SharpaPinkRetargeter requires the optional ILTools 'sharpa' extra."
        ) from exc
    return pin, pink, solve_ik, ConfigurationLimit, VelocityLimit, (FrameTask, Rotation)


@dataclass(frozen=True, slots=True)
class SharpaRetargetResult:
    """One retargeted hand trajectory in source-compatible layout."""

    joint_names: tuple[str, ...]
    frame_names: tuple[str, ...]
    frame_task_names: tuple[str, ...]
    wrist_position: np.ndarray
    wrist_wxyz: np.ndarray
    finger_joints: np.ndarray
    frames: np.ndarray
    frame_task_errors: np.ndarray
    optimization_iterations: np.ndarray


class _SharpaHandSolver:
    def __init__(
        self,
        *,
        side: Literal["left", "right"],
        mjcf_path: str | Path,
        solver: str,
        max_iterations: int,
        frequency: float,
        convergence_threshold: float,
    ) -> None:
        pin, pink, solve_ik, ConfigurationLimit, VelocityLimit, task_imports = (
            _imports()
        )
        FrameTask, Rotation = task_imports
        self.pin = pin
        self.solve_ik = solve_ik
        self.Rotation = Rotation
        self.side = side
        self.solver = solver
        self.max_iterations = int(max_iterations)
        self.dt = 1.0 / float(frequency)
        self.convergence_threshold = float(convergence_threshold)
        self.robot = pin.RobotWrapper.BuildFromMJCF(
            filename=str(Path(mjcf_path).expanduser().resolve()),
            root_joint=pin.JointModelFreeFlyer(),
        )
        self.configuration = pink.Configuration(
            self.robot.model, self.robot.data, self.robot.q0
        )
        self.limits = [
            ConfigurationLimit(self.robot.model),
            VelocityLimit(self.robot.model),
        ]
        self.joint_names = tuple(
            self.robot.model.names[index]
            for index in range(2, self.robot.model.nq - 7 + 2)
        )
        self.frame_names = tuple(frame.name for frame in self.robot.model.frames)
        self.task_specs = {
            pattern.replace(".*", side): spec for pattern, spec in SHARPA_TASKS.items()
        }
        self.tasks: dict[str, Any] = {}
        for frame_name, (_, position_cost, orientation_cost) in self.task_specs.items():
            task = FrameTask(
                frame_name,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                lm_damping=1.0,
            )
            task.set_target_from_configuration(self.configuration)
            self.tasks[frame_name] = task
        self.wrist_correction = (
            Rotation.from_quat((0.5, -0.5, 0.5, 0.5), scalar_first=True)
            .inv()
            .as_matrix()
        )

    def _set_targets(self, joints: np.ndarray, wxyz: np.ndarray, scale: float) -> None:
        wrist_index = MANO_JOINT_NAMES.index("wrist")
        wrist_position = joints[wrist_index]
        for frame_name, (mano_name, _, _) in self.task_specs.items():
            source_index = MANO_JOINT_NAMES.index(mano_name)
            target_position = (
                wrist_position + (joints[source_index] - wrist_position) * scale
            )
            target_rotation = self.Rotation.from_quat(
                wxyz[source_index], scalar_first=True
            ).as_matrix()
            if frame_name == f"{self.side}_hand_C_MC":
                target_rotation = target_rotation @ self.wrist_correction
            target = self.tasks[frame_name].transform_target_to_world
            target.translation = target_position.copy()
            target.rotation = target_rotation.copy()

    def solve(
        self,
        joints: np.ndarray,
        wxyz: np.ndarray,
        *,
        scale: float,
        q_seed: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        self._set_targets(joints, wxyz, scale)
        if q_seed is None:
            q_seed = self.robot.q0.copy()
            q_seed[:3] = joints[0]
            wrist_rotation = self.Rotation.from_quat(
                wxyz[0], scalar_first=True
            ) * self.Rotation.from_matrix(self.wrist_correction)
            q_seed[3:7] = wrist_rotation.as_quat(scalar_first=False)
        self.configuration.q = q_seed.copy()
        previous = {name: float("inf") for name in self.tasks}
        converged = {name: False for name in self.tasks}
        iterations = 0
        for _ in range(self.max_iterations):
            velocity = self.solve_ik(
                configuration=self.configuration,
                tasks=list(self.tasks.values()),
                dt=self.dt,
                solver=self.solver,
                safety_break=False,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(velocity, self.dt)
            iterations += 1
            for name, task in self.tasks.items():
                error = float(
                    np.linalg.norm(
                        np.asarray(task.compute_error(self.configuration))[:3]
                    )
                )
                converged[name] = (
                    abs(error - previous[name]) < self.convergence_threshold
                )
                previous[name] = error
            if all(converged.values()):
                break
        frame_poses = []
        for frame_name in self.frame_names:
            transform = self.configuration.get_transform_frame_to_world(frame_name)
            quat = self.Rotation.from_matrix(transform.rotation).as_quat(
                scalar_first=True
            )
            frame_poses.append(np.concatenate((transform.translation, quat)))
        return (
            self.configuration.q.copy(),
            np.asarray(frame_poses, dtype=np.float32),
            np.asarray([previous[name] for name in self.tasks], dtype=np.float32),
            iterations,
        )


@dataclass(slots=True)
class SharpaPinkRetargeter:
    """Retarget a sequence of MANO joint poses to both Sharpa hands."""

    left_mjcf_path: str | Path
    right_mjcf_path: str | Path
    solver: str = "daqp"
    # The source dataset scripts build their solver through
    # ``setup_sharpa_kinematics``, whose default is 100 iterations.
    max_iterations: int = 100
    frequency: float = 200.0
    convergence_threshold: float = 1.0e-6

    def _retarget_side(
        self, row: Mapping[str, Any], side: Literal["left", "right"]
    ) -> SharpaRetargetResult:
        solver = _SharpaHandSolver(
            side=side,
            mjcf_path=self.left_mjcf_path if side == "left" else self.right_mjcf_path,
            solver=self.solver,
            max_iterations=self.max_iterations,
            frequency=self.frequency,
            convergence_threshold=self.convergence_threshold,
        )
        joints = np.asarray(row[f"mano_{side}_joints"], dtype=np.float64)
        wxyz = np.asarray(row[f"mano_{side}_joints_wxyz"], dtype=np.float64)
        if joints.ndim != 3 or joints.shape[0] < 2 or joints.shape[1:] != (21, 3):
            raise ValueError(
                f"Invalid MANO {side} joint trajectory shape: expected "
                f"[frames>=2, 21, 3], got {joints.shape}."
            )
        if wxyz.shape != (joints.shape[0], 21, 4):
            raise ValueError(
                f"Invalid MANO {side} quaternion trajectory shape: expected "
                f"[{joints.shape[0]}, 21, 4], got {wxyz.shape}."
            )
        outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        seed = None
        # A freshly loaded row can carry a null scale: the released pipeline
        # supplies it per dataset on the retarget script's command line.
        raw_scale = row.get("mano_to_robot_scale")
        scale = 1.0 if raw_scale is None else float(raw_scale)
        for frame_joints, frame_wxyz in zip(joints, wxyz, strict=True):
            output = solver.solve(frame_joints, frame_wxyz, scale=scale, q_seed=seed)
            seed = output[0]
            outputs.append(output)
        qpos = np.stack([item[0] for item in outputs])
        return SharpaRetargetResult(
            joint_names=solver.joint_names,
            frame_names=solver.frame_names,
            frame_task_names=tuple(solver.tasks),
            wrist_position=qpos[:, :3].astype(np.float32),
            wrist_wxyz=qpos[:, 3:7][:, [3, 0, 1, 2]].astype(np.float32),
            finger_joints=qpos[:, 7:].astype(np.float32),
            frames=np.stack([item[1] for item in outputs]),
            frame_task_errors=np.stack([item[2] for item in outputs]),
            optimization_iterations=np.asarray(
                [item[3] for item in outputs], dtype=np.int32
            ),
        )

    def retarget_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Return a processed-row dictionary accepted by ``ManoSharpaLoader``."""

        result = dict(row)
        for side in ("left", "right"):
            hand = self._retarget_side(row, side)
            result[f"{side}_robot_finger_joint_names"] = list(hand.joint_names)
            result[f"{side}_robot_frame_names"] = list(hand.frame_names)
            result[f"{side}_robot_frame_task_names"] = list(hand.frame_task_names)
            result[f"robot_{side}_wrist_position"] = hand.wrist_position.tolist()
            result[f"robot_{side}_wrist_wxyz"] = hand.wrist_wxyz.tolist()
            result[f"robot_{side}_finger_joints"] = hand.finger_joints.tolist()
            result[f"robot_{side}_frames"] = hand.frames.tolist()
            result[f"robot_{side}_frame_task_errors"] = hand.frame_task_errors.tolist()
            result[f"robot_{side}_num_optimization_iterations"] = (
                hand.optimization_iterations.tolist()
            )
        result["robot_name"] = "sharpa_wave"
        # Provenance so a downstream reference can say which solver produced
        # its Sharpa solution instead of trusting stored ``robot_*`` columns.
        result["retarget_source"] = "iltools_pink"
        result["retarget_solver"] = str(self.solver)
        result["retarget_max_iterations"] = int(self.max_iterations)
        result["retarget_frequency_hz"] = float(self.frequency)
        result["retarget_mjcf_sha256"] = {
            "left": sha256_file(Path(self.left_mjcf_path).expanduser().resolve()),
            "right": sha256_file(Path(self.right_mjcf_path).expanduser().resolve()),
        }
        return result


__all__ = [
    "MANO_JOINT_NAMES",
    "SHARPA_TASKS",
    "SharpaPinkRetargeter",
    "SharpaRetargetResult",
]
