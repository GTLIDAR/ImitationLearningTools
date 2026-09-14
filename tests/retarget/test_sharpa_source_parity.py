"""Optional numerical parity against the adjacent video-to-data checkout."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest


def test_sharpa_pink_matches_source_fixture() -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("pink")
    scipy = pytest.importorskip("scipy.spatial.transform")
    workspace = Path(__file__).resolve().parents[3].parent
    source_package = (
        workspace / "video_to_data/robotic_grounding/source/robotic_grounding"
    )
    assets = source_package / "robotic_grounding/assets"
    loaded = assets / "human_motion_data/synthbox/synthbox_loaded"
    if not loaded.is_dir():
        pytest.skip("Adjacent video-to-data source fixture is not available.")

    # The source kinematics module imports visualization helpers at module
    # scope. They are not used by this numerical test.
    viser = types.ModuleType("viser")
    viser.ViserServer = object
    rate_module = types.ModuleType("loop_rate_limiters")

    class RateLimiter:
        def __init__(self, frequency: float, warn: bool = False) -> None:
            del warn
            self.period = 1.0 / frequency

    rate_module.RateLimiter = RateLimiter
    visualizer = types.ModuleType(
        "robotic_grounding.retarget.pinocchio_viser_visualizer"
    )
    visualizer.ViserVisualizer = object
    sys.modules.setdefault("viser", viser)
    sys.modules.setdefault("loop_rate_limiters", rate_module)
    sys.modules.setdefault(
        "robotic_grounding.retarget.pinocchio_viser_visualizer", visualizer
    )
    sys.path.insert(0, str(source_package))
    try:
        from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics
    finally:
        sys.path.remove(str(source_package))

    from iltools.datasets import ManoSharpaLoader
    from iltools.retarget.sharpa_pink import _SharpaHandSolver

    Rotation = scipy.Rotation
    row = ManoSharpaLoader(loaded).rows()[0]
    max_position = 0.0
    max_rotation = 0.0
    max_joint = 0.0
    offset = Rotation.from_quat((0.5, -0.5, 0.5, 0.5), scalar_first=True)
    for side in ("left", "right"):
        mjcf = assets / f"xmls/sharpawave/{side}_sharpawave.xml"
        source = SharpaHandKinematics(
            side=side,
            robot_asset_path=str(mjcf),
            source_model="mano",
        )
        port = _SharpaHandSolver(
            side=side,
            mjcf_path=mjcf,
            solver="daqp",
            max_iterations=200,
            frequency=200.0,
            convergence_threshold=1.0e-6,
        )
        source_seed = None
        port_seed = None
        joints = np.asarray(row[f"mano_{side}_joints"], dtype=np.float64)
        rotations = np.asarray(row[f"mano_{side}_joints_wxyz"], dtype=np.float64)
        assert len(joints) == len(rotations) == 155
        assert tuple(port.tasks) == tuple(source.frame_tasks)
        for frame_joints, frame_rotations in zip(joints, rotations, strict=True):
            if source_seed is None:
                source_seed = source.robot.q0.copy()
                source_seed[:3] = frame_joints[0]
                source_seed[3:7] = (
                    Rotation.from_quat(frame_rotations[0], scalar_first=True)
                    * offset.inv()
                ).as_quat()
            source_result = source.compute(
                frame_joints,
                frame_rotations,
                source_to_robot_scale=float(row["mano_to_robot_scale"]),
                qpos=source_seed,
            )
            source_seed = source_result["q"]
            port_seed, _, _, _ = port.solve(
                frame_joints,
                frame_rotations,
                scale=float(row["mano_to_robot_scale"]),
                q_seed=port_seed,
            )
            max_position = max(
                max_position,
                float(np.linalg.norm(port_seed[:3] - source_seed[:3])),
            )
            max_rotation = max(
                max_rotation,
                float(
                    (
                        Rotation.from_quat(port_seed[3:7]).inv()
                        * Rotation.from_quat(source_seed[3:7])
                    ).magnitude()
                ),
            )
            max_joint = max(
                max_joint,
                float(np.max(np.abs(port_seed[7:] - source_seed[7:]))),
            )

    assert max_position < 1.0e-3
    assert max_rotation < 1.0e-2
    assert max_joint < 1.0e-3
