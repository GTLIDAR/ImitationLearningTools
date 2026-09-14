"""Contact recovery must measure geometry, never assume it."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from iltools.retarget.contact_recovery import (
    ContactRecoveryConfig,
    recover_contact_sequence,
)


# One slide joint drives a "fingertip" sphere toward a fixed "object" sphere.
# Both radii are 0.05, so the surfaces touch when their centres are 0.10 apart.
_MODEL_XML = """
<mujoco model="contact_recovery_fixture">
  <worldbody>
    <body name="hand" pos="0 0 0">
      <joint name="slide_x" type="slide" axis="1 0 0" range="-1 1"/>
      <geom name="right_tip" type="sphere" size="0.05"/>
    </body>
    <body name="object_body" mocap="true" pos="0.30 0 0">
      <geom name="object_geom" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

TOUCH_SEPARATION = 0.10


def _fixture():
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
    return model, data, [int(model.jnt_qposadr[joint])]


def _recover(centre_gaps, *, activity=None, config=None, use_geom_ids=False):
    """Run recovery with the tip placed at each requested centre gap."""

    model, data, addresses = _fixture()
    object_x = 0.30
    qpos = np.array([[object_x - gap] for gap in centre_gaps], dtype=np.float64)
    frames = len(centre_gaps)
    poses = np.tile(
        np.array([[object_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]), (frames, 1)
    ).reshape(frames, 1, 7)
    if activity is None:
        activity = np.ones((frames, 1), dtype=bool)
    return recover_contact_sequence(
        model,
        data,
        qpos=qpos,
        qpos_addresses=addresses,
        object_geom_names=[
            model.geom("object_geom").id if use_geom_ids else "object_geom"
        ],
        fingertip_geom_names={
            "right": [model.geom("right_tip").id if use_geom_ids else "right_tip"]
        },
        contact_link_names={"right": ["r_index_finger_distal"]},
        source_active=np.asarray(activity, dtype=bool),
        object_mocap_poses=poses,
        object_mocap_body_names=["object_body"],
        config=config,
    )


def test_a_touching_fingertip_yields_measured_witness_points() -> None:
    sequence, report = _recover([TOUCH_SEPARATION])

    assert bool(sequence.active[0, 0, 0])
    assert report.recovered_contacts == 1
    assert report.recovery_rate == pytest.approx(1.0)

    # Surfaces meet midway between the centres, at x = 0.25.
    link_point = sequence.link_positions_w[0, 0, 0]
    object_point = sequence.object_positions_w[0, 0, 0]
    assert link_point == pytest.approx([0.25, 0.0, 0.0], abs=1e-6)
    assert object_point == pytest.approx([0.25, 0.0, 0.0], abs=1e-6)


def test_contact_recovery_accepts_unnamed_geom_ids() -> None:
    sequence, report = _recover([TOUCH_SEPARATION], use_geom_ids=True)

    assert bool(sequence.active[0, 0, 0])
    assert report.recovered_contacts == 1


def test_normals_are_unit_and_oppositely_directed() -> None:
    # Sit just outside touching so the witness segment defines a direction.
    sequence, _ = _recover([TOUCH_SEPARATION + 0.001])

    link_normal = sequence.link_normals_w[0, 0, 0]
    object_normal = sequence.object_normals_w[0, 0, 0]
    assert np.linalg.norm(link_normal) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(object_normal) == pytest.approx(1.0, abs=1e-5)
    # The link normal points from the link toward the object, here +x.
    assert link_normal == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)
    assert object_normal == pytest.approx(-link_normal, abs=1e-6)


def test_a_separated_fingertip_stays_inactive_even_when_the_source_says_active() -> (
    None
):
    """A binary flag is not evidence of contact."""

    sequence, report = _recover([TOUCH_SEPARATION + 0.02])

    assert not bool(sequence.active[0, 0, 0])
    assert report.source_active_side_frames == 1
    assert report.recovered_side_frames == 0
    assert report.unreached_side_frames == 1
    assert report.unreached_frames["right"] == [0]
    assert report.recovery_rate == pytest.approx(0.0)
    # Inactive slots keep zero geometry and an invalid object index.
    assert sequence.link_positions_w[0, 0, 0] == pytest.approx([0.0, 0.0, 0.0])
    assert int(sequence.object_indices[0, 0, 0]) == -1


def test_an_inactive_source_frame_is_never_queried() -> None:
    # Touching geometry, but the source says the hand is not in contact.
    sequence, report = _recover([TOUCH_SEPARATION], activity=[[False]])

    assert not bool(sequence.active[0, 0, 0])
    assert report.source_active_side_frames == 0
    assert report.recovered_contacts == 0


def test_penetration_beyond_the_limit_stops_recovery() -> None:
    with pytest.raises(ValueError, match="penetrates object"):
        _recover([TOUCH_SEPARATION - 0.01])


def test_shallow_penetration_within_the_limit_is_still_a_contact() -> None:
    sequence, report = _recover(
        [TOUCH_SEPARATION - 0.001],
        config=ContactRecoveryConfig(max_penetration_m=0.002),
    )
    assert bool(sequence.active[0, 0, 0])
    assert report.minimum_distance_m < 0.0


def test_the_recovered_sequence_satisfies_the_training_contract() -> None:
    """Exactly the checks verify_training_qualification applies."""

    sequence, _ = _recover([TOUCH_SEPARATION + 0.001, TOUCH_SEPARATION + 0.001])
    sequence.validate(frame_count=2, object_count=1)

    assert np.any(sequence.active)
    selected = sequence.active
    for values in (sequence.link_positions_w, sequence.object_positions_w):
        active_values = values[selected]
        assert np.isfinite(active_values).all()
        assert np.all(np.linalg.norm(active_values, axis=-1) > 1.0e-8)
    for values in (sequence.link_normals_w, sequence.object_normals_w):
        norms = np.linalg.norm(values[selected], axis=-1)
        assert np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-3)
    indices = sequence.object_indices[selected]
    assert np.all(indices >= 0) and np.all(indices < 1)


def test_recovery_reports_a_mixed_trajectory_honestly() -> None:
    gaps = [
        TOUCH_SEPARATION,  # touching
        TOUCH_SEPARATION + 0.05,  # far
        TOUCH_SEPARATION + 0.001,  # touching within tolerance
        TOUCH_SEPARATION + 0.03,  # far
    ]
    sequence, report = _recover(gaps)

    assert list(sequence.active[:, 0, 0]) == [True, False, True, False]
    assert report.source_active_side_frames == 4
    assert report.recovered_side_frames == 2
    assert report.unreached_side_frames == 2
    assert report.unreached_frames["right"] == [1, 3]
    assert report.recovery_rate == pytest.approx(0.5)


def test_config_rejects_a_cutoff_below_the_contact_tolerance() -> None:
    with pytest.raises(ValueError, match="query_cutoff_m must exceed"):
        ContactRecoveryConfig(contact_tolerance_m=0.01, query_cutoff_m=0.005)


def test_report_serializes_for_the_provenance_record() -> None:
    _, report = _recover([TOUCH_SEPARATION, TOUCH_SEPARATION + 0.05])
    record = report.as_dict()

    assert record["recovered_side_frames"] == 1
    assert record["unreached_side_frames"] == 1
    assert record["recovery_rate"] == pytest.approx(0.5)
    assert record["unreached_frames"]["right"] == [1]
