from iltools_core.metadata_schema import DatasetMeta


def test_dataset_meta_basic():
    meta = DatasetMeta(
        name="demo",
        source="synthetic",
        version="0.0.1",
        citation="n/a",
        num_trajectories=3,
        keys=["observations/qpos", "observations/qvel", "actions/target"],
        trajectory_lengths=[100, 120, 80],
        dt=[0.02, 0.02, 0.02],
        joint_names=["root", "joint1"],
        body_names=["pelvis"],
        site_names=["foot_left", "foot_right"],
        metadata={"notes": "test"},
    )

    assert meta.name == "demo"
    assert meta.num_trajectories == 3
    assert isinstance(meta.trajectory_lengths, list)
    assert len(meta.keys) == 3
