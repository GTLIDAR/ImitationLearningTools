from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from iltools.core import (
    CollisionAssetDependency,
    CollisionClearanceQualification,
    ContactSequence,
    DexterousReference,
    ScenePhysics,
    TrainingQualification,
    build_urdf_collision_asset_dependencies,
    create_dexterous_reference_manifest,
    derive_object_twists_w,
    load_dexterous_reference_manifest,
    load_dexterous_reference_npz,
    load_dexterous_reference_set,
    save_dexterous_reference_npz,
    verify_training_qualification,
)


def _make_reference(*, sequence_id: str = "cube_pick") -> DexterousReference:
    frames = 3
    objects = np.zeros((frames, 1, 7), dtype=np.float32)
    objects[..., 0] = np.arange(frames)[:, None] * 0.01
    objects[..., 3] = 1.0
    active = np.zeros((frames, 2, 2), dtype=bool)
    active[:, 0, 0] = True
    link_normals = np.zeros((frames, 2, 2, 3), dtype=np.float32)
    object_normals = np.zeros_like(link_normals)
    link_normals[..., 2] = 1.0
    object_normals[..., 2] = -1.0
    object_indices = np.full((frames, 2, 2), -1, dtype=np.int32)
    object_indices[:, 0, 0] = 0
    contacts = ContactSequence(
        hand_sides=("left", "right"),
        link_names=np.asarray(
            [["l_index_distal", "l_thumb_distal"], ["r_index_distal", ""]]
        ),
        link_positions_w=np.zeros((frames, 2, 2, 3), dtype=np.float32),
        link_normals_w=link_normals,
        object_positions_w=np.zeros((frames, 2, 2, 3), dtype=np.float32),
        object_normals_w=object_normals,
        object_indices=object_indices,
        active=active,
    )
    wrists = np.zeros((frames, 7), dtype=np.float32)
    wrists[:, 3] = 1.0
    fingertips = wrists[:, None, :].copy()
    return DexterousReference(
        sequence_id=sequence_id,
        robot_name="vega_wuji",
        fps=50.0,
        joint_names=("arm", "finger"),
        qpos=np.asarray([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]]),
        qvel=None,
        fixed_root_pose_w=np.asarray([0.0, 0.0, 0.19, 1.0, 0.0, 0.0, 0.0]),
        left_wrist_pose_w=wrists,
        right_wrist_pose_w=wrists,
        left_wrist_frame_name="left_palm",
        right_wrist_frame_name="right_palm",
        object_names=("cube",),
        object_poses_w=objects,
        object_asset_paths=("assets/cube.usda",),
        object_scales=np.asarray([[1.5, 1.0, 0.5]]),
        object_radii=np.asarray([0.04]),
        left_hand_frame_names=("l_index_tip",),
        left_hand_frame_poses_w=fingertips,
        right_hand_frame_names=("r_index_tip",),
        right_hand_frame_poses_w=fingertips,
        support_surface_names=("table",),
        support_surface_asset_paths=("assets/table.usda",),
        support_surface_scales=np.asarray([[2.0, 2.0, 1.0]]),
        support_surface_poses_w=np.asarray([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        contacts=contacts,
        metadata={"source": "synthetic"},
    )


def _write_scene_assets(directory: Path) -> tuple[Path, Path]:
    asset_directory = directory / "assets"
    asset_directory.mkdir(parents=True, exist_ok=True)
    object_path = asset_directory / "cube.usda"
    support_path = asset_directory / "table.usda"
    object_path.write_text('#usda 1.0\ndef Cube "cube" {}\n', encoding="utf-8")
    support_path.write_text('#usda 1.0\ndef Xform "table" {}\n', encoding="utf-8")
    return object_path, support_path


def _write_urdf_object(directory: Path) -> tuple[Path, Path]:
    asset_directory = directory / "urdf_object"
    asset_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = asset_directory / "collision.obj"
    # The missing material is intentionally not part of collision physics.
    mesh_path.write_text(
        "mtllib missing_visual_material.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    urdf_path = asset_directory / "object.urdf"
    urdf_path.write_text(
        """<robot name="object">
  <link name="object">
    <visual><geometry><mesh filename="collision.obj"/></geometry></visual>
    <collision><geometry><mesh filename="collision.obj"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return urdf_path, mesh_path


def _set_urdf_object(
    reference: DexterousReference, directory: Path
) -> tuple[Path, Path]:
    urdf_path, mesh_path = _write_urdf_object(directory)
    reference.object_asset_paths = (str(urdf_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    )
    reference.collision_asset_dependencies = build_urdf_collision_asset_dependencies(
        urdf_path,
        asset_role="object",
        asset_index=0,
    )
    return urdf_path, mesh_path


def _write_hashed_reference(directory: Path, *, sequence_id: str) -> Path:
    object_path, support_path = _write_scene_assets(directory)
    reference = _make_reference(sequence_id=sequence_id)
    reference.object_asset_sha256 = (
        hashlib.sha256(object_path.read_bytes()).hexdigest(),
    )
    reference.support_surface_asset_sha256 = (
        hashlib.sha256(support_path.read_bytes()).hexdigest(),
    )
    return save_dexterous_reference_npz(
        reference,
        directory / f"{sequence_id}.npz",
    )


def _training_qualification(
    frame_count: int,
    *,
    runtime_qualified: bool = True,
    isaac_runtime_qualified: bool = True,
    inspection_only: bool = False,
) -> TrainingQualification:
    return TrainingQualification(
        runtime_qualified=runtime_qualified,
        isaac_runtime_qualified=isaac_runtime_qualified,
        inspection_only=inspection_only,
        contact_geometry_provenance="manual contact annotation batch 7",
        collision_clearance=CollisionClearanceQualification(
            qualified=True,
            method="signed-distance replay",
            scope="robot, object, and support geometry for every frame",
            provenance="Isaac Newton replay audit run 42",
            checked_frame_count=frame_count,
            minimum_signed_distance_m=-0.0001,
            penetration_tolerance_m=0.0002,
        ),
    )


def _make_training_reference(directory: Path) -> DexterousReference:
    object_path, support_path = _write_scene_assets(directory)
    reference = _make_reference()
    reference.object_asset_paths = (str(object_path),)
    reference.support_surface_asset_paths = (str(support_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(object_path.read_bytes()).hexdigest(),
    )
    reference.support_surface_asset_sha256 = (
        hashlib.sha256(support_path.read_bytes()).hexdigest(),
    )
    assert reference.contacts is not None
    reference.contacts.link_positions_w[reference.contacts.active] = (
        0.1,
        0.2,
        0.3,
    )
    reference.contacts.object_positions_w[reference.contacts.active] = (
        0.101,
        0.2,
        0.3,
    )
    reference.scene_physics = ScenePhysics(
        object_mass_kg=np.asarray([0.35]),
        object_center_of_mass_m=np.asarray([[0.0, 0.0, 0.01]]),
        object_diagonal_inertia_kg_m2=np.asarray([[0.001, 0.001, 0.001]]),
        object_static_friction=np.asarray([0.8]),
        object_dynamic_friction=np.asarray([0.6]),
        object_restitution=np.asarray([0.05]),
        support_static_friction=np.asarray([0.9]),
        support_dynamic_friction=np.asarray([0.7]),
        support_restitution=np.asarray([0.02]),
    )
    reference.scene_physics.validate(object_count=1, support_count=1)
    reference.training_qualification = _training_qualification(reference.frame_count)
    return reference


def test_dexterous_reference_npz_round_trip_keeps_runtime_keys(tmp_path) -> None:
    output = save_dexterous_reference_npz(_make_reference(), tmp_path / "cube_pick.npz")

    with np.load(output, allow_pickle=False) as archive:
        assert {
            "qpos",
            "qvel",
            "joint_names",
            "fps",
            "object_twists_w",
        }.issubset(archive.files)

    loaded = load_dexterous_reference_npz(output)
    assert loaded.sequence_id == "cube_pick"
    assert loaded.joint_names == ("arm", "finger")
    assert loaded.object_names == ("cube",)
    assert loaded.source_path == output
    assert loaded.object_asset_paths == (
        str((output.parent / "assets" / "cube.usda").resolve()),
    )
    assert loaded.support_surface_asset_paths == (
        str((output.parent / "assets" / "table.usda").resolve()),
    )
    np.testing.assert_allclose(loaded.object_radii, [0.04])
    np.testing.assert_allclose(loaded.object_scales, [[1.5, 1.0, 0.5]])
    np.testing.assert_allclose(loaded.support_surface_scales, [[2.0, 2.0, 1.0]])
    assert loaded.left_hand_frame_names == ("l_index_tip",)
    assert loaded.left_wrist_frame_name == "left_palm"
    assert loaded.right_wrist_frame_name == "right_palm"
    assert loaded.support_surface_names == ("table",)
    assert loaded.contacts is not None
    assert loaded.scene_physics is None
    assert loaded.training_qualification is None
    np.testing.assert_array_equal(loaded.contacts.active[:, 0, 0], True)
    np.testing.assert_allclose(loaded.qvel[0], [5.0, 10.0])
    np.testing.assert_allclose(loaded.object_twists_w[..., 0], 0.5)
    np.testing.assert_allclose(loaded.object_twists_w[..., 1:], 0.0)


def test_object_twist_derivation_handles_quaternion_signs_and_endpoints() -> None:
    angles = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
    poses = np.zeros((4, 1, 7), dtype=np.float64)
    poses[:, 0, 0] = [0.0, 0.1, 0.4, 0.9]
    poses[:, 0, 3] = np.cos(0.5 * angles)
    poses[:, 0, 6] = np.sin(0.5 * angles)
    poses[1::2, 0, 3:7] *= -1.0

    twists = derive_object_twists_w(poses, fps=10.0)

    np.testing.assert_allclose(twists[:, 0, 0], [1.0, 2.0, 4.0, 5.0])
    np.testing.assert_allclose(twists[:, 0, 1:3], 0.0)
    np.testing.assert_allclose(twists[:, 0, 3:5], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(twists[:, 0, 5], 1.0, atol=1.0e-6)


def test_loading_legacy_pose_only_npz_derives_object_twists(tmp_path) -> None:
    current = save_dexterous_reference_npz(_make_reference(), tmp_path / "current.npz")
    with np.load(current, allow_pickle=False) as archive:
        fields = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "object_twists_w"
        }
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **fields)

    loaded = load_dexterous_reference_npz(legacy)

    assert loaded.object_twists_w.shape == (3, 1, 6)
    np.testing.assert_allclose(loaded.object_twists_w[..., 0], 0.5)


def test_reference_schema_v1_loads_as_fixed_base_v2(tmp_path) -> None:
    current = save_dexterous_reference_npz(_make_reference(), tmp_path / "current.npz")
    v2_only = {
        "robot_layout",
        "left_joint_names",
        "right_joint_names",
        "left_wrist_twist_w",
        "right_wrist_twist_w",
    }
    with np.load(current, allow_pickle=False) as archive:
        fields = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name not in v2_only
        }
    fields["schema_version"] = np.asarray("iltools_dexterous_reference/v1")
    legacy = tmp_path / "legacy_v1_reference.npz"
    np.savez(legacy, **fields)

    loaded = load_dexterous_reference_npz(legacy)

    assert loaded.schema_version == "iltools_dexterous_reference/v2"
    assert loaded.robot_layout == "fixed_base"
    assert loaded.left_joint_names == ()
    assert loaded.right_joint_names == ()
    assert loaded.left_wrist_twist_w.shape == (loaded.frame_count, 6)
    assert loaded.right_wrist_twist_w.shape == (loaded.frame_count, 6)


def test_scene_asset_hash_verification_is_strict_and_content_bound(tmp_path) -> None:
    object_path = tmp_path / "cube.usda"
    support_path = tmp_path / "table.usda"
    object_path.write_text("#usda 1.0\n", encoding="utf-8")
    support_path.write_text(
        '#usda 1.0\ndef Xform "table" {}\n',
        encoding="utf-8",
    )
    reference = _make_reference()
    reference.object_asset_paths = (str(object_path),)
    reference.support_surface_asset_paths = (str(support_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(object_path.read_bytes()).hexdigest(),
    )
    reference.support_surface_asset_sha256 = (
        hashlib.sha256(support_path.read_bytes()).hexdigest(),
    )

    reference.verify_scene_assets(require_hashes=True)
    object_path.write_text(
        '#usda 1.0\ndef Cube "changed" {}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Object asset hash mismatch"):
        reference.verify_scene_assets(require_hashes=True)


def test_urdf_collision_dependencies_round_trip_and_ignore_materials(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    _, mesh_path = _set_urdf_object(reference, tmp_path)
    output = save_dexterous_reference_npz(reference, tmp_path / "urdf_reference.npz")

    with np.load(output, allow_pickle=False) as archive:
        assert "collision_asset_dependencies_json" in archive.files
    loaded = load_dexterous_reference_npz(output)

    assert loaded.collision_asset_dependencies == (
        CollisionAssetDependency(
            asset_role="object",
            asset_index=0,
            uri="collision.obj",
            sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        ),
    )
    verify_training_qualification(loaded)


def test_legacy_urdf_reference_loads_but_fails_strict_dependency_gate(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    _set_urdf_object(reference, tmp_path)
    current = save_dexterous_reference_npz(reference, tmp_path / "current.npz")
    with np.load(current, allow_pickle=False) as archive:
        fields = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "collision_asset_dependencies_json"
        }
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, **fields)

    loaded = load_dexterous_reference_npz(legacy)

    assert loaded.collision_asset_dependencies == ()
    with pytest.raises(ValueError, match=r"missing=\['collision.obj'\]"):
        verify_training_qualification(loaded)


def test_training_qualification_rejects_changed_urdf_collision_mesh(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    _, mesh_path = _set_urdf_object(reference, tmp_path)
    mesh_path.write_text("v 0 0 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dependency hash mismatch"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_unlisted_urdf_collision_mesh(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    urdf_path, _ = _set_urdf_object(reference, tmp_path)
    second_mesh = urdf_path.parent / "second.obj"
    second_mesh.write_text("v 0 0 0\n", encoding="utf-8")
    urdf_path.write_text(
        urdf_path.read_text(encoding="utf-8").replace(
            "</link>",
            '<collision><geometry><mesh filename="second.obj"/></geometry>'
            "</collision></link>",
        ),
        encoding="utf-8",
    )
    reference.object_asset_sha256 = (
        hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match=r"missing=\['second.obj'\]"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_nonlocal_urdf_collision_uri(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    urdf_path, _ = _write_urdf_object(tmp_path)
    urdf_path.write_text(
        urdf_path.read_text(encoding="utf-8").replace(
            'filename="collision.obj"',
            'filename="package://objects/collision.obj"',
        ),
        encoding="utf-8",
    )
    reference.object_asset_paths = (str(urdf_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    )
    reference.collision_asset_dependencies = ()

    with pytest.raises(ValueError, match="requires local files"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_visual_only_urdf(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    urdf_path, _ = _write_urdf_object(tmp_path)
    urdf_path.write_text(
        '<robot name="object"><link name="object"><visual><geometry>'
        '<mesh filename="collision.obj"/></geometry></visual></link></robot>\n',
        encoding="utf-8",
    )
    reference.object_asset_paths = (str(urdf_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    )
    reference.collision_asset_dependencies = ()

    with pytest.raises(ValueError, match="has no collision geometry"):
        verify_training_qualification(reference)


def test_training_qualification_accepts_positive_urdf_collision_primitive(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    urdf_path = tmp_path / "primitive.urdf"
    urdf_path.write_text(
        '<robot name="object"><link name="object"><collision><geometry>'
        '<box size="0.1 0.2 0.3"/></geometry></collision></link></robot>\n',
        encoding="utf-8",
    )
    reference.object_asset_paths = (str(urdf_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
    )
    reference.collision_asset_dependencies = ()

    verify_training_qualification(reference)


def test_urdf_dependency_builder_deduplicates_resolved_mesh_aliases(tmp_path) -> None:
    urdf_path, _ = _write_urdf_object(tmp_path)
    urdf_path.write_text(
        urdf_path.read_text(encoding="utf-8").replace(
            "</link>",
            '<collision><geometry><mesh filename="./collision.obj"/></geometry>'
            "</collision></link>",
        ),
        encoding="utf-8",
    )

    dependencies = build_urdf_collision_asset_dependencies(
        urdf_path,
        asset_role="object",
        asset_index=0,
    )

    assert len(dependencies) == 1
    assert dependencies[0].uri == "collision.obj"


def test_training_qualification_rejects_external_buffer_mesh_format(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    gltf_path = tmp_path / "collision.gltf"
    gltf_path.write_text('{"buffers": [{"uri": "collision.bin"}]}\n')
    reference.object_asset_paths = (str(gltf_path),)
    reference.object_asset_sha256 = (
        hashlib.sha256(gltf_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="does not support scene asset format"):
        verify_training_qualification(reference)


def test_urdf_dependency_builder_rejects_external_buffer_mesh_format(
    tmp_path,
) -> None:
    urdf_path, _ = _write_urdf_object(tmp_path)
    gltf_path = urdf_path.parent / "collision.gltf"
    gltf_path.write_text('{"buffers": [{"uri": "collision.bin"}]}\n')
    urdf_path.write_text(
        urdf_path.read_text(encoding="utf-8").replace(
            'filename="collision.obj"',
            'filename="collision.gltf"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not support mesh format '.gltf'"):
        build_urdf_collision_asset_dependencies(
            urdf_path,
            asset_role="object",
            asset_index=0,
        )


def test_training_qualification_accepts_self_contained_ascii_dot_usd(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    support_path = tmp_path / "support.usd"
    support_path.write_text(
        '#usda 1.0\ndef Cube "support" {}\n',
        encoding="utf-8",
    )
    reference.support_surface_asset_paths = (str(support_path),)
    reference.support_surface_asset_sha256 = (
        hashlib.sha256(support_path.read_bytes()).hexdigest(),
    )

    verify_training_qualification(reference)


@pytest.mark.parametrize(
    ("suffix", "contents", "match"),
    (
        (
            ".usda",
            '#usda 1.0\ndef Xform "support" (references = @other.usda@) {}\n',
            "external-reference USD coverage",
        ),
        (".usd", "PXR-USDC binary placeholder", "USDA text header"),
        (".usdc", "PXR-USDC binary placeholder", "without OpenUSD/pxr"),
        (".usdz", "package placeholder", "without OpenUSD/pxr"),
    ),
)
def test_training_qualification_rejects_unverified_usd_dependencies(
    tmp_path,
    suffix: str,
    contents: str,
    match: str,
) -> None:
    reference = _make_training_reference(tmp_path)
    support_path = tmp_path / f"support{suffix}"
    support_path.write_text(contents, encoding="utf-8")
    reference.support_surface_asset_paths = (str(support_path),)
    reference.support_surface_asset_sha256 = (
        hashlib.sha256(support_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match=match):
        verify_training_qualification(reference)


def test_training_qualification_round_trip_is_typed_and_verified(tmp_path) -> None:
    output = save_dexterous_reference_npz(
        _make_training_reference(tmp_path),
        tmp_path / "qualified.npz",
    )

    with np.load(output, allow_pickle=False) as archive:
        assert "training_qualification_json" in archive.files
    loaded = load_dexterous_reference_npz(output)
    qualification = verify_training_qualification(loaded)

    assert qualification.runtime_qualified is True
    assert qualification.isaac_runtime_qualified is True
    assert qualification.inspection_only is False
    assert qualification.collision_clearance.checked_frame_count == 3
    assert loaded.scene_physics is not None
    np.testing.assert_allclose(loaded.scene_physics.object_mass_kg, [0.35])


def test_legacy_v1_qualification_loads_for_inspection_but_cannot_train(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    assert reference.training_qualification is not None
    reference.training_qualification = replace(
        reference.training_qualification,
        schema_version="iltools_dexterous_training_qualification/v1",
    )
    output = save_dexterous_reference_npz(reference, tmp_path / "legacy_v1.npz")

    loaded = load_dexterous_reference_npz(output)

    assert loaded.training_qualification is not None
    assert (
        loaded.training_qualification.schema_version
        == "iltools_dexterous_training_qualification/v1"
    )
    with pytest.raises(ValueError, match="dependency-unaware record"):
        verify_training_qualification(loaded)


def test_training_qualification_rejects_metadata_only_inspection_reference(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    reference.training_qualification = None
    reference.metadata.update(
        {
            "runtime_qualified": True,
            "isaac_runtime_qualified": True,
            "inspection_only": False,
            "contacts": {"contact_geometry": "manual"},
            "geometry_clearance_audit": {"passed": True},
        }
    )

    with pytest.raises(ValueError, match="no typed training qualification"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_inactive_contact_geometry(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    assert reference.contacts is not None
    reference.contacts.active.fill(False)

    with pytest.raises(ValueError, match="no active contact slot"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_missing_scene_physics(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    reference.scene_physics = None

    with pytest.raises(ValueError, match="no typed ScenePhysics"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_invalid_scene_physics(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    assert reference.scene_physics is not None
    reference.scene_physics.object_dynamic_friction[:] = 1.0

    with pytest.raises(ValueError, match="dynamic friction must not exceed"):
        verify_training_qualification(reference)


@pytest.mark.parametrize(
    ("qualification_kwargs", "match"),
    (
        ({"runtime_qualified": False}, "not runtime_qualified"),
        ({"isaac_runtime_qualified": False}, "not isaac_runtime_qualified"),
        ({"inspection_only": True}, "is inspection_only"),
    ),
)
def test_training_qualification_requires_runtime_flags(
    tmp_path,
    qualification_kwargs,
    match: str,
) -> None:
    reference = _make_training_reference(tmp_path)
    reference.training_qualification = _training_qualification(
        reference.frame_count,
        **qualification_kwargs,
    )

    with pytest.raises(ValueError, match=match):
        verify_training_qualification(reference)


def test_training_qualification_rejects_zero_active_contact_points(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    assert reference.contacts is not None
    reference.contacts.link_positions_w[reference.contacts.active] = 0.0

    with pytest.raises(ValueError, match="finite, non-zero geometry"):
        verify_training_qualification(reference)


def test_training_qualification_verifies_bound_scene_asset_bytes(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    object_path = Path(reference.object_asset_paths[0])
    if not object_path.is_absolute():
        object_path = tmp_path / object_path
    object_path.write_text('#usda 1.0\ndef Cube "changed" {}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Object asset hash mismatch"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_placeholder_contact_provenance() -> None:
    with pytest.raises(ValueError, match="not unavailable or placeholder"):
        TrainingQualification(
            runtime_qualified=True,
            isaac_runtime_qualified=True,
            inspection_only=False,
            contact_geometry_provenance="unavailable placeholder",
            collision_clearance=CollisionClearanceQualification(
                qualified=True,
                method="signed distance",
                scope="all scene pairs",
                provenance="audit run 1",
                checked_frame_count=3,
                minimum_signed_distance_m=0.0,
                penetration_tolerance_m=0.0,
            ),
        )


def test_training_qualification_rejects_failed_clearance_record(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    reference.training_qualification = TrainingQualification(
        runtime_qualified=True,
        isaac_runtime_qualified=True,
        inspection_only=False,
        contact_geometry_provenance="manual contact annotation batch 7",
        collision_clearance=CollisionClearanceQualification(
            qualified=False,
            method="signed-distance replay",
            scope="robot, object, and support geometry for every frame",
            provenance="Isaac Newton replay audit run 42",
            checked_frame_count=reference.frame_count,
            minimum_signed_distance_m=-0.01,
            penetration_tolerance_m=0.0002,
        ),
    )

    with pytest.raises(ValueError, match="no passing collision and clearance"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_incomplete_clearance_record(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    reference.training_qualification = TrainingQualification(
        runtime_qualified=True,
        isaac_runtime_qualified=True,
        inspection_only=False,
        contact_geometry_provenance="manual contact annotation batch 7",
        collision_clearance=CollisionClearanceQualification(
            qualified=True,
            method="signed-distance replay",
            scope="robot, object, and support geometry for every frame",
            provenance="Isaac Newton replay audit run 42",
            checked_frame_count=reference.frame_count - 1,
            minimum_signed_distance_m=0.0,
            penetration_tolerance_m=0.0002,
        ),
    )

    with pytest.raises(ValueError, match="covers 2 frames, expected 3"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_clearance_below_tolerance(tmp_path) -> None:
    reference = _make_training_reference(tmp_path)
    reference.training_qualification = TrainingQualification(
        runtime_qualified=True,
        isaac_runtime_qualified=True,
        inspection_only=False,
        contact_geometry_provenance="manual contact annotation batch 7",
        collision_clearance=CollisionClearanceQualification(
            qualified=True,
            method="signed-distance replay",
            scope="robot, object, and support geometry for every frame",
            provenance="Isaac Newton replay audit run 42",
            checked_frame_count=reference.frame_count,
            minimum_signed_distance_m=-0.001,
            penetration_tolerance_m=0.0002,
        ),
    )

    with pytest.raises(ValueError, match="exceeds its penetration tolerance"):
        verify_training_qualification(reference)


def test_training_qualification_rejects_unbounded_penetration_tolerance(
    tmp_path,
) -> None:
    reference = _make_training_reference(tmp_path)
    reference.training_qualification = TrainingQualification(
        runtime_qualified=True,
        isaac_runtime_qualified=True,
        inspection_only=False,
        contact_geometry_provenance="manual contact annotation batch 7",
        collision_clearance=CollisionClearanceQualification(
            qualified=True,
            method="signed-distance replay",
            scope="robot, object, and support geometry for every frame",
            provenance="Isaac Newton replay audit run 42",
            checked_frame_count=reference.frame_count,
            minimum_signed_distance_m=-0.001,
            penetration_tolerance_m=0.01,
        ),
    )

    with pytest.raises(ValueError, match="exceeds the training maximum"):
        verify_training_qualification(reference)


def test_manifest_round_trip_resolves_relative_paths_and_hashes(tmp_path) -> None:
    first = _write_hashed_reference(
        tmp_path / "motions",
        sequence_id="first",
    )
    second = _write_hashed_reference(
        tmp_path / "motions",
        sequence_id="second",
    )
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    manifest_path = create_dexterous_reference_manifest(
        (first, second),
        tmp_path / "manifests" / "references.json",
        dataset_name="synthetic cube",
        model_path=model,
        metadata={"purpose": "contract test"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["motion_count"] == 2
    assert not payload["motions"][0]["path"].startswith("/")
    manifest = load_dexterous_reference_manifest(manifest_path)
    references = manifest.load_references()
    assert tuple(item.sequence_id for item in references) == ("first", "second")
    assert len(load_dexterous_reference_set(manifest_path)) == 2

    model.write_text("<changed/>", encoding="utf-8")
    with pytest.raises(ValueError, match="Robot model hash mismatch"):
        manifest.load_references()
    model.write_text("<mujoco/>", encoding="utf-8")

    save_dexterous_reference_npz(_make_reference(sequence_id="changed"), first)
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest.load_references()


def test_manifest_creation_rejects_missing_scene_asset_hashes(tmp_path) -> None:
    motion_directory = tmp_path / "motions"
    _write_scene_assets(motion_directory)
    valid_reference_path = _write_hashed_reference(
        motion_directory,
        sequence_id="valid",
    )
    reference_path = save_dexterous_reference_npz(
        _make_reference(sequence_id="unhashed"),
        motion_directory / "unhashed.npz",
    )
    output_path = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="has no declared SHA-256"):
        create_dexterous_reference_manifest(
            (valid_reference_path, reference_path),
            output_path,
            dataset_name="missing scene hashes",
        )
    assert not output_path.exists()


def _write_runtime_binding_manifest(tmp_path, *, declare_model: bool):
    reference_path = _write_hashed_reference(tmp_path, sequence_id="cube_pick")
    declared_model = tmp_path / "declared_robot.xml"
    declared_model.write_text('<mujoco model="vega_wuji"/>', encoding="utf-8")
    manifest_path = create_dexterous_reference_manifest(
        (reference_path,),
        tmp_path / "manifest.json",
        dataset_name="runtime binding",
        model_path=declared_model if declare_model else None,
    )
    return manifest_path, declared_model


def test_runtime_model_hash_binding_accepts_matching_copy(tmp_path) -> None:
    manifest_path, declared_model = _write_runtime_binding_manifest(
        tmp_path, declare_model=True
    )
    runtime_model = tmp_path / "runtime" / "robot.xml"
    runtime_model.parent.mkdir()
    runtime_model.write_bytes(declared_model.read_bytes())

    manifest = load_dexterous_reference_manifest(manifest_path)
    assert manifest.verify_runtime_model(runtime_model) == runtime_model.resolve()
    declared_model.unlink()
    references = load_dexterous_reference_set(
        manifest_path,
        runtime_model_path=runtime_model,
        require_model_hash=True,
    )
    assert len(references) == 1


def test_runtime_model_hash_binding_rejects_mismatch(tmp_path) -> None:
    manifest_path, _ = _write_runtime_binding_manifest(tmp_path, declare_model=True)
    runtime_model = tmp_path / "runtime_robot.xml"
    runtime_model.write_text('<mujoco model="other"/>', encoding="utf-8")

    with pytest.raises(ValueError, match="Runtime robot model hash mismatch"):
        load_dexterous_reference_set(
            manifest_path,
            runtime_model_path=runtime_model,
            require_model_hash=True,
        )


def test_runtime_model_hash_binding_requires_declaration_in_strict_mode(
    tmp_path,
) -> None:
    manifest_path, runtime_model = _write_runtime_binding_manifest(
        tmp_path, declare_model=False
    )
    manifest = load_dexterous_reference_manifest(manifest_path)

    with pytest.raises(ValueError, match="does not declare model"):
        manifest.verify_runtime_model(runtime_model)
    assert (
        manifest.verify_runtime_model(runtime_model, require_declared=False)
        == runtime_model.resolve()
    )
    with pytest.raises(ValueError, match="does not declare model"):
        load_dexterous_reference_set(
            manifest_path,
            runtime_model_path=runtime_model,
            require_model_hash=True,
        )


def test_reference_rejects_non_unit_quaternion() -> None:
    reference = _make_reference()
    reference.left_wrist_pose_w[:, 3] = 2.0

    with pytest.raises(ValueError, match="unit WXYZ"):
        DexterousReference(
            sequence_id=reference.sequence_id,
            robot_name=reference.robot_name,
            fps=reference.fps,
            joint_names=reference.joint_names,
            qpos=reference.qpos,
            qvel=reference.qvel,
            fixed_root_pose_w=reference.fixed_root_pose_w,
            left_wrist_pose_w=reference.left_wrist_pose_w,
            right_wrist_pose_w=reference.right_wrist_pose_w,
            left_wrist_frame_name=reference.left_wrist_frame_name,
            right_wrist_frame_name=reference.right_wrist_frame_name,
            object_names=reference.object_names,
            object_poses_w=reference.object_poses_w,
        )


def test_contact_rejects_active_unknown_object() -> None:
    reference = _make_reference()
    assert reference.contacts is not None
    reference.contacts.object_indices[:, 0, 0] = 4

    with pytest.raises(ValueError, match="declared object"):
        reference.contacts.validate(frame_count=reference.frame_count, object_count=1)


def test_contact_rejects_duplicate_named_link_within_hand() -> None:
    reference = _make_reference()
    assert reference.contacts is not None
    contacts = reference.contacts
    duplicate_names = contacts.link_names.copy()
    duplicate_names[0, 1] = duplicate_names[0, 0]

    with pytest.raises(ValueError, match="unique within each hand"):
        ContactSequence(
            hand_sides=contacts.hand_sides,
            link_names=duplicate_names,
            link_positions_w=contacts.link_positions_w,
            link_normals_w=contacts.link_normals_w,
            object_positions_w=contacts.object_positions_w,
            object_normals_w=contacts.object_normals_w,
            object_indices=contacts.object_indices,
            active=contacts.active,
        )


def _relocation_reference(tmp_path, asset_dir_name="assets"):
    """A dual-hand reference whose object asset lives in a sibling tree."""

    from iltools.core import sha256_file

    asset_dir = tmp_path / "home" / "sharpa" / asset_dir_name / "box"
    asset_dir.mkdir(parents=True)
    asset = asset_dir / "object.urdf"
    asset.write_text("<robot name='box'><link name='object'/></robot>", "utf-8")
    frames = 2
    identity = np.zeros((frames, 4), dtype=np.float32)
    identity[:, 0] = 1.0
    reference = DexterousReference(
        sequence_id="relocatable",
        robot_name="sharpa_wave",
        fps=20.0,
        joint_names=("j0",),
        qpos=np.zeros((frames, 1), dtype=np.float32),
        fixed_root_pose_w=np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        left_wrist_pose_w=np.concatenate(
            (np.zeros((frames, 3), dtype=np.float32), identity), axis=-1
        ),
        right_wrist_pose_w=np.concatenate(
            (np.zeros((frames, 3), dtype=np.float32), identity), axis=-1
        ),
        left_wrist_frame_name="left_hand_C_MC",
        right_wrist_frame_name="right_hand_C_MC",
        object_names=("box",),
        object_poses_w=np.concatenate(
            (np.zeros((frames, 1, 3), dtype=np.float32), identity[:, None]), axis=-1
        ),
        object_asset_paths=(str(asset),),
        object_asset_sha256=(sha256_file(asset),),
    )
    return reference, asset


def test_a_relocated_asset_tree_still_resolves_and_verifies(tmp_path):
    """A Reference set copied to another root finds its asset by content."""

    reference, _asset = _relocation_reference(tmp_path)
    save_dexterous_reference_npz(
        reference, tmp_path / "home" / "sharpa" / "set" / "references" / "00000.npz"
    )
    # Move the whole tree, exactly as staging to a cluster bind does.
    moved = tmp_path / "data" / "sharpa"
    moved.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "home" / "sharpa").rename(moved)

    loaded = load_dexterous_reference_npz(moved / "set" / "references" / "00000.npz")
    assert loaded.object_asset_paths[0] == str(moved / "assets" / "box" / "object.urdf")
    loaded.verify_scene_assets(require_hashes=True)


def test_relocation_refuses_a_same_named_file_with_other_content(tmp_path):
    """Only content decides; a decoy with the recorded name is not accepted."""

    reference, _asset = _relocation_reference(tmp_path)
    save_dexterous_reference_npz(
        reference, tmp_path / "home" / "sharpa" / "set" / "references" / "00000.npz"
    )
    moved = tmp_path / "data" / "sharpa"
    moved.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "home" / "sharpa").rename(moved)
    # Same path tail, different bytes.
    (moved / "assets" / "box" / "object.urdf").write_text("<robot/>", "utf-8")

    loaded = load_dexterous_reference_npz(moved / "set" / "references" / "00000.npz")
    with pytest.raises(FileNotFoundError):
        loaded.verify_scene_assets(require_hashes=True)


def test_relocation_needs_a_recorded_hash(tmp_path):
    """Without a declared digest there is nothing to verify, so nothing moves."""

    from iltools.core.dexterous_reference import _relocated_asset

    recorded = tmp_path / "gone" / "assets" / "box" / "object.urdf"
    base = tmp_path / "set" / "references"
    base.mkdir(parents=True)
    assert _relocated_asset(recorded, base, "") is None


def test_an_existing_asset_path_is_left_alone(tmp_path):
    """Relocation is a fallback; a resolvable path is never rewritten."""

    reference, asset = _relocation_reference(tmp_path)
    npz = save_dexterous_reference_npz(
        reference, tmp_path / "home" / "sharpa" / "set" / "references" / "00000.npz"
    )
    loaded = load_dexterous_reference_npz(npz)
    assert loaded.object_asset_paths[0] == str(asset)
    loaded.verify_scene_assets(require_hashes=True)
