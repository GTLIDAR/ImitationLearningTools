from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from iltools.retarget.soma_chord import (
    SOMA_HAND_LINK_JOINTS,
    extract_chord_contact_targets,
    infer_soma_frame_bridge,
    invert_soma_global_rotations,
    reconstruct_soma_identity_rest_pose,
    vertex_normals,
)


def _wxyz(matrices: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    return xyzw[:, (3, 0, 1, 2)].reshape((*matrices.shape[:-2], 4))


def test_reconstruct_soma_identity_rest_pose_binds_public_joint_order() -> None:
    import torch

    class FakeLayer:
        public_joint_names = ("Root", "Hips", "LeftHand")

        def __call__(
            self,
            pose,
            identity,
            *,
            scale_params,
            apply_correctives,
        ):
            assert tuple(pose.shape) == (1, 2, 3)
            assert tuple(identity.shape) == (1, 2)
            assert tuple(scale_params.shape) == (1, 1)
            assert apply_correctives is False
            return {
                "joints": torch.tensor(
                    [[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]], dtype=torch.float32
                ),
                "transforms": torch.eye(4, dtype=torch.float32).repeat(1, 3, 1, 1),
            }

    result = reconstruct_soma_identity_rest_pose(
        identity_coeffs=np.asarray((0.2, -0.1)),
        scale_params=np.asarray((1.0,)),
        joint_names=("Hips", "LeftHand"),
        layer=FakeLayer(),
    )

    assert result.joint_names == ("Hips", "LeftHand")
    np.testing.assert_allclose(result.joints[1], (0.1, 0.2, 0.3), atol=1.0e-7)
    assert result.transforms.shape == (3, 4, 4)


def test_reconstruct_soma_identity_rest_pose_rejects_joint_order_mismatch() -> None:
    class WrongLayer:
        public_joint_names = ("Root", "LeftHand", "Hips")

    with pytest.raises(ValueError, match="public joints"):
        reconstruct_soma_identity_rest_pose(
            identity_coeffs=np.asarray((0.0,)),
            scale_params=np.asarray((1.0,)),
            joint_names=("Hips", "LeftHand"),
            layer=WrongLayer(),
        )


def test_soma_non_thumb_contact_links_skip_the_metacarpal() -> None:
    assert SOMA_HAND_LINK_JOINTS["palm"] == (
        "Hand",
        "HandThumb1",
        "HandIndex1",
        "HandMiddle1",
        "HandRing1",
        "HandPinky1",
    )
    assert SOMA_HAND_LINK_JOINTS["index1"] == ("HandIndex2", "HandIndex3")
    assert SOMA_HAND_LINK_JOINTS["index2"] == ("HandIndex3", "HandIndex4")
    assert SOMA_HAND_LINK_JOINTS["index3"] == ("HandIndex4", "HandIndexEnd")


def test_invert_soma_global_rotations_removes_root_normalization() -> None:
    parents = np.asarray((0, 0, 1, 2), dtype=np.int64)
    orient = Rotation.from_rotvec(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (0.2, -0.1, 0.05),
                (-0.1, 0.15, 0.08),
                (0.04, -0.06, 0.12),
            )
        )
    ).as_matrix()
    relative = (
        Rotation.from_rotvec(
            np.asarray(
                (
                    (
                        (0.0, 0.0, 0.0),
                        (0.3, 0.0, -0.1),
                        (0.0, 0.2, 0.0),
                        (0.1, 0.0, 0.2),
                    ),
                    (
                        (0.0, 0.0, 0.0),
                        (0.4, -0.1, -0.05),
                        (0.0, 0.3, 0.1),
                        (0.2, 0.0, 0.25),
                    ),
                )
            ).reshape(-1, 3)
        )
        .as_matrix()
        .reshape(2, 4, 3, 3)
    )
    oriented = orient[parents][None].transpose(0, 1, 3, 2) @ relative @ orient[None]
    world = oriented.copy()
    for joint in range(1, len(parents)):
        world[:, joint] = world[:, parents[joint]] @ oriented[:, joint]
    normalization = relative[0, 1].T
    normalized_public = normalization[None, None] @ world[:, 1:]

    recovered = invert_soma_global_rotations(
        _wxyz(normalized_public),
        joint_orient_world=orient,
        joint_parent_ids=parents,
    )

    expected = relative[:, 1:].copy()
    expected[:, 0] = normalization @ expected[:, 0]
    np.testing.assert_allclose(recovered, expected, rtol=0.0, atol=1.0e-7)


def test_infer_soma_frame_bridge_recovers_rotation_and_translation() -> None:
    rotation = Rotation.from_euler("xyz", (0.2, -0.3, 0.4)).as_matrix()
    source_positions = np.asarray(((0.0, 0.0, 0.0), (0.1, -0.2, 0.3)))
    source_rotations = Rotation.from_euler(
        "xyz", ((0.1, 0.2, 0.3), (-0.2, 0.1, 0.5))
    ).as_matrix()
    translations = np.asarray(((1.0, 2.0, 3.0), (1.0, 2.0, 3.1)))
    target_positions = source_positions @ rotation.T + translations
    target_rotations = rotation[None] @ source_rotations

    bridge = infer_soma_frame_bridge(
        soma_root_positions=source_positions,
        soma_root_wxyz=_wxyz(source_rotations),
        target_root_positions=target_positions,
        target_root_wxyz=_wxyz(target_rotations),
    )

    np.testing.assert_allclose(bridge.rotation, rotation, atol=1.0e-7)
    np.testing.assert_allclose(bridge.translations, translations, atol=1.0e-7)
    assert bridge.rotation_relation == "left_multiply"
    points = np.asarray((((0.2, 0.3, 0.4),), ((-0.1, 0.2, 0.5),)))
    np.testing.assert_allclose(
        bridge.apply(points), points @ rotation.T + translations[:, None], atol=1.0e-7
    )


def test_infer_soma_frame_bridge_handles_basis_similarity_rotations() -> None:
    rotation = Rotation.from_euler("xyz", (0.2, -0.3, 0.4)).as_matrix()
    source_positions = np.asarray(((0.0, 0.0, 0.0), (0.1, -0.2, 0.3), (0.4, 0.1, -0.2)))
    source_rotations = Rotation.from_euler(
        "xyz", ((0.1, 0.2, 0.3), (-0.2, 0.1, 0.5), (0.4, -0.1, 0.2))
    ).as_matrix()
    translations = np.asarray(((1.0, 2.0, 3.0),) * 3)
    target_positions = source_positions @ rotation.T + translations
    target_rotations = rotation[None] @ source_rotations @ rotation.T[None]

    bridge = infer_soma_frame_bridge(
        soma_root_positions=source_positions,
        soma_root_wxyz=_wxyz(source_rotations),
        target_root_positions=target_positions,
        target_root_wxyz=_wxyz(target_rotations),
    )

    np.testing.assert_allclose(bridge.rotation, rotation, atol=1.0e-7)
    np.testing.assert_allclose(bridge.translations, translations, atol=1.0e-7)
    assert bridge.rotation_relation == "basis_similarity"


def test_vertex_normals_for_planar_square() -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    faces = np.asarray(((0, 1, 2), (0, 2, 3)))
    normals = vertex_normals(vertices, faces)
    np.testing.assert_allclose(
        normals,
        np.broadcast_to(np.asarray((0.0, 0.0, 1.0)), normals.shape),
        atol=1.0e-8,
    )


def test_extract_chord_contacts_uses_strict_one_centimetre_mesh_rule() -> None:
    suffixes = (
        "Hand",
        "HandThumb1",
        "HandThumb2",
        "HandThumb3",
        "HandThumbEnd",
        "HandIndex1",
        "HandIndex2",
        "HandIndex3",
        "HandIndex4",
        "HandIndexEnd",
        "HandMiddle1",
        "HandMiddle2",
        "HandMiddle3",
        "HandMiddle4",
        "HandMiddleEnd",
        "HandRing1",
        "HandRing2",
        "HandRing3",
        "HandRing4",
        "HandRingEnd",
        "HandPinky1",
        "HandPinky2",
        "HandPinky3",
        "HandPinky4",
        "HandPinkyEnd",
    )
    names = tuple(
        f"{side}{suffix}" for side in ("Left", "Right") for suffix in suffixes
    )
    joints = np.zeros((2, len(names), 3), dtype=np.float64)
    joints[:, len(suffixes) :, 0] = 1.0
    vertices = np.asarray(
        (
            ((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.0, 0.02, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, 0.0, 0.02), (0.02, 0.0, 0.02), (0.0, 0.02, 0.02), (1.0, 0.0, 0.0)),
        )
    )
    faces = np.asarray(((0, 1, 2),), dtype=np.int64)
    object_points = np.asarray(((0.005, 0.005, 0.005),), dtype=np.float64)
    object_normals = np.asarray(((0.0, 0.0, -1.0),), dtype=np.float64)
    poses = np.asarray(((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),) * 2)

    contacts = extract_chord_contact_targets(
        joint_names=names,
        joints_world=joints,
        vertices_world=vertices,
        faces=faces,
        hand_vertex_indices={"left": np.asarray((0, 1, 2)), "right": np.asarray((3,))},
        object_surface_points=object_points,
        object_surface_normals=object_normals,
        object_poses_wxyz=poses,
        source_active=np.asarray(((True, False), (True, False))),
        threshold_m=0.01,
    )

    assert contacts.report.per_side_contact_frames == {"left": 1, "right": 0}
    assert contacts.report.source_active_recalled == 1
    assert contacts.report.source_active_recall == pytest.approx(0.5)
    assert np.any(contacts.active[0, 0])
    assert not np.any(contacts.active[1])
