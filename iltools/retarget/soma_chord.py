"""Reconstruct SOMA-X hands and extract CHORD-style contact targets.

The SOMA payload used by the retargeting pipeline stores 77 global joint
rotations and positions, while :class:`soma.SOMALayer` internally includes a
synthetic ``Root`` joint.  This module provides the exact inverse of SOMA-X's
joint-orient/FK convention, reconstructs the posed 18,056-vertex mesh, and
implements the contact rule used by CHORD's data loader:

* sample 4,096 points on each object surface;
* for every object point, find the closest hand-mesh vertex;
* keep pairs whose distance is strictly below 1 cm;
* reduce the resulting cloud to one target per source hand link.

Heavy SOMA-X and trimesh imports are lazy so the rest of ILTools does not need
the optional reconstruction dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SOMA_HAND_LINK_JOINTS: dict[str, tuple[str, ...]] = {
    "palm": (
        "Hand",
        "HandThumb1",
        "HandIndex1",
        "HandMiddle1",
        "HandRing1",
        "HandPinky1",
    ),
    "thumb1": ("HandThumb1", "HandThumb2"),
    "thumb2": ("HandThumb2", "HandThumb3"),
    "thumb3": ("HandThumb3", "HandThumbEnd"),
    # The non-thumb SOMA chain includes a metacarpal joint before the MCP:
    # Hand -> *1 (CMC/metacarpal base) -> *2 (MCP) -> *3 (PIP) -> *4
    # (DIP) -> *End. Keep the metacarpal in the palm assignment and align
    # the three articulated phalanges with Wuji proximal/middle/distal links.
    "index1": ("HandIndex2", "HandIndex3"),
    "index2": ("HandIndex3", "HandIndex4"),
    "index3": ("HandIndex4", "HandIndexEnd"),
    "middle1": ("HandMiddle2", "HandMiddle3"),
    "middle2": ("HandMiddle3", "HandMiddle4"),
    "middle3": ("HandMiddle4", "HandMiddleEnd"),
    "ring1": ("HandRing2", "HandRing3"),
    "ring2": ("HandRing3", "HandRing4"),
    "ring3": ("HandRing4", "HandRingEnd"),
    "pinky1": ("HandPinky2", "HandPinky3"),
    "pinky2": ("HandPinky3", "HandPinky4"),
    "pinky3": ("HandPinky4", "HandPinkyEnd"),
}


def _as_finite(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _rotation() -> Any:
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "SOMA/CHORD retargeting requires scipy. Install iltools[soma]."
        ) from exc
    return Rotation


def _wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    quaternions = _as_finite(value, name="WXYZ quaternions")
    if quaternions.shape[-1] != 4:
        raise ValueError("WXYZ quaternions must end in four components.")
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-10):
        raise ValueError("WXYZ quaternions contain a zero quaternion.")
    quaternions = quaternions / norms
    flat = quaternions.reshape(-1, 4)
    xyzw = flat[:, (1, 2, 3, 0)]
    return (
        _rotation().from_quat(xyzw).as_matrix().reshape((*quaternions.shape[:-1], 3, 3))
    )


def _matrix_to_wxyz(value: np.ndarray) -> np.ndarray:
    matrices = _as_finite(value, name="rotation matrices")
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must end in shape [3, 3].")
    xyzw = _rotation().from_matrix(matrices.reshape(-1, 3, 3)).as_quat()
    return xyzw[:, (3, 0, 1, 2)].reshape((*matrices.shape[:-2], 4))


def invert_soma_global_rotations(
    global_joint_wxyz: np.ndarray,
    *,
    joint_orient_world: np.ndarray,
    joint_parent_ids: Sequence[int],
) -> np.ndarray:
    """Invert SOMA-X global rotations into 77 local pose matrices.

    ``global_joint_wxyz`` excludes SOMA-X's synthetic ``Root`` and may have
    been left-multiplied by a constant world-frame normalization.  Such a
    normalization cancels for all parent-relative joints.  For ``Hips``, the
    synthetic root is set to identity; this recovers the normalized root pose
    used by a layer that reproduces the normalized joint positions.

    The inverse matches ``remove_joint_orient_local(joint_world_to_local(...))``::

        absolute_local[j] = world[parent[j]].T @ world[j]
        relative_local[j] = orient[parent[j]] @ absolute_local[j] @ orient[j].T
    """

    rotations = _as_finite(global_joint_wxyz, name="global_joint_wxyz", ndim=3)
    if rotations.shape[-1] != 4:
        raise ValueError("global_joint_wxyz must have shape [T, J, 4].")
    frame_count, joint_count = rotations.shape[:2]
    orient = _as_finite(joint_orient_world, name="joint_orient_world", ndim=3)
    parents = np.asarray(joint_parent_ids, dtype=np.int64)
    if orient.shape != (joint_count + 1, 3, 3):
        raise ValueError(
            "joint_orient_world must include synthetic Root and have shape "
            f"[{joint_count + 1}, 3, 3], got {orient.shape}."
        )
    if parents.shape != (joint_count + 1,):
        raise ValueError(
            "joint_parent_ids must include synthetic Root and have length "
            f"{joint_count + 1}."
        )
    if (
        parents[0] != 0
        or np.any(parents < 0)
        or np.any(parents > np.arange(len(parents)))
    ):
        raise ValueError("joint_parent_ids must be a root-first SOMA hierarchy.")
    if not np.allclose(orient[0], np.eye(3), rtol=0.0, atol=1.0e-6):
        raise ValueError("SOMA synthetic Root orientation must be identity.")

    world = np.empty((frame_count, joint_count + 1, 3, 3), dtype=np.float64)
    world[:, 0] = np.eye(3)
    world[:, 1:] = _wxyz_to_matrix(rotations)
    absolute_local = np.swapaxes(world[:, parents], -2, -1) @ world
    absolute_local[:, 0] = world[:, 0]
    relative_local = (
        orient[parents][None] @ absolute_local @ np.swapaxes(orient, -2, -1)[None]
    )
    return relative_local[:, 1:]


@dataclass(frozen=True)
class SomaReconstructionReport:
    """Numerical evidence that reconstructed SOMA joints match the payload."""

    frame_count: int
    joint_count: int
    mean_joint_error_m: float
    p95_joint_error_m: float
    max_joint_error_m: float
    left_hand_vertex_count: int
    right_hand_vertex_count: int

    def as_dict(self) -> dict[str, int | float]:
        return {
            "frame_count": self.frame_count,
            "joint_count": self.joint_count,
            "mean_joint_error_m": self.mean_joint_error_m,
            "p95_joint_error_m": self.p95_joint_error_m,
            "max_joint_error_m": self.max_joint_error_m,
            "left_hand_vertex_count": self.left_hand_vertex_count,
            "right_hand_vertex_count": self.right_hand_vertex_count,
        }


@dataclass(frozen=True)
class SomaMotionReconstruction:
    """Posed SOMA mesh and the local parameters that reproduce it."""

    joint_names: tuple[str, ...]
    poses_rotvec: np.ndarray
    translations: np.ndarray
    joints: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray
    hand_vertex_indices: Mapping[str, np.ndarray]
    report: SomaReconstructionReport


@dataclass(frozen=True)
class SomaIdentityRestPose:
    """Identity-specific SOMA joints and joint frames at zero local pose.

    ``joints`` follows ``joint_names`` and excludes the synthetic ``Root``.
    ``transforms`` includes that root at index zero and therefore follows
    ``("Root", *joint_names)``.  Positions are in meters.
    """

    joint_names: tuple[str, ...]
    joints: np.ndarray
    transforms: np.ndarray


def reconstruct_soma_identity_rest_pose(
    *,
    identity_coeffs: np.ndarray,
    scale_params: np.ndarray,
    joint_names: Sequence[str],
    layer: Any | None = None,
    device: str = "cpu",
) -> SomaIdentityRestPose:
    """Evaluate one identity-specific SOMA skeleton at zero local pose.

    This is the geometry-only calibration witness needed when articulated
    metacarpals make posed MCP locations unsuitable for a palm-frame fit.
    Heavy SOMA-X imports remain lazy, as in :func:`reconstruct_soma_motion`.
    """

    names = tuple(str(name) for name in joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be non-empty and unique.")
    identity = _as_finite(identity_coeffs, name="identity_coeffs", ndim=1).astype(
        np.float32
    )
    scales = _as_finite(scale_params, name="scale_params", ndim=1).astype(np.float32)
    if layer is None:
        try:
            from soma import SOMALayer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SOMA reconstruction requires py-soma-x. Install iltools[soma]."
            ) from exc
        layer = SOMALayer(
            device=device,
            identity_model_type="mhr",
            enable_procedural_transforms=False,
            correctives_model_path=None,
        )

    public_names = tuple(str(name) for name in layer.public_joint_names)
    if public_names != ("Root", *names):
        raise ValueError(
            "SOMA payload names do not match the layer's public joints after Root."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - ILTools requires torch
        raise ImportError("SOMA reconstruction requires torch.") from exc
    torch_device = torch.device(device)
    with torch.no_grad():
        output = layer(
            torch.zeros((1, len(names), 3), dtype=torch.float32, device=torch_device),
            torch.from_numpy(identity).to(torch_device).unsqueeze(0),
            scale_params=torch.from_numpy(scales).to(torch_device).unsqueeze(0),
            apply_correctives=False,
        )
    joints = np.asarray(output["joints"].detach().cpu().numpy()[0], dtype=np.float64)
    transforms = np.asarray(
        output["transforms"].detach().cpu().numpy()[0], dtype=np.float64
    )
    if joints.shape != (len(names), 3) or transforms.shape != (
        len(names) + 1,
        4,
        4,
    ):
        raise ValueError(
            "SOMALayer zero-pose output has an unexpected joint/transform shape."
        )
    if not np.isfinite(joints).all() or not np.isfinite(transforms).all():
        raise ValueError("SOMALayer zero-pose output contains non-finite values.")
    return SomaIdentityRestPose(
        joint_names=names,
        joints=joints,
        transforms=transforms,
    )


def _layer_faces(layer: Any) -> np.ndarray:
    rig_data = layer.rig_data
    keys = set(rig_data.files) if hasattr(rig_data, "files") else set(rig_data)
    for key in ("triangles", "faces"):
        if key in keys:
            faces = np.asarray(rig_data[key], dtype=np.int64)
            if faces.ndim == 2 and faces.shape[1] == 3:
                return faces
    raise ValueError("SOMALayer rig data does not expose triangular mesh faces.")


def _soma_hand_vertex_indices(layer: Any) -> dict[str, np.ndarray]:
    try:
        from soma.geometry.rig_utils import get_body_part_vertex_ids
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "SOMA reconstruction requires py-soma-x. Install iltools[soma]."
        ) from exc

    names = tuple(str(name) for name in layer.public_joint_names)
    parents = layer.output_joint_parent_ids
    weights = layer.public_skinning_weights()
    result: dict[str, np.ndarray] = {}
    for side, root_name in (("left", "LeftHand"), ("right", "RightHand")):
        if root_name not in names:
            raise ValueError(f"SOMALayer has no public joint {root_name!r}.")
        ids = get_body_part_vertex_ids(
            weights,
            parents,
            names.index(root_name),
            include_root=True,
            weight_threshold=0.01,
        )
        result[side] = np.asarray(ids, dtype=np.int64)
    return result


def reconstruct_soma_motion(
    *,
    global_joint_positions: np.ndarray,
    global_joint_wxyz: np.ndarray,
    identity_coeffs: np.ndarray,
    scale_params: np.ndarray,
    joint_names: Sequence[str],
    layer: Any | None = None,
    device: str = "cpu",
    chunk_size: int = 128,
    max_joint_error_m: float = 0.05,
) -> SomaMotionReconstruction:
    """Reconstruct a posed SOMA-X mesh from a global-joint payload.

    A supplied layer is useful for long-lived pipelines.  Otherwise a public
    MHR SOMA layer is created in non-procedural mode with correctives disabled.
    The recovered translation is the exact per-frame shift aligning the
    layer's Hips joint to the payload Hips joint.
    """

    positions = _as_finite(
        global_joint_positions, name="global_joint_positions", ndim=3
    )
    rotations = _as_finite(global_joint_wxyz, name="global_joint_wxyz", ndim=3)
    if positions.shape[:2] != rotations.shape[:2] or positions.shape[-1] != 3:
        raise ValueError("Joint positions/rotations must have shapes [T, J, 3/4].")
    if rotations.shape[-1] != 4:
        raise ValueError("global_joint_wxyz must have shape [T, J, 4].")
    names = tuple(str(name) for name in joint_names)
    if len(names) != positions.shape[1]:
        raise ValueError("joint_names must align with the joint payload.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if not np.isfinite(max_joint_error_m) or max_joint_error_m <= 0.0:
        raise ValueError("max_joint_error_m must be finite and positive.")

    if layer is None:
        try:
            from soma import SOMALayer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "SOMA reconstruction requires py-soma-x. Install iltools[soma]."
            ) from exc
        layer = SOMALayer(
            device=device,
            identity_model_type="mhr",
            enable_procedural_transforms=False,
            correctives_model_path=None,
        )

    public_names = tuple(str(name) for name in layer.public_joint_names)
    if public_names[:1] != ("Root",) or public_names[1:] != names:
        raise ValueError(
            "SOMA payload names do not match the layer's 77 public joints after Root."
        )
    orient = layer.t_pose_world.detach().cpu().numpy()[..., :3, :3]
    parents = layer.output_joint_parent_ids.detach().cpu().numpy()
    pose_matrices = invert_soma_global_rotations(
        rotations,
        joint_orient_world=orient,
        joint_parent_ids=parents,
    )
    pose_rotvec = _rotation().from_matrix(pose_matrices.reshape(-1, 3, 3)).as_rotvec()
    pose_rotvec = pose_rotvec.reshape((*pose_matrices.shape[:2], 3)).astype(np.float32)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - ILTools requires torch
        raise ImportError("SOMA reconstruction requires torch.") from exc
    torch_device = torch.device(device)
    identity = _as_finite(identity_coeffs, name="identity_coeffs", ndim=1).astype(
        np.float32
    )
    scales = _as_finite(scale_params, name="scale_params", ndim=1).astype(np.float32)
    zero_joints: list[np.ndarray] = []
    zero_vertices: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(positions), chunk_size):
            stop = min(start + chunk_size, len(positions))
            count = stop - start
            output = layer(
                torch.from_numpy(pose_rotvec[start:stop]).to(torch_device),
                torch.from_numpy(identity)
                .to(torch_device)
                .unsqueeze(0)
                .expand(count, -1),
                scale_params=torch.from_numpy(scales)
                .to(torch_device)
                .unsqueeze(0)
                .expand(count, -1),
                apply_correctives=False,
            )
            zero_joints.append(output["joints"].detach().cpu().numpy())
            zero_vertices.append(output["vertices"].detach().cpu().numpy())
    reconstructed_zero = np.concatenate(zero_joints, axis=0).astype(np.float64)
    vertices_zero = np.concatenate(zero_vertices, axis=0).astype(np.float64)
    translations = positions[:, 0] - reconstructed_zero[:, 0]
    reconstructed = reconstructed_zero + translations[:, None, :]
    vertices = vertices_zero + translations[:, None, :]
    errors = np.linalg.norm(reconstructed - positions, axis=-1)
    hand_vertices = _soma_hand_vertex_indices(layer)
    report = SomaReconstructionReport(
        frame_count=len(positions),
        joint_count=positions.shape[1],
        mean_joint_error_m=float(np.mean(errors)),
        p95_joint_error_m=float(np.percentile(errors, 95.0)),
        max_joint_error_m=float(np.max(errors)),
        left_hand_vertex_count=len(hand_vertices["left"]),
        right_hand_vertex_count=len(hand_vertices["right"]),
    )
    if report.max_joint_error_m > max_joint_error_m:
        raise ValueError(
            "SOMA reconstruction does not reproduce the payload: maximum joint "
            f"error {report.max_joint_error_m:.6f} m exceeds {max_joint_error_m:.6f} m."
        )
    return SomaMotionReconstruction(
        joint_names=names,
        poses_rotvec=pose_rotvec.astype(np.float64),
        translations=translations,
        joints=reconstructed,
        vertices=vertices,
        faces=_layer_faces(layer),
        hand_vertex_indices=hand_vertices,
        report=report,
    )


@dataclass(frozen=True)
class SomaFrameBridge:
    """Rigid rotation plus per-frame translation into the object world frame."""

    rotation: np.ndarray
    translations: np.ndarray
    max_rotation_deviation_rad: float
    rotation_relation: str

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = _as_finite(points, name="bridge points")
        if (
            values.ndim != 3
            or values.shape[0] != len(self.translations)
            or values.shape[-1] != 3
        ):
            raise ValueError("Bridge points must have shape [T, N, 3].")
        return values @ self.rotation.T + self.translations[:, None, :]


def infer_soma_frame_bridge(
    *,
    soma_root_positions: np.ndarray,
    soma_root_wxyz: np.ndarray,
    target_root_positions: np.ndarray,
    target_root_wxyz: np.ndarray,
    max_rotation_deviation_rad: float = 1.0e-3,
) -> SomaFrameBridge:
    """Infer the saved SOMA-to-object-world transform from root witnesses.

    Upstream datasets use one of two rotation conventions. World rotations
    are left-multiplied (``B = R A``), while body-local rotations are changed
    to the robot basis (``B = R A R.T``). Both imply the same point transform.
    This function fits both contracts and selects the one with lower angular
    residual, failing closed when neither describes one rigid rotation.
    """

    source_positions = _as_finite(
        soma_root_positions, name="soma_root_positions", ndim=2
    )
    target_positions = _as_finite(
        target_root_positions, name="target_root_positions", ndim=2
    )
    if source_positions.shape != target_positions.shape or source_positions.shape[
        1:
    ] != (3,):
        raise ValueError("Root position arrays must share shape [T, 3].")
    source_rotations = _wxyz_to_matrix(soma_root_wxyz)
    target_rotations = _wxyz_to_matrix(target_root_wxyz)
    if source_rotations.shape != target_rotations.shape or source_rotations.shape[
        0
    ] != len(source_positions):
        raise ValueError("Root rotation arrays must share shape [T, 4].")
    Rotation = _rotation()
    left_bridges = target_rotations @ np.swapaxes(source_rotations, -2, -1)
    left_rotation = Rotation.from_matrix(left_bridges).mean()
    left_deviations = (
        left_rotation.inv() * Rotation.from_matrix(left_bridges)
    ).magnitude()
    left_max = float(np.max(left_deviations))

    source_rotvec = Rotation.from_matrix(source_rotations).as_rotvec()
    target_rotvec = Rotation.from_matrix(target_rotations).as_rotvec()
    informative = np.linalg.norm(source_rotvec, axis=1) > 1.0e-5
    if np.count_nonzero(informative) >= 2:
        similarity_rotation, _ = Rotation.align_vectors(
            target_rotvec[informative], source_rotvec[informative]
        )
        similarity_matrix = similarity_rotation.as_matrix()
        predicted = (
            similarity_matrix[None] @ source_rotations @ similarity_matrix.T[None]
        )
        similarity_deviations = (
            Rotation.from_matrix(predicted).inv()
            * Rotation.from_matrix(target_rotations)
        ).magnitude()
        similarity_max = float(np.max(similarity_deviations))
    else:
        similarity_rotation = Rotation.identity()
        similarity_max = float("inf")

    if left_max <= similarity_max:
        mean_rotation = left_rotation
        max_deviation = left_max
        relation = "left_multiply"
    else:
        mean_rotation = similarity_rotation
        max_deviation = similarity_max
        relation = "basis_similarity"
    if max_deviation > max_rotation_deviation_rad:
        raise ValueError(
            "SOMA root witnesses do not define one rigid frame rotation: maximum "
            f"deviation {max_deviation:.6g} rad exceeds {max_rotation_deviation_rad:.6g}."
        )
    rotation = mean_rotation.as_matrix()
    translations = target_positions - source_positions @ rotation.T
    return SomaFrameBridge(
        rotation=rotation,
        translations=translations,
        max_rotation_deviation_rad=max_deviation,
        rotation_relation=relation,
    )


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute outward area-weighted vertex normals for one posed mesh."""

    points = _as_finite(vertices, name="vertices", ndim=2)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.shape[1:] != (3,) or triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ValueError("vertices/faces must have shapes [V, 3] and [F, 3].")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("faces contain an out-of-range vertex index.")
    tri = points[triangles]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(
        normals, lengths, out=np.zeros_like(normals), where=lengths > 1.0e-12
    )


def sample_object_surface(
    mesh_path: str | Path,
    *,
    count: int = 4096,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample CHORD's object surface representation deterministically."""

    if count < 1:
        raise ValueError("count must be positive.")
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "SOMA/CHORD contact extraction requires trimesh. Install iltools[soma]."
        ) from exc
    mesh = trimesh.load(Path(mesh_path).expanduser(), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    points, face_ids = trimesh.sample.sample_surface_even(mesh, count, seed=seed)
    outward = np.asarray(mesh.face_normals[face_ids], dtype=np.float64)
    lengths = np.linalg.norm(outward, axis=1, keepdims=True)
    outward = np.divide(
        outward, lengths, out=np.zeros_like(outward), where=lengths > 1.0e-12
    )
    return np.asarray(points, dtype=np.float64), -outward


def _side_link_contract(
    joint_names: Sequence[str], side: str
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    prefix = "Left" if side == "left" else "Right"
    names = tuple(str(name) for name in joint_names)
    index = {name: idx for idx, name in enumerate(names)}
    link_names: list[str] = []
    link_joint_ids: list[np.ndarray] = []
    for link_name, suffixes in SOMA_HAND_LINK_JOINTS.items():
        full_names = tuple(f"{prefix}{suffix}" for suffix in suffixes)
        missing = [name for name in full_names if name not in index]
        if missing:
            raise ValueError(f"SOMA payload is missing {side} hand joints: {missing}.")
        link_names.append(f"{side}_{link_name}")
        link_joint_ids.append(
            np.asarray([index[name] for name in full_names], dtype=np.int64)
        )
    return tuple(link_names), tuple(link_joint_ids)


@dataclass(frozen=True)
class ChordContactReport:
    """Contact counts and agreement with an optional source activity signal."""

    frame_count: int
    contact_side_frames: int
    contact_links: int
    per_side_contact_frames: Mapping[str, int]
    source_active_side_frames: int | None = None
    source_active_recalled: int | None = None
    source_active_recall: float | None = None
    source_inactive_false_positives: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "contact_side_frames": self.contact_side_frames,
            "contact_links": self.contact_links,
            "per_side_contact_frames": dict(self.per_side_contact_frames),
            "source_active_side_frames": self.source_active_side_frames,
            "source_active_recalled": self.source_active_recalled,
            "source_active_recall": self.source_active_recall,
            "source_inactive_false_positives": self.source_inactive_false_positives,
        }


@dataclass(frozen=True)
class ChordContactTargets:
    """Dense per-SOMA-link contact targets in the shared object world frame."""

    sides: tuple[str, ...]
    link_names: tuple[tuple[str, ...], ...]
    active: np.ndarray
    hand_positions: np.ndarray
    hand_normals: np.ndarray
    object_positions: np.ndarray
    object_normals: np.ndarray
    object_part_ids: np.ndarray
    minimum_distances: np.ndarray
    report: ChordContactReport


def extract_chord_contact_targets(
    *,
    joint_names: Sequence[str],
    joints_world: np.ndarray,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    hand_vertex_indices: Mapping[str, np.ndarray],
    object_surface_points: np.ndarray,
    object_surface_normals: np.ndarray,
    object_poses_wxyz: np.ndarray,
    source_active: np.ndarray | None = None,
    threshold_m: float = 0.01,
) -> ChordContactTargets:
    """Apply CHORD's 1 cm mesh rule and reduce contacts to SOMA hand links."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "SOMA/CHORD contact extraction requires scipy. Install iltools[soma]."
        ) from exc
    joints = _as_finite(joints_world, name="joints_world", ndim=3)
    vertices = _as_finite(vertices_world, name="vertices_world", ndim=3)
    if (
        joints.shape[0] != vertices.shape[0]
        or joints.shape[-1] != 3
        or vertices.shape[-1] != 3
    ):
        raise ValueError("joints_world and vertices_world must share T and end in XYZ.")
    if joints.shape[1] != len(joint_names):
        raise ValueError("joint_names must align with joints_world.")
    surface_points = _as_finite(
        object_surface_points, name="object_surface_points", ndim=2
    )
    surface_normals = _as_finite(
        object_surface_normals, name="object_surface_normals", ndim=2
    )
    if surface_points.shape != surface_normals.shape or surface_points.shape[1:] != (
        3,
    ):
        raise ValueError("Object surface points/normals must share shape [N, 3].")
    poses = _as_finite(object_poses_wxyz, name="object_poses_wxyz", ndim=2)
    if poses.shape != (len(joints), 7):
        raise ValueError("object_poses_wxyz must have shape [T, 7] as XYZ+WXYZ.")
    if not np.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError("threshold_m must be finite and positive.")
    sides = ("left", "right")
    contracts = [_side_link_contract(joint_names, side) for side in sides]
    slot_count = max(len(contract[0]) for contract in contracts)
    shape = (len(joints), len(sides), slot_count)
    active = np.zeros(shape, dtype=bool)
    hand_positions = np.zeros((*shape, 3), dtype=np.float64)
    hand_normals = np.zeros((*shape, 3), dtype=np.float64)
    object_positions = np.zeros((*shape, 3), dtype=np.float64)
    object_normals = np.zeros((*shape, 3), dtype=np.float64)
    part_ids = np.zeros(shape, dtype=np.int32)
    minimum_distances = np.full(shape, np.inf, dtype=np.float64)
    object_rotations = _wxyz_to_matrix(poses[:, 3:7])

    for frame in range(len(joints)):
        object_points_w = surface_points @ object_rotations[frame].T + poses[frame, :3]
        object_normals_w = surface_normals @ object_rotations[frame].T
        object_tree = cKDTree(object_points_w)
        all_normals = vertex_normals(vertices[frame], faces)
        for side_index, side in enumerate(sides):
            hand_ids = np.asarray(hand_vertex_indices[side], dtype=np.int64)
            if (
                hand_ids.ndim != 1
                or np.any(hand_ids < 0)
                or np.any(hand_ids >= vertices.shape[1])
            ):
                raise ValueError(f"{side} hand_vertex_indices are invalid.")
            hand_points = vertices[frame, hand_ids]
            hand_tree = cKDTree(hand_points)
            distances, nearest_hand = hand_tree.query(object_points_w, k=1)
            contact_mask = distances < threshold_m
            if not np.any(contact_mask):
                continue
            cloud_points = hand_points[nearest_hand[contact_mask]]
            cloud_normals = all_normals[hand_ids[nearest_hand[contact_mask]]]
            cloud_distances = distances[contact_mask]
            link_names, link_joint_ids = contracts[side_index]
            link_centres = np.stack(
                [joints[frame, ids].mean(axis=0) for ids in link_joint_ids], axis=0
            )
            assignment = np.argmin(
                np.linalg.norm(
                    cloud_points[:, None, :] - link_centres[None, :, :], axis=-1
                ),
                axis=1,
            )
            for slot in range(len(link_names)):
                mask = assignment == slot
                if not np.any(mask):
                    continue
                logits = -cloud_distances[mask]
                weights = np.exp(logits - np.max(logits))
                weights /= np.sum(weights)
                point = np.sum(cloud_points[mask] * weights[:, None], axis=0)
                normal = np.sum(cloud_normals[mask] * weights[:, None], axis=0)
                normal_length = float(np.linalg.norm(normal))
                if normal_length <= 1.0e-12:
                    continue
                _, object_index = object_tree.query(point, k=1)
                active[frame, side_index, slot] = True
                hand_positions[frame, side_index, slot] = point
                hand_normals[frame, side_index, slot] = normal / normal_length
                object_positions[frame, side_index, slot] = object_points_w[
                    object_index
                ]
                object_normals[frame, side_index, slot] = object_normals_w[object_index]
                part_ids[frame, side_index, slot] = 1
                minimum_distances[frame, side_index, slot] = float(
                    np.min(cloud_distances[mask])
                )

    side_active = np.any(active, axis=-1)
    source_count: int | None = None
    recalled: int | None = None
    recall: float | None = None
    false_positives: int | None = None
    if source_active is not None:
        source = np.asarray(source_active, dtype=bool)
        if source.shape != side_active.shape:
            raise ValueError("source_active must have shape [T, 2].")
        source_count = int(np.count_nonzero(source))
        recalled = int(np.count_nonzero(source & side_active))
        recall = recalled / source_count if source_count else 0.0
        false_positives = int(np.count_nonzero(~source & side_active))
    report = ChordContactReport(
        frame_count=len(joints),
        contact_side_frames=int(np.count_nonzero(side_active)),
        contact_links=int(np.count_nonzero(active)),
        per_side_contact_frames={
            side: int(np.count_nonzero(side_active[:, index]))
            for index, side in enumerate(sides)
        },
        source_active_side_frames=source_count,
        source_active_recalled=recalled,
        source_active_recall=recall,
        source_inactive_false_positives=false_positives,
    )
    return ChordContactTargets(
        sides=sides,
        link_names=tuple(contract[0] for contract in contracts),
        active=active,
        hand_positions=hand_positions,
        hand_normals=hand_normals,
        object_positions=object_positions,
        object_normals=object_normals,
        object_part_ids=part_ids,
        minimum_distances=minimum_distances,
        report=report,
    )


__all__ = [
    "ChordContactReport",
    "ChordContactTargets",
    "SOMA_HAND_LINK_JOINTS",
    "SomaFrameBridge",
    "SomaIdentityRestPose",
    "SomaMotionReconstruction",
    "SomaReconstructionReport",
    "extract_chord_contact_targets",
    "infer_soma_frame_bridge",
    "invert_soma_global_rotations",
    "reconstruct_soma_motion",
    "reconstruct_soma_identity_rest_pose",
    "sample_object_surface",
    "vertex_normals",
]
