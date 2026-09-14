"""Validated robot, object, and contact references for dexterous imitation.

The on-disk format is an NPZ that can be loaded with ``allow_pickle=False``.
It keeps the historical ``qpos``, ``qvel``, ``joint_names``, and ``fps`` keys
used by the Isaac runtime.  Added arrays carry the fixed robot root, two named
wrist-frame poses, object poses and twists, and optional padded contact slots.
All quaternions use WXYZ order, all poses use
``[x, y, z, qw, qx, qy, qz]``, and object twists use world-frame
``[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

import numpy as np


REFERENCE_SCHEMA_VERSION = "iltools_dexterous_reference/v2"
LEGACY_REFERENCE_SCHEMA_VERSION = "iltools_dexterous_reference/v1"
MANIFEST_SCHEMA_VERSION = "iltools_dexterous_reference_manifest/v1"
TRAINING_QUALIFICATION_SCHEMA_VERSION = "iltools_dexterous_training_qualification/v2"
_LEGACY_TRAINING_QUALIFICATION_SCHEMA_VERSION = (
    "iltools_dexterous_training_qualification/v1"
)
MAX_TRAINING_PENETRATION_TOLERANCE_M = 1.0e-3
"""Largest collision-audit tolerance accepted at the training boundary."""
_HAND_SIDES = frozenset({"left", "right"})
_COLLISION_ASSET_ROLES = frozenset({"object", "support_surface"})
_DIRECT_COLLISION_ASSET_EXTENSIONS = frozenset(
    {".dae", ".glb", ".obj", ".off", ".ply", ".stl"}
)
_UNAVAILABLE_PROVENANCE_MARKERS = ("placeholder", "unavailable")
_USD_ASSET_TOKEN = re.compile(r"@[^@\r\n]+@")


def _string_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{name} must not contain empty names.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique names.")
    return result


def _array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type[Any],
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}.")
    if np.issubdtype(result.dtype, np.number) and not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def _validate_pose_quaternions(value: np.ndarray, *, name: str) -> None:
    if value.shape[-1] != 7:
        raise ValueError(f"{name} must contain seven-value poses.")
    quaternions = value[..., 3:7].reshape(-1, 4)
    if len(quaternions) == 0:
        return
    norms = np.linalg.norm(quaternions, axis=-1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-4):
        raise ValueError(f"{name} must contain unit WXYZ quaternions.")


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply arrays of WXYZ quaternions with broadcast-compatible shapes."""

    left_w = left[..., :1]
    right_w = right[..., :1]
    left_v = left[..., 1:4]
    right_v = right[..., 1:4]
    return np.concatenate(
        (
            left_w * right_w - np.sum(left_v * right_v, axis=-1, keepdims=True),
            left_w * right_v + right_w * left_v + np.cross(left_v, right_v),
        ),
        axis=-1,
    )


def derive_object_twists_w(object_poses_w: np.ndarray, fps: float) -> np.ndarray:
    """Derive deterministic world-frame object twists from a pose sequence.

    Args:
        object_poses_w: Pose samples with shape ``[T, O, 7]`` in XYZ+WXYZ
            convention.  Quaternion signs may change between samples.
        fps: Positive sampling rate in frames per second.

    Returns:
        An ``float32`` array with shape ``[T, O, 6]``.  The final dimension is
        world-frame linear XYZ followed by world-frame angular XYZ.  Interior
        samples average the two adjacent interval velocities; endpoints use
        their sole adjacent interval.  Quaternion deltas are mapped through
        the shortest rotation, making the result invariant to quaternion sign.
    """

    poses = np.asarray(object_poses_w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[-1] != 7 or poses.shape[0] < 2:
        raise ValueError("object_poses_w must have shape [T, O, 7] with T >= 2.")
    if not np.isfinite(poses).all():
        raise ValueError("object_poses_w contains non-finite values.")
    sample_rate = float(fps)
    if not np.isfinite(sample_rate) or sample_rate <= 0.0:
        raise ValueError("fps must be finite and positive.")

    quaternions = poses[..., 3:7].copy()
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("object_poses_w contains an invalid quaternion.")
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1.0e-4):
        raise ValueError("object_poses_w must contain unit WXYZ quaternions.")
    quaternions /= norms

    # Choose one continuous representative from each q/-q pair before taking
    # differences.  The shortest-delta check below is retained for numerical
    # stability at a sign boundary and rotations close to pi.
    for frame in range(1, quaternions.shape[0]):
        flip = np.sum(quaternions[frame - 1] * quaternions[frame], axis=-1) < 0.0
        quaternions[frame, flip] *= -1.0

    linear_interval = np.diff(poses[..., :3], axis=0) * sample_rate
    previous_conjugate = quaternions[:-1].copy()
    previous_conjugate[..., 1:4] *= -1.0
    delta = _quaternion_multiply_wxyz(quaternions[1:], previous_conjugate)
    delta[delta[..., 0] < 0.0] *= -1.0
    delta /= np.linalg.norm(delta, axis=-1, keepdims=True)
    vector = delta[..., 1:4]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., 0], -1.0, 1.0))
    scale = np.empty_like(vector_norm)
    small = vector_norm <= 1.0e-10
    scale[small] = 2.0
    scale[~small] = angle[~small] / vector_norm[~small]
    angular_interval = vector * scale[..., None] * sample_rate

    interval_twists = np.concatenate((linear_interval, angular_interval), axis=-1)
    twists = np.empty((poses.shape[0], poses.shape[1], 6), dtype=np.float64)
    twists[0] = interval_twists[0]
    twists[-1] = interval_twists[-1]
    if poses.shape[0] > 2:
        twists[1:-1] = 0.5 * (interval_twists[:-1] + interval_twists[1:])
    return twists.astype(np.float32)


def _decode_string_array(value: np.ndarray) -> tuple[str, ...]:
    result: list[str] = []
    for item in np.asarray(value).reshape(-1).tolist():
        result.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return tuple(result)


MAX_ASSET_RELOCATION_DEPTH = 6
"""How far above a Reference file relocation looks for a moved scene asset."""


def _relocated_asset(recorded: Path, base_dir: Path, digest: str) -> Path | None:
    """Find a moved scene asset near ``base_dir`` whose content matches ``digest``.

    A Reference records absolute asset paths, so a set copied to another host
    (a cluster bind, another checkout) points at paths that no longer exist.
    This searches the Reference's own directory and a bounded number of its
    parents for the tail of the recorded path, longest tail first.

    A candidate is accepted only when its SHA-256 equals the recorded digest,
    so relocation never weakens the hash-bound contract: it changes where a
    file may be found, never whether its content is the declared one. Without
    a recorded digest there is nothing to verify against and the caller keeps
    the original path.
    """

    if not digest:
        return None
    roots = [base_dir, *list(base_dir.parents)[:MAX_ASSET_RELOCATION_DEPTH]]
    parts = recorded.parts
    # Longest tail first: the most specific match wins, so a bare basename
    # collision elsewhere in the tree cannot shadow the real asset.
    for depth in range(min(len(parts), MAX_ASSET_RELOCATION_DEPTH + 1), 0, -1):
        tail = Path(*parts[-depth:])
        for root in roots:
            candidate = root / tail
            if candidate.is_file() and sha256_file(candidate) == digest:
                return candidate.resolve()
    return None


def _resolve_asset_paths(
    values: Sequence[str],
    *,
    base_dir: Path,
    digests: Sequence[str] = (),
) -> tuple[str, ...]:
    result: list[str] = []
    for index, value in enumerate(values):
        if not value or "://" in value:
            result.append(value)
            continue
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (base_dir / path).resolve()
        if not resolved.is_file():
            digest = str(digests[index]).lower() if index < len(digests) else ""
            relocated = _relocated_asset(resolved, base_dir, digest)
            if relocated is not None:
                resolved = relocated
        result.append(str(resolved))
    return tuple(result)


def _asset_hash_tuple(
    values: Sequence[str],
    *,
    count: int,
    name: str,
) -> tuple[str, ...]:
    if not values:
        return ("",) * count
    result = tuple(str(value).lower() for value in values)
    if len(result) != count:
        raise ValueError(f"{name} must align with its asset names.")
    for value in result:
        if value and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} values must be empty or 64-digit SHA-256.")
    return result


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CollisionAssetDependency:
    """One hash-bound local collision file referenced by a scene asset.

    ``uri`` is the dependency path as declared by the owner asset. Relative
    values are resolved from the owner asset's directory, not from the NPZ.
    """

    asset_role: str
    asset_index: int
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        role = str(self.asset_role)
        if role not in _COLLISION_ASSET_ROLES:
            raise ValueError(
                "collision asset dependency asset_role must be 'object' or "
                "'support_surface'."
            )
        if isinstance(self.asset_index, (bool, np.bool_)) or not isinstance(
            self.asset_index, (int, np.integer)
        ):
            raise ValueError(
                "collision asset dependency asset_index must be an integer."
            )
        index = int(self.asset_index)
        if index < 0:
            raise ValueError(
                "collision asset dependency asset_index must be non-negative."
            )
        uri = str(self.uri).strip()
        if not uri:
            raise ValueError("collision asset dependency uri must be non-empty.")
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "collision asset dependency sha256 must be a 64-digit hex digest."
            )
        object.__setattr__(self, "asset_role", role)
        object.__setattr__(self, "asset_index", index)
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_role": self.asset_role,
            "asset_index": self.asset_index,
            "uri": self.uri,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CollisionAssetDependency:
        required = ("asset_role", "asset_index", "uri", "sha256")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                f"Collision asset dependency is missing fields: {missing}."
            )
        unexpected = sorted(set(payload) - set(required))
        if unexpected:
            raise ValueError(
                f"Collision asset dependency has unknown fields: {unexpected}."
            )
        return cls(**{name: payload[name] for name in required})


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _resolve_local_dependency_uri(owner_path: Path, uri: str) -> Path:
    if "%" in uri or "$" in uri or "\\" in uri or uri.startswith("~"):
        raise ValueError(
            "Collision dependency URI must be a literal local path without "
            "percent encoding, environment substitution, backslashes, or '~': "
            f"{uri!r}."
        )
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment:
        raise ValueError(
            f"Collision dependency URI must not have a query or fragment: {uri!r}."
        )
    if parsed.scheme:
        if parsed.scheme.casefold() != "file":
            raise ValueError(
                "Strict collision dependency verification requires local files; "
                f"unsupported URI {uri!r} in {owner_path}."
            )
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"Collision dependency file URI must be local: {uri!r}.")
        dependency = Path(unquote(parsed.path))
        if not dependency.is_absolute():
            raise ValueError(
                f"Collision dependency file URI must be absolute: {uri!r}."
            )
    else:
        if uri.startswith("//"):
            raise ValueError(
                f"Collision dependency network path is not local: {uri!r}."
            )
        dependency = Path(unquote(uri)).expanduser()
    if not dependency.is_absolute():
        dependency = owner_path.parent / dependency
    resolved = dependency.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Collision asset dependency is missing: {resolved} "
            f"(declared by {owner_path})."
        )
    return resolved


def _verify_self_contained_ascii_usd(path: Path) -> None:
    try:
        usd_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Strict USD verification requires ASCII USDA text: {path}."
        ) from exc
    normalized = usd_text.lstrip("\ufeff \t\r\n")
    if not normalized.startswith("#usda"):
        raise ValueError(
            f"Strict USD verification requires a USDA text header: {path}."
        )
    if _USD_ASSET_TOKEN.search(usd_text):
        raise ValueError(
            "Strict collision dependency verification cannot prove "
            f"external-reference USD coverage without OpenUSD/pxr: {path}. "
            "Use a self-contained USDA or direct mesh."
        )


def _positive_urdf_values(
    value: str,
    *,
    count: int,
    attribute: str,
    path: Path,
) -> None:
    try:
        values = tuple(float(item) for item in value.split())
    except ValueError as exc:
        raise ValueError(
            f"URDF collision {attribute} in {path} must contain numbers."
        ) from exc
    if (
        len(values) != count
        or not np.isfinite(values).all()
        or any(item <= 0.0 for item in values)
    ):
        raise ValueError(
            f"URDF collision {attribute} in {path} must contain {count} "
            "positive finite values."
        )


def _urdf_collision_meshes(path: Path) -> tuple[tuple[str, Path], ...]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF collision asset {path}: {exc}.") from exc
    collisions = tuple(
        element
        for element in root.iter()
        if _xml_local_name(element.tag) == "collision"
    )
    if not collisions:
        raise ValueError(f"URDF asset has no collision geometry: {path}.")
    result: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for collision in collisions:
        geometries = tuple(
            child for child in collision if _xml_local_name(child.tag) == "geometry"
        )
        if len(geometries) != 1:
            raise ValueError(
                f"Each URDF collision in {path} must have exactly one geometry."
            )
        shapes = tuple(child for child in geometries[0] if isinstance(child.tag, str))
        if len(shapes) != 1:
            raise ValueError(
                f"Each URDF collision geometry in {path} must have one shape."
            )
        shape = shapes[0]
        shape_name = _xml_local_name(shape.tag)
        if shape_name == "mesh":
            uri = str(shape.attrib.get("filename", "")).strip()
            if not uri:
                raise ValueError(f"URDF collision mesh in {path} has no filename URI.")
            scale = str(shape.attrib.get("scale", "")).strip()
            if scale:
                _positive_urdf_values(
                    scale,
                    count=3,
                    attribute="mesh scale",
                    path=path,
                )
            dependency_path = _resolve_local_dependency_uri(path, uri)
            dependency_suffix = dependency_path.suffix.casefold()
            if dependency_suffix not in _DIRECT_COLLISION_ASSET_EXTENSIONS:
                raise ValueError(
                    "Strict URDF collision dependency verification does not "
                    f"support mesh format {dependency_suffix!r}: "
                    f"{dependency_path}."
                )
            if dependency_path not in seen_paths:
                seen_paths.add(dependency_path)
                result.append((uri, dependency_path))
        elif shape_name == "box":
            _positive_urdf_values(
                str(shape.attrib.get("size", "")),
                count=3,
                attribute="box size",
                path=path,
            )
        elif shape_name == "sphere":
            _positive_urdf_values(
                str(shape.attrib.get("radius", "")),
                count=1,
                attribute="sphere radius",
                path=path,
            )
        elif shape_name == "cylinder":
            _positive_urdf_values(
                str(shape.attrib.get("radius", "")),
                count=1,
                attribute="cylinder radius",
                path=path,
            )
            _positive_urdf_values(
                str(shape.attrib.get("length", "")),
                count=1,
                attribute="cylinder length",
                path=path,
            )
        else:
            raise ValueError(
                f"Unsupported URDF collision geometry {shape_name!r} in {path}."
            )
    return tuple(result)


def build_urdf_collision_asset_dependencies(
    path: str | Path,
    *,
    asset_role: str,
    asset_index: int,
) -> tuple[CollisionAssetDependency, ...]:
    """Parse and hash all local URDF ``collision/geometry/mesh`` files.

    Visual meshes, materials, and textures are intentionally outside this
    physics contract.
    """

    owner_path = Path(path).expanduser().resolve()
    if not owner_path.is_file():
        raise FileNotFoundError(f"URDF asset is missing: {owner_path}")
    if owner_path.suffix.casefold() != ".urdf":
        raise ValueError(f"Expected a .urdf asset, got {owner_path}.")
    return tuple(
        CollisionAssetDependency(
            asset_role=asset_role,
            asset_index=asset_index,
            uri=uri,
            sha256=sha256_file(dependency_path),
        )
        for uri, dependency_path in _urdf_collision_meshes(owner_path)
    )


def _require_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a Boolean.")
    return bool(value)


def _qualification_text(value: Any, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    lowered = result.casefold()
    if any(marker in lowered for marker in _UNAVAILABLE_PROVENANCE_MARKERS):
        raise ValueError(
            f"{name} must identify real evidence, not unavailable or placeholder data."
        )
    if lowered in {"none", "n/a", "na", "unknown"}:
        raise ValueError(f"{name} must identify real evidence.")
    return result


@dataclass(frozen=True, slots=True)
class CollisionClearanceQualification:
    """Evidence that collision and clearance checks passed for all frames."""

    qualified: bool
    method: str
    scope: str
    provenance: str
    checked_frame_count: int
    minimum_signed_distance_m: float
    penetration_tolerance_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "qualified",
            _require_bool(self.qualified, name="collision clearance qualified"),
        )
        for name in ("method", "scope", "provenance"):
            object.__setattr__(
                self,
                name,
                _qualification_text(
                    getattr(self, name), name=f"collision clearance {name}"
                ),
            )
        if isinstance(self.checked_frame_count, (bool, np.bool_)) or not isinstance(
            self.checked_frame_count, (int, np.integer)
        ):
            raise ValueError(
                "collision clearance checked_frame_count must be an integer."
            )
        checked_frame_count = int(self.checked_frame_count)
        if checked_frame_count <= 0:
            raise ValueError(
                "collision clearance checked_frame_count must be a positive integer."
            )
        object.__setattr__(self, "checked_frame_count", checked_frame_count)
        minimum_distance = float(self.minimum_signed_distance_m)
        tolerance = float(self.penetration_tolerance_m)
        if not np.isfinite(minimum_distance):
            raise ValueError(
                "collision clearance minimum_signed_distance_m must be finite."
            )
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "collision clearance penetration_tolerance_m must be finite and "
                "non-negative."
            )
        object.__setattr__(self, "minimum_signed_distance_m", minimum_distance)
        object.__setattr__(self, "penetration_tolerance_m", tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualified": self.qualified,
            "method": self.method,
            "scope": self.scope,
            "provenance": self.provenance,
            "checked_frame_count": self.checked_frame_count,
            "minimum_signed_distance_m": self.minimum_signed_distance_m,
            "penetration_tolerance_m": self.penetration_tolerance_m,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CollisionClearanceQualification:
        required = (
            "qualified",
            "method",
            "scope",
            "provenance",
            "checked_frame_count",
            "minimum_signed_distance_m",
            "penetration_tolerance_m",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "Training qualification collision_clearance is missing fields: "
                f"{missing}."
            )
        return cls(**{name: payload[name] for name in required})


@dataclass(frozen=True, slots=True)
class TrainingQualification:
    """Typed, fail-closed evidence that a Reference can be used for training."""

    runtime_qualified: bool
    isaac_runtime_qualified: bool
    inspection_only: bool
    contact_geometry_provenance: str
    collision_clearance: CollisionClearanceQualification
    schema_version: str = TRAINING_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            _LEGACY_TRAINING_QUALIFICATION_SCHEMA_VERSION,
            TRAINING_QUALIFICATION_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"Unsupported training qualification schema {self.schema_version!r}; "
                "expected a supported v1 inspection record or the current "
                f"{TRAINING_QUALIFICATION_SCHEMA_VERSION!r} training record."
            )
        for name in (
            "runtime_qualified",
            "isaac_runtime_qualified",
            "inspection_only",
        ):
            object.__setattr__(
                self,
                name,
                _require_bool(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "contact_geometry_provenance",
            _qualification_text(
                self.contact_geometry_provenance,
                name="contact_geometry_provenance",
            ),
        )
        if not isinstance(self.collision_clearance, CollisionClearanceQualification):
            raise ValueError(
                "collision_clearance must be a CollisionClearanceQualification."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_qualified": self.runtime_qualified,
            "isaac_runtime_qualified": self.isaac_runtime_qualified,
            "inspection_only": self.inspection_only,
            "contact_geometry_provenance": self.contact_geometry_provenance,
            "collision_clearance": self.collision_clearance.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrainingQualification:
        required = (
            "schema_version",
            "runtime_qualified",
            "isaac_runtime_qualified",
            "inspection_only",
            "contact_geometry_provenance",
            "collision_clearance",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Training qualification is missing fields: {missing}.")
        collision_payload = payload["collision_clearance"]
        if not isinstance(collision_payload, Mapping):
            raise ValueError(
                "Training qualification collision_clearance must be an object."
            )
        return cls(
            schema_version=str(payload["schema_version"]),
            runtime_qualified=payload["runtime_qualified"],
            isaac_runtime_qualified=payload["isaac_runtime_qualified"],
            inspection_only=payload["inspection_only"],
            contact_geometry_provenance=str(payload["contact_geometry_provenance"]),
            collision_clearance=CollisionClearanceQualification.from_dict(
                collision_payload
            ),
        )


@dataclass(slots=True)
class ScenePhysics:
    """Explicit rigid-object and support material properties for one scene.

    Object arrays align with ``object_names``. The center of mass is in the
    scaled runtime object's local frame. Diagonal inertia is about that center
    of mass in the same local axes. Support arrays align with
    ``support_surface_names``.
    """

    object_mass_kg: np.ndarray
    object_center_of_mass_m: np.ndarray
    object_diagonal_inertia_kg_m2: np.ndarray
    object_static_friction: np.ndarray
    object_dynamic_friction: np.ndarray
    object_restitution: np.ndarray
    support_static_friction: np.ndarray
    support_dynamic_friction: np.ndarray
    support_restitution: np.ndarray

    def validate(self, *, object_count: int, support_count: int) -> None:
        self.object_mass_kg = _array(
            self.object_mass_kg,
            dtype=np.float32,
            name="scene physics object_mass_kg",
            shape=(object_count,),
        )
        self.object_center_of_mass_m = _array(
            self.object_center_of_mass_m,
            dtype=np.float32,
            name="scene physics object_center_of_mass_m",
            shape=(object_count, 3),
        )
        self.object_diagonal_inertia_kg_m2 = _array(
            self.object_diagonal_inertia_kg_m2,
            dtype=np.float32,
            name="scene physics object_diagonal_inertia_kg_m2",
            shape=(object_count, 3),
        )
        for name in (
            "object_static_friction",
            "object_dynamic_friction",
            "object_restitution",
        ):
            setattr(
                self,
                name,
                _array(
                    getattr(self, name),
                    dtype=np.float32,
                    name=f"scene physics {name}",
                    shape=(object_count,),
                ),
            )
        for name in (
            "support_static_friction",
            "support_dynamic_friction",
            "support_restitution",
        ):
            setattr(
                self,
                name,
                _array(
                    getattr(self, name),
                    dtype=np.float32,
                    name=f"scene physics {name}",
                    shape=(support_count,),
                ),
            )
        if np.any(self.object_mass_kg <= 0.0):
            raise ValueError("scene physics object_mass_kg must be positive.")
        inertia = self.object_diagonal_inertia_kg_m2
        if np.any(inertia <= 0.0):
            raise ValueError(
                "scene physics object_diagonal_inertia_kg_m2 must be positive."
            )
        if np.any(2.0 * inertia > np.sum(inertia, axis=-1, keepdims=True) + 1.0e-9):
            raise ValueError(
                "scene physics diagonal inertia must satisfy the triangle inequality."
            )
        for prefix in ("object", "support"):
            static = getattr(self, f"{prefix}_static_friction")
            dynamic = getattr(self, f"{prefix}_dynamic_friction")
            restitution = getattr(self, f"{prefix}_restitution")
            if np.any(static < 0.0) or np.any(dynamic < 0.0):
                raise ValueError(
                    f"scene physics {prefix} friction must be non-negative."
                )
            if np.any(dynamic > static):
                raise ValueError(
                    f"scene physics {prefix} dynamic friction must not exceed "
                    "static friction."
                )
            if np.any(restitution < 0.0) or np.any(restitution > 1.0):
                raise ValueError(
                    f"scene physics {prefix} restitution must be in [0, 1]."
                )

    def to_npz_fields(self) -> dict[str, np.ndarray]:
        return {
            f"scene_physics_{name}": getattr(self, name)
            for name in (
                "object_mass_kg",
                "object_center_of_mass_m",
                "object_diagonal_inertia_kg_m2",
                "object_static_friction",
                "object_dynamic_friction",
                "object_restitution",
                "support_static_friction",
                "support_dynamic_friction",
                "support_restitution",
            )
        }

    @classmethod
    def from_npz(cls, data: Mapping[str, np.ndarray]) -> ScenePhysics | None:
        names = (
            "object_mass_kg",
            "object_center_of_mass_m",
            "object_diagonal_inertia_kg_m2",
            "object_static_friction",
            "object_dynamic_friction",
            "object_restitution",
            "support_static_friction",
            "support_dynamic_friction",
            "support_restitution",
        )
        keys = tuple(f"scene_physics_{name}" for name in names)
        present = [key for key in keys if key in data]
        if not present:
            return None
        missing = [key for key in keys if key not in data]
        if missing:
            raise ValueError(
                f"Reference NPZ has an incomplete scene physics group: {missing}."
            )
        return cls(**{name: np.asarray(data[key]) for name, key in zip(names, keys)})


@dataclass(slots=True)
class ContactSequence:
    """Padded hand-object contacts with a fixed slot count.

    The vector arrays have shape ``[T, S, C, 3]``.  ``T`` is the frame count,
    ``S`` is the number of hand sides, and ``C`` is the contact slot count.
    ``link_names`` has shape ``[S, C]``.  ``object_indices`` and ``active``
    have shape ``[T, S, C]``.  Inactive slots can use object index ``-1`` and
    zero vectors.
    """

    hand_sides: tuple[str, ...]
    link_names: np.ndarray
    link_positions_w: np.ndarray
    link_normals_w: np.ndarray
    object_positions_w: np.ndarray
    object_normals_w: np.ndarray
    object_indices: np.ndarray
    active: np.ndarray

    def __post_init__(self) -> None:
        self.hand_sides = _string_tuple(self.hand_sides, name="contact hand_sides")
        unknown = set(self.hand_sides) - _HAND_SIDES
        if unknown:
            raise ValueError(f"Unknown contact hand sides: {sorted(unknown)}.")
        link_names = np.asarray(self.link_names)
        if link_names.ndim != 2 or link_names.shape[0] != len(self.hand_sides):
            raise ValueError(
                "contact link_names must have shape [hand sides, contact slots]."
            )
        self.link_names = np.asarray(link_names, dtype=np.str_).copy()
        for side, row in zip(self.hand_sides, self.link_names, strict=True):
            names = [str(name) for name in row if str(name)]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    "contact link_names must be unique within each hand; "
                    f"{side} duplicates: {duplicates}."
                )

    @property
    def slot_count(self) -> int:
        return int(self.link_names.shape[1])

    def validate(self, *, frame_count: int, object_count: int) -> None:
        side_count = len(self.hand_sides)
        slot_count = self.slot_count
        vector_shape = (frame_count, side_count, slot_count, 3)
        index_shape = (frame_count, side_count, slot_count)
        self.link_positions_w = _array(
            self.link_positions_w,
            dtype=np.float32,
            name="contact link_positions_w",
            shape=vector_shape,
        )
        self.link_normals_w = _array(
            self.link_normals_w,
            dtype=np.float32,
            name="contact link_normals_w",
            shape=vector_shape,
        )
        self.object_positions_w = _array(
            self.object_positions_w,
            dtype=np.float32,
            name="contact object_positions_w",
            shape=vector_shape,
        )
        self.object_normals_w = _array(
            self.object_normals_w,
            dtype=np.float32,
            name="contact object_normals_w",
            shape=vector_shape,
        )
        self.object_indices = _array(
            self.object_indices,
            dtype=np.int32,
            name="contact object_indices",
            shape=index_shape,
        )
        self.active = _array(
            self.active,
            dtype=np.bool_,
            name="contact active",
            shape=index_shape,
        )
        if np.any(self.active):
            active_indices = self.object_indices[self.active]
            if (
                object_count == 0
                or np.any(active_indices < 0)
                or np.any(active_indices >= object_count)
            ):
                raise ValueError(
                    "Active contact object_indices must identify a declared object."
                )
            expanded_names = np.broadcast_to(
                self.link_names[None, ...], self.active.shape
            )
            if np.any(expanded_names[self.active] == ""):
                raise ValueError("Each active contact slot must have a link name.")
            for name, normals in (
                ("link_normals_w", self.link_normals_w),
                ("object_normals_w", self.object_normals_w),
            ):
                if np.any(np.linalg.norm(normals[self.active], axis=-1) <= 1.0e-8):
                    raise ValueError(f"Active contact {name} must be non-zero.")

    def to_npz_fields(self) -> dict[str, np.ndarray]:
        return {
            "contact_hand_sides": np.asarray(self.hand_sides),
            "contact_link_names": self.link_names,
            "contact_link_positions_w": self.link_positions_w,
            "contact_link_normals_w": self.link_normals_w,
            "contact_object_positions_w": self.object_positions_w,
            "contact_object_normals_w": self.object_normals_w,
            "contact_object_indices": self.object_indices,
            "contact_active": self.active,
        }

    @classmethod
    def from_npz(cls, data: Mapping[str, np.ndarray]) -> ContactSequence | None:
        if "contact_active" not in data:
            return None
        required = (
            "contact_hand_sides",
            "contact_link_names",
            "contact_link_positions_w",
            "contact_link_normals_w",
            "contact_object_positions_w",
            "contact_object_normals_w",
            "contact_object_indices",
            "contact_active",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                f"Reference NPZ has an incomplete contact group: {missing}."
            )
        return cls(
            hand_sides=_decode_string_array(data["contact_hand_sides"]),
            link_names=np.asarray(data["contact_link_names"]),
            link_positions_w=np.asarray(data["contact_link_positions_w"]),
            link_normals_w=np.asarray(data["contact_link_normals_w"]),
            object_positions_w=np.asarray(data["contact_object_positions_w"]),
            object_normals_w=np.asarray(data["contact_object_normals_w"]),
            object_indices=np.asarray(data["contact_object_indices"]),
            active=np.asarray(data["contact_active"]),
        )


@dataclass(slots=True)
class DexterousReference:
    """One robot reference with hands, objects, and contacts.

    ``robot_layout`` distinguishes the historical fixed-base robot layout from
    the dual free-floating hand layout used by Sharpa V2D.  In the dual-hand
    layout, ``qpos`` and ``qvel`` contain all left finger joints followed by all
    right finger joints.  Wrist motion is stored separately as poses and
    world-frame twists.
    """

    sequence_id: str
    robot_name: str
    fps: float
    joint_names: tuple[str, ...]
    qpos: np.ndarray
    fixed_root_pose_w: np.ndarray
    left_wrist_pose_w: np.ndarray
    right_wrist_pose_w: np.ndarray
    left_wrist_frame_name: str
    right_wrist_frame_name: str
    object_names: tuple[str, ...]
    object_poses_w: np.ndarray
    robot_layout: str = "fixed_base"
    left_joint_names: tuple[str, ...] = ()
    right_joint_names: tuple[str, ...] = ()
    left_wrist_twist_w: np.ndarray | None = None
    right_wrist_twist_w: np.ndarray | None = None
    qvel: np.ndarray | None = None
    object_twists_w: np.ndarray | None = None
    object_asset_paths: tuple[str, ...] = ()
    object_asset_sha256: tuple[str, ...] = ()
    object_scales: np.ndarray | None = None
    object_radii: np.ndarray | None = None
    left_hand_frame_names: tuple[str, ...] = ()
    left_hand_frame_poses_w: np.ndarray | None = None
    right_hand_frame_names: tuple[str, ...] = ()
    right_hand_frame_poses_w: np.ndarray | None = None
    support_surface_names: tuple[str, ...] = ()
    support_surface_asset_paths: tuple[str, ...] = ()
    support_surface_asset_sha256: tuple[str, ...] = ()
    support_surface_scales: np.ndarray | None = None
    support_surface_poses_w: np.ndarray | None = None
    scene_physics: ScenePhysics | None = None
    contacts: ContactSequence | None = None
    training_qualification: TrainingQualification | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, repr=False)
    schema_version: str = REFERENCE_SCHEMA_VERSION
    collision_asset_dependencies: tuple[CollisionAssetDependency, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported reference schema {self.schema_version!r}; "
                f"expected {REFERENCE_SCHEMA_VERSION!r}."
            )
        if self.robot_layout not in {"fixed_base", "dual_floating_hand"}:
            raise ValueError(
                "robot_layout must be 'fixed_base' or 'dual_floating_hand'."
            )
        if not self.sequence_id or not self.robot_name:
            raise ValueError("sequence_id and robot_name must be non-empty.")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive.")
        self.fps = float(self.fps)
        self.joint_names = _string_tuple(self.joint_names, name="joint_names")
        qpos = np.asarray(self.qpos)
        if qpos.ndim != 2 or qpos.shape[0] < 2:
            raise ValueError("qpos must have shape [frames, joints] with two frames.")
        frame_count, joint_count = qpos.shape
        if joint_count != len(self.joint_names):
            raise ValueError(
                f"qpos has {joint_count} joints but joint_names has "
                f"{len(self.joint_names)} names."
            )
        self.qpos = _array(
            qpos,
            dtype=np.float32,
            name="qpos",
            shape=(frame_count, joint_count),
        )
        if self.qvel is None:
            qvel = np.zeros_like(self.qpos)
            qvel[1:] = np.diff(self.qpos, axis=0) * self.fps
            qvel[0] = qvel[1]
            self.qvel = qvel
        else:
            self.qvel = _array(
                self.qvel,
                dtype=np.float32,
                name="qvel",
                shape=(frame_count, joint_count),
            )

        self.left_joint_names = _string_tuple(
            self.left_joint_names, name="left_joint_names"
        )
        self.right_joint_names = _string_tuple(
            self.right_joint_names, name="right_joint_names"
        )
        if self.robot_layout == "dual_floating_hand":
            if not self.left_joint_names or not self.right_joint_names:
                raise ValueError(
                    "dual_floating_hand references require left_joint_names and "
                    "right_joint_names."
                )
            expected_joint_names = self.left_joint_names + self.right_joint_names
            if self.joint_names != expected_joint_names:
                raise ValueError(
                    "dual_floating_hand joint_names must be left_joint_names "
                    "followed by right_joint_names."
                )

        self.fixed_root_pose_w = _array(
            self.fixed_root_pose_w,
            dtype=np.float32,
            name="fixed_root_pose_w",
            shape=(7,),
        )
        self.left_wrist_pose_w = _array(
            self.left_wrist_pose_w,
            dtype=np.float32,
            name="left_wrist_pose_w",
            shape=(frame_count, 7),
        )
        self.right_wrist_pose_w = _array(
            self.right_wrist_pose_w,
            dtype=np.float32,
            name="right_wrist_pose_w",
            shape=(frame_count, 7),
        )
        if self.left_wrist_twist_w is None:
            self.left_wrist_twist_w = derive_object_twists_w(
                self.left_wrist_pose_w[:, None, :], self.fps
            )[:, 0, :]
        else:
            self.left_wrist_twist_w = _array(
                self.left_wrist_twist_w,
                dtype=np.float32,
                name="left_wrist_twist_w",
                shape=(frame_count, 6),
            )
        if self.right_wrist_twist_w is None:
            self.right_wrist_twist_w = derive_object_twists_w(
                self.right_wrist_pose_w[:, None, :], self.fps
            )[:, 0, :]
        else:
            self.right_wrist_twist_w = _array(
                self.right_wrist_twist_w,
                dtype=np.float32,
                name="right_wrist_twist_w",
                shape=(frame_count, 6),
            )
        wrist_frame_names = _string_tuple(
            (self.left_wrist_frame_name, self.right_wrist_frame_name),
            name="wrist frame names",
        )
        self.left_wrist_frame_name, self.right_wrist_frame_name = wrist_frame_names
        self.object_names = _string_tuple(self.object_names, name="object_names")
        object_count = len(self.object_names)
        self.object_poses_w = _array(
            self.object_poses_w,
            dtype=np.float32,
            name="object_poses_w",
            shape=(frame_count, object_count, 7),
        )
        if self.object_twists_w is None:
            self.object_twists_w = derive_object_twists_w(
                self.object_poses_w,
                self.fps,
            )
        else:
            self.object_twists_w = _array(
                self.object_twists_w,
                dtype=np.float32,
                name="object_twists_w",
                shape=(frame_count, object_count, 6),
            )
        if not self.object_asset_paths:
            self.object_asset_paths = ("",) * object_count
        else:
            self.object_asset_paths = tuple(
                str(value) for value in self.object_asset_paths
            )
        if len(self.object_asset_paths) != object_count:
            raise ValueError("object_asset_paths must align with object_names.")
        self.object_asset_sha256 = _asset_hash_tuple(
            self.object_asset_sha256,
            count=object_count,
            name="object_asset_sha256",
        )
        if self.object_scales is None:
            self.object_scales = np.ones((object_count, 3), dtype=np.float32)
        else:
            self.object_scales = _array(
                self.object_scales,
                dtype=np.float32,
                name="object_scales",
                shape=(object_count, 3),
            )
        if np.any(self.object_scales <= 0.0):
            raise ValueError("object_scales must be positive.")
        if self.object_radii is None:
            self.object_radii = np.zeros(object_count, dtype=np.float32)
        else:
            self.object_radii = _array(
                self.object_radii,
                dtype=np.float32,
                name="object_radii",
                shape=(object_count,),
            )
        if np.any(self.object_radii < 0.0):
            raise ValueError("object_radii must be non-negative.")

        self.left_hand_frame_names = _string_tuple(
            self.left_hand_frame_names, name="left_hand_frame_names"
        )
        self.right_hand_frame_names = _string_tuple(
            self.right_hand_frame_names, name="right_hand_frame_names"
        )
        if self.left_hand_frame_poses_w is None:
            self.left_hand_frame_poses_w = np.empty(
                (frame_count, len(self.left_hand_frame_names), 7), dtype=np.float32
            )
        else:
            self.left_hand_frame_poses_w = _array(
                self.left_hand_frame_poses_w,
                dtype=np.float32,
                name="left_hand_frame_poses_w",
                shape=(frame_count, len(self.left_hand_frame_names), 7),
            )
        if self.right_hand_frame_poses_w is None:
            self.right_hand_frame_poses_w = np.empty(
                (frame_count, len(self.right_hand_frame_names), 7), dtype=np.float32
            )
        else:
            self.right_hand_frame_poses_w = _array(
                self.right_hand_frame_poses_w,
                dtype=np.float32,
                name="right_hand_frame_poses_w",
                shape=(frame_count, len(self.right_hand_frame_names), 7),
            )

        self.support_surface_names = _string_tuple(
            self.support_surface_names, name="support_surface_names"
        )
        surface_count = len(self.support_surface_names)
        if not self.support_surface_asset_paths:
            self.support_surface_asset_paths = ("",) * surface_count
        else:
            self.support_surface_asset_paths = tuple(
                str(value) for value in self.support_surface_asset_paths
            )
        if len(self.support_surface_asset_paths) != surface_count:
            raise ValueError(
                "support_surface_asset_paths must align with support_surface_names."
            )
        self.support_surface_asset_sha256 = _asset_hash_tuple(
            self.support_surface_asset_sha256,
            count=surface_count,
            name="support_surface_asset_sha256",
        )
        dependencies = tuple(self.collision_asset_dependencies)
        if any(
            not isinstance(dependency, CollisionAssetDependency)
            for dependency in dependencies
        ):
            raise ValueError(
                "collision_asset_dependencies must contain "
                "CollisionAssetDependency values."
            )
        self.collision_asset_dependencies = tuple(
            sorted(
                dependencies,
                key=lambda dependency: (
                    dependency.asset_role,
                    dependency.asset_index,
                    dependency.uri,
                ),
            )
        )
        dependency_keys: set[tuple[str, int, str]] = set()
        for dependency in self.collision_asset_dependencies:
            asset_count = (
                object_count if dependency.asset_role == "object" else surface_count
            )
            if dependency.asset_index >= asset_count:
                raise ValueError(
                    "collision asset dependency index is outside its "
                    f"{dependency.asset_role} asset group."
                )
            key = (
                dependency.asset_role,
                dependency.asset_index,
                dependency.uri,
            )
            if key in dependency_keys:
                raise ValueError(
                    "collision_asset_dependencies must not contain duplicates."
                )
            dependency_keys.add(key)
        if self.support_surface_scales is None:
            self.support_surface_scales = np.ones((surface_count, 3), dtype=np.float32)
        else:
            self.support_surface_scales = _array(
                self.support_surface_scales,
                dtype=np.float32,
                name="support_surface_scales",
                shape=(surface_count, 3),
            )
        if np.any(self.support_surface_scales <= 0.0):
            raise ValueError("support_surface_scales must be positive.")
        if self.support_surface_poses_w is None:
            self.support_surface_poses_w = np.empty(
                (surface_count, 7), dtype=np.float32
            )
        else:
            self.support_surface_poses_w = _array(
                self.support_surface_poses_w,
                dtype=np.float32,
                name="support_surface_poses_w",
                shape=(surface_count, 7),
            )

        _validate_pose_quaternions(
            self.fixed_root_pose_w[None, :], name="fixed_root_pose_w"
        )
        _validate_pose_quaternions(self.left_wrist_pose_w, name="left_wrist_pose_w")
        _validate_pose_quaternions(self.right_wrist_pose_w, name="right_wrist_pose_w")
        _validate_pose_quaternions(self.object_poses_w, name="object_poses_w")
        _validate_pose_quaternions(
            self.left_hand_frame_poses_w, name="left_hand_frame_poses_w"
        )
        _validate_pose_quaternions(
            self.right_hand_frame_poses_w, name="right_hand_frame_poses_w"
        )
        _validate_pose_quaternions(
            self.support_surface_poses_w, name="support_surface_poses_w"
        )
        if self.contacts is not None:
            self.contacts.validate(frame_count=frame_count, object_count=object_count)
        if self.scene_physics is not None:
            if not isinstance(self.scene_physics, ScenePhysics):
                raise ValueError("scene_physics must be a ScenePhysics or None.")
            self.scene_physics.validate(
                object_count=object_count,
                support_count=surface_count,
            )
        if self.training_qualification is not None and not isinstance(
            self.training_qualification, TrainingQualification
        ):
            raise ValueError(
                "training_qualification must be a TrainingQualification or None."
            )
        try:
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "metadata must contain JSON-compatible finite values."
            ) from exc

    @property
    def frame_count(self) -> int:
        return int(self.qpos.shape[0])

    def verify_scene_assets(
        self,
        *,
        require_hashes: bool = False,
        require_collision_dependencies: bool = False,
    ) -> None:
        """Verify local scene files and declared collision dependencies.

        Strict collision-dependency verification parses URDF collision meshes.
        It accepts self-contained ASCII USDA files. External-reference USD and
        binary/package USD require OpenUSD dependency discovery and are rejected.
        """

        groups = (
            (
                "object",
                self.object_asset_paths,
                self.object_asset_sha256,
            ),
            (
                "support surface",
                self.support_surface_asset_paths,
                self.support_surface_asset_sha256,
            ),
        )
        for role, paths, hashes in groups:
            dependency_role = role.replace(" ", "_")
            for asset_index, (path_value, expected_hash) in enumerate(
                zip(paths, hashes, strict=True)
            ):
                if not path_value:
                    raise ValueError(f"Each {role} must declare an asset path.")
                if "://" in path_value:
                    if require_hashes:
                        raise ValueError(
                            f"Strict scene verification requires a local {role} "
                            f"asset, got {path_value!r}."
                        )
                    continue
                path = Path(path_value).expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"{role.title()} asset is missing: {path}")
                if not expected_hash:
                    if require_hashes:
                        raise ValueError(
                            f"{role.title()} asset has no declared SHA-256: {path}"
                        )
                    continue
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"{role.title()} asset hash mismatch for {path}: expected "
                        f"{expected_hash}, got {actual_hash}."
                    )
                owner_dependencies = tuple(
                    dependency
                    for dependency in self.collision_asset_dependencies
                    if dependency.asset_role == dependency_role
                    and dependency.asset_index == asset_index
                )
                resolved_dependencies: dict[Path, CollisionAssetDependency] = {}
                for dependency in owner_dependencies:
                    dependency_path = _resolve_local_dependency_uri(
                        path, dependency.uri
                    )
                    if dependency_path in resolved_dependencies:
                        raise ValueError(
                            f"Collision dependencies for {path} resolve to the same "
                            f"file more than once: {dependency_path}."
                        )
                    resolved_dependencies[dependency_path] = dependency
                    actual_dependency_hash = sha256_file(dependency_path)
                    if actual_dependency_hash != dependency.sha256:
                        raise ValueError(
                            "Collision asset dependency hash mismatch for "
                            f"{dependency_path}: expected {dependency.sha256}, got "
                            f"{actual_dependency_hash}."
                        )

                if not require_collision_dependencies:
                    continue
                suffix = path.suffix.casefold()
                if suffix == ".urdf":
                    discovered = {
                        dependency_path: uri
                        for uri, dependency_path in _urdf_collision_meshes(path)
                    }
                    missing = [
                        uri
                        for dependency_path, uri in discovered.items()
                        if dependency_path not in resolved_dependencies
                    ]
                    extra = [
                        dependency.uri
                        for dependency_path, dependency in resolved_dependencies.items()
                        if dependency_path not in discovered
                    ]
                    if missing or extra:
                        raise ValueError(
                            f"URDF collision dependency declarations for {path} "
                            f"do not match its collision meshes; missing={missing}, "
                            f"extra={extra}."
                        )
                elif suffix in {".usd", ".usda"}:
                    _verify_self_contained_ascii_usd(path)
                    if owner_dependencies:
                        raise ValueError(
                            "Strict collision dependency verification cannot prove "
                            f"external-reference USD coverage without OpenUSD/pxr: "
                            f"{path}. Use a self-contained USDA or direct mesh."
                        )
                elif suffix in {".usdc", ".usdz"}:
                    raise ValueError(
                        "Strict collision dependency verification cannot inspect "
                        f"binary or package USD without OpenUSD/pxr: {path}."
                    )
                else:
                    if suffix not in _DIRECT_COLLISION_ASSET_EXTENSIONS:
                        raise ValueError(
                            "Strict collision dependency verification does not "
                            f"support scene asset format {suffix!r}: {path}."
                        )
                    if owner_dependencies:
                        raise ValueError(
                            "Direct collision mesh assets must not declare external "
                            f"collision dependencies: {path}."
                        )


def verify_training_qualification(
    reference: DexterousReference,
) -> TrainingQualification:
    """Verify that one Reference has complete training evidence.

    General NPZ loading stays backward compatible. This function is the strict
    boundary for training consumers and manifest writers.
    """

    qualification = reference.training_qualification
    if qualification is None:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no typed training "
            "qualification. Free-form metadata does not qualify a Reference."
        )
    if qualification.schema_version != TRAINING_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError(
            f"Reference {reference.sequence_id!r} uses legacy training "
            f"qualification {qualification.schema_version!r}. It can be loaded "
            "for inspection, but its dependency-unaware record cannot qualify "
            "for training."
        )
    if not qualification.runtime_qualified:
        raise ValueError(
            f"Reference {reference.sequence_id!r} is not runtime_qualified."
        )
    if not qualification.isaac_runtime_qualified:
        raise ValueError(
            f"Reference {reference.sequence_id!r} is not isaac_runtime_qualified."
        )
    if qualification.inspection_only:
        raise ValueError(
            f"Reference {reference.sequence_id!r} is inspection_only and cannot "
            "be used for training."
        )

    for name, expected in (
        ("runtime_qualified", qualification.runtime_qualified),
        ("isaac_runtime_qualified", qualification.isaac_runtime_qualified),
        ("inspection_only", qualification.inspection_only),
    ):
        if name in reference.metadata and reference.metadata[name] is not expected:
            raise ValueError(
                f"Reference {reference.sequence_id!r} metadata {name} conflicts "
                "with its typed training qualification."
            )

    reference.verify_scene_assets(
        require_hashes=True,
        require_collision_dependencies=True,
    )
    if reference.scene_physics is None:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no typed ScenePhysics; "
            "training must not use simulator defaults."
        )
    reference.scene_physics.validate(
        object_count=len(reference.object_names),
        support_count=len(reference.support_surface_names),
    )
    contacts = reference.contacts
    if contacts is None:
        raise ValueError(f"Reference {reference.sequence_id!r} has no ContactSequence.")
    contacts.validate(
        frame_count=reference.frame_count,
        object_count=len(reference.object_names),
    )
    if not np.any(contacts.active):
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no active contact slot."
        )
    active = contacts.active
    for name, values in (
        ("link_positions_w", contacts.link_positions_w),
        ("object_positions_w", contacts.object_positions_w),
    ):
        active_values = values[active]
        if not np.isfinite(active_values).all() or np.any(
            np.linalg.norm(active_values, axis=-1) <= 1.0e-8
        ):
            raise ValueError(
                f"Reference {reference.sequence_id!r} active contact {name} must "
                "contain finite, non-zero geometry."
            )
    for name, values in (
        ("link_normals_w", contacts.link_normals_w),
        ("object_normals_w", contacts.object_normals_w),
    ):
        norms = np.linalg.norm(values[active], axis=-1)
        if not np.isfinite(norms).all() or not np.allclose(
            norms, 1.0, rtol=0.0, atol=1.0e-3
        ):
            raise ValueError(
                f"Reference {reference.sequence_id!r} active contact {name} must "
                "contain finite unit normals."
            )
    active_indices = contacts.object_indices[active]
    if np.any(active_indices < 0) or np.any(
        active_indices >= len(reference.object_names)
    ):
        raise ValueError(
            f"Reference {reference.sequence_id!r} active contact object_indices "
            "must identify declared objects."
        )

    collision = qualification.collision_clearance
    if not collision.qualified:
        raise ValueError(
            f"Reference {reference.sequence_id!r} has no passing collision and "
            "clearance qualification."
        )
    if (
        collision.penetration_tolerance_m
        > MAX_TRAINING_PENETRATION_TOLERANCE_M + 1.0e-12
    ):
        raise ValueError(
            f"Reference {reference.sequence_id!r} collision and clearance "
            f"tolerance {collision.penetration_tolerance_m:.9g} m exceeds the "
            f"training maximum {MAX_TRAINING_PENETRATION_TOLERANCE_M:.9g} m."
        )
    if collision.checked_frame_count != reference.frame_count:
        raise ValueError(
            f"Reference {reference.sequence_id!r} collision and clearance record "
            f"covers {collision.checked_frame_count} frames, expected "
            f"{reference.frame_count}."
        )
    if (
        collision.minimum_signed_distance_m
        < -collision.penetration_tolerance_m - 1.0e-9
    ):
        raise ValueError(
            f"Reference {reference.sequence_id!r} collision and clearance record "
            "exceeds its penetration tolerance."
        )
    return qualification


def save_dexterous_reference_npz(
    reference: DexterousReference,
    output_path: str | Path,
    *,
    compressed: bool = True,
) -> Path:
    """Write one validated reference to an Isaac-ready NPZ."""

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields: dict[str, Any] = {
        "schema_version": np.asarray(reference.schema_version),
        "sequence_id": np.asarray(reference.sequence_id),
        "robot_name": np.asarray(reference.robot_name),
        "robot_layout": np.asarray(reference.robot_layout),
        "fps": np.asarray(reference.fps, dtype=np.float32),
        "joint_names": np.asarray(reference.joint_names),
        "qpos": reference.qpos,
        "qvel": reference.qvel,
        "left_joint_names": np.asarray(reference.left_joint_names),
        "right_joint_names": np.asarray(reference.right_joint_names),
        "fixed_root_pose_w": reference.fixed_root_pose_w,
        "left_wrist_pose_w": reference.left_wrist_pose_w,
        "right_wrist_pose_w": reference.right_wrist_pose_w,
        "left_wrist_twist_w": reference.left_wrist_twist_w,
        "right_wrist_twist_w": reference.right_wrist_twist_w,
        "left_wrist_frame_name": np.asarray(reference.left_wrist_frame_name),
        "right_wrist_frame_name": np.asarray(reference.right_wrist_frame_name),
        "object_names": np.asarray(reference.object_names),
        "object_poses_w": reference.object_poses_w,
        "object_twists_w": reference.object_twists_w,
        "object_asset_paths": np.asarray(reference.object_asset_paths),
        "object_asset_sha256": np.asarray(reference.object_asset_sha256),
        "object_scales": reference.object_scales,
        "object_radii": reference.object_radii,
        "left_hand_frame_names": np.asarray(reference.left_hand_frame_names),
        "left_hand_frame_poses_w": reference.left_hand_frame_poses_w,
        "right_hand_frame_names": np.asarray(reference.right_hand_frame_names),
        "right_hand_frame_poses_w": reference.right_hand_frame_poses_w,
        "support_surface_names": np.asarray(reference.support_surface_names),
        "support_surface_asset_paths": np.asarray(
            reference.support_surface_asset_paths
        ),
        "support_surface_asset_sha256": np.asarray(
            reference.support_surface_asset_sha256
        ),
        "collision_asset_dependencies_json": np.asarray(
            json.dumps(
                [
                    dependency.to_dict()
                    for dependency in reference.collision_asset_dependencies
                ],
                sort_keys=True,
                allow_nan=False,
            )
        ),
        "support_surface_scales": reference.support_surface_scales,
        "support_surface_poses_w": reference.support_surface_poses_w,
        "metadata_json": np.asarray(
            json.dumps(reference.metadata, sort_keys=True, allow_nan=False)
        ),
    }
    if reference.training_qualification is not None:
        fields["training_qualification_json"] = np.asarray(
            json.dumps(
                reference.training_qualification.to_dict(),
                sort_keys=True,
                allow_nan=False,
            )
        )
    if reference.scene_physics is not None:
        fields.update(reference.scene_physics.to_npz_fields())
    if reference.contacts is not None:
        fields.update(reference.contacts.to_npz_fields())
    writer: Any = np.savez_compressed if compressed else np.savez
    writer(output, **fields)
    return output.resolve()


def _scalar_string(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar string.")
    item = array.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def load_dexterous_reference_npz(path: str | Path) -> DexterousReference:
    """Load and validate one reference without enabling pickle."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dexterous reference NPZ is missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = (
            "schema_version",
            "sequence_id",
            "robot_name",
            "fps",
            "joint_names",
            "qpos",
            "fixed_root_pose_w",
            "left_wrist_pose_w",
            "right_wrist_pose_w",
            "left_wrist_frame_name",
            "right_wrist_frame_name",
            "object_names",
            "object_poses_w",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Dexterous reference NPZ is missing fields: {missing}.")
        fields = {name: np.asarray(archive[name]) for name in archive.files}

    metadata_text = (
        _scalar_string(fields["metadata_json"], name="metadata_json")
        if "metadata_json" in fields
        else "{}"
    )
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata_json in {source}.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    training_qualification: TrainingQualification | None = None
    if "training_qualification_json" in fields:
        qualification_text = _scalar_string(
            fields["training_qualification_json"],
            name="training_qualification_json",
        )
        try:
            qualification_payload = json.loads(qualification_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid training_qualification_json in {source}."
            ) from exc
        if not isinstance(qualification_payload, dict):
            raise ValueError("training_qualification_json must decode to an object.")
        training_qualification = TrainingQualification.from_dict(qualification_payload)
    collision_asset_dependencies: tuple[CollisionAssetDependency, ...] = ()
    if "collision_asset_dependencies_json" in fields:
        dependency_text = _scalar_string(
            fields["collision_asset_dependencies_json"],
            name="collision_asset_dependencies_json",
        )
        try:
            dependency_payload = json.loads(dependency_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid collision_asset_dependencies_json in {source}."
            ) from exc
        if not isinstance(dependency_payload, list) or any(
            not isinstance(item, Mapping) for item in dependency_payload
        ):
            raise ValueError(
                "collision_asset_dependencies_json must decode to a list of objects."
            )
        collision_asset_dependencies = tuple(
            CollisionAssetDependency.from_dict(item) for item in dependency_payload
        )

    loaded_schema = _scalar_string(fields["schema_version"], name="schema_version")
    if loaded_schema not in {REFERENCE_SCHEMA_VERSION, LEGACY_REFERENCE_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported reference schema {loaded_schema!r}; expected "
            f"{REFERENCE_SCHEMA_VERSION!r} or {LEGACY_REFERENCE_SCHEMA_VERSION!r}."
        )
    return DexterousReference(
        schema_version=REFERENCE_SCHEMA_VERSION,
        sequence_id=_scalar_string(fields["sequence_id"], name="sequence_id"),
        robot_name=_scalar_string(fields["robot_name"], name="robot_name"),
        robot_layout=(
            _scalar_string(fields["robot_layout"], name="robot_layout")
            if "robot_layout" in fields
            else "fixed_base"
        ),
        fps=float(np.asarray(fields["fps"]).reshape(())),
        joint_names=_decode_string_array(fields["joint_names"]),
        qpos=fields["qpos"],
        qvel=fields.get("qvel"),
        left_joint_names=(
            _decode_string_array(fields["left_joint_names"])
            if "left_joint_names" in fields
            else ()
        ),
        right_joint_names=(
            _decode_string_array(fields["right_joint_names"])
            if "right_joint_names" in fields
            else ()
        ),
        fixed_root_pose_w=fields["fixed_root_pose_w"],
        left_wrist_pose_w=fields["left_wrist_pose_w"],
        right_wrist_pose_w=fields["right_wrist_pose_w"],
        left_wrist_twist_w=fields.get("left_wrist_twist_w"),
        right_wrist_twist_w=fields.get("right_wrist_twist_w"),
        left_wrist_frame_name=_scalar_string(
            fields["left_wrist_frame_name"], name="left_wrist_frame_name"
        ),
        right_wrist_frame_name=_scalar_string(
            fields["right_wrist_frame_name"], name="right_wrist_frame_name"
        ),
        object_names=_decode_string_array(fields["object_names"]),
        object_poses_w=fields["object_poses_w"],
        object_twists_w=fields.get("object_twists_w"),
        object_asset_paths=_resolve_asset_paths(
            (
                _decode_string_array(fields["object_asset_paths"])
                if "object_asset_paths" in fields
                else ()
            ),
            base_dir=source.parent,
            digests=(
                _decode_string_array(fields["object_asset_sha256"])
                if "object_asset_sha256" in fields
                else ()
            ),
        ),
        object_asset_sha256=(
            _decode_string_array(fields["object_asset_sha256"])
            if "object_asset_sha256" in fields
            else ()
        ),
        object_scales=fields.get("object_scales"),
        object_radii=fields.get("object_radii"),
        left_hand_frame_names=(
            _decode_string_array(fields["left_hand_frame_names"])
            if "left_hand_frame_names" in fields
            else ()
        ),
        left_hand_frame_poses_w=fields.get("left_hand_frame_poses_w"),
        right_hand_frame_names=(
            _decode_string_array(fields["right_hand_frame_names"])
            if "right_hand_frame_names" in fields
            else ()
        ),
        right_hand_frame_poses_w=fields.get("right_hand_frame_poses_w"),
        support_surface_names=(
            _decode_string_array(fields["support_surface_names"])
            if "support_surface_names" in fields
            else ()
        ),
        support_surface_asset_paths=_resolve_asset_paths(
            (
                _decode_string_array(fields["support_surface_asset_paths"])
                if "support_surface_asset_paths" in fields
                else ()
            ),
            base_dir=source.parent,
            digests=(
                _decode_string_array(fields["support_surface_asset_sha256"])
                if "support_surface_asset_sha256" in fields
                else ()
            ),
        ),
        support_surface_asset_sha256=(
            _decode_string_array(fields["support_surface_asset_sha256"])
            if "support_surface_asset_sha256" in fields
            else ()
        ),
        collision_asset_dependencies=collision_asset_dependencies,
        support_surface_scales=fields.get("support_surface_scales"),
        support_surface_poses_w=fields.get("support_surface_poses_w"),
        scene_physics=ScenePhysics.from_npz(fields),
        contacts=ContactSequence.from_npz(fields),
        training_qualification=training_qualification,
        metadata=metadata,
        source_path=source,
    )


@dataclass(frozen=True, slots=True)
class ReferenceManifestEntry:
    """One motion entry in a JSON Reference manifest."""

    name: str
    path: str
    sha256: str
    frames: int
    object_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.path:
            raise ValueError("Manifest motion name and path must be non-empty.")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256.lower()
        ):
            raise ValueError("Manifest motion sha256 must be a 64-digit hex digest.")
        if self.frames < 2:
            raise ValueError("Manifest motions must contain at least two frames.")


@dataclass(slots=True)
class DexterousReferenceManifest:
    """A hash-bound JSON Manifest of compatible dexterous references."""

    dataset_name: str
    robot_name: str
    reference_fps: float
    joint_names: tuple[str, ...]
    left_wrist_frame_name: str
    right_wrist_frame_name: str
    motions: tuple[ReferenceManifestEntry, ...]
    model: str | None = None
    model_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = field(default=None, repr=False)
    schema: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema {self.schema!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION!r}."
            )
        if not self.dataset_name or not self.robot_name:
            raise ValueError("Manifest dataset_name and robot_name must be non-empty.")
        if not np.isfinite(self.reference_fps) or self.reference_fps <= 0.0:
            raise ValueError("Manifest reference_fps must be finite and positive.")
        self.reference_fps = float(self.reference_fps)
        self.joint_names = _string_tuple(self.joint_names, name="manifest joint_names")
        wrist_frame_names = _string_tuple(
            (self.left_wrist_frame_name, self.right_wrist_frame_name),
            name="manifest wrist frame names",
        )
        self.left_wrist_frame_name, self.right_wrist_frame_name = wrist_frame_names
        self.motions = tuple(self.motions)
        if not self.motions:
            raise ValueError("Manifest motions must be non-empty.")
        names = tuple(item.name for item in self.motions)
        if len(set(names)) != len(names):
            raise ValueError("Manifest motion names must be unique.")
        if (self.model is None) != (self.model_sha256 is None):
            raise ValueError("Manifest model and model_sha256 must be set together.")
        try:
            json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Manifest metadata must be JSON-compatible.") from exc

    def resolve_motion_path(self, entry: ReferenceManifestEntry) -> Path:
        path = Path(entry.path).expanduser()
        if path.is_absolute():
            return path.resolve()
        if self.source_path is None:
            raise ValueError("A relative manifest entry needs a saved manifest path.")
        return (self.source_path.parent / path).resolve()

    def resolve_model_path(self) -> Path | None:
        if self.model is None:
            return None
        path = Path(self.model).expanduser()
        if path.is_absolute():
            return path.resolve()
        if self.source_path is None:
            raise ValueError("A relative model path needs a saved manifest path.")
        return (self.source_path.parent / path).resolve()

    def verify_model(self) -> Path | None:
        path = self.resolve_model_path()
        expected_hash = self.model_sha256
        if path is None:
            return None
        if expected_hash is None:
            raise RuntimeError("Manifest model hash invariant is invalid.")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Robot model hash mismatch for {path}: expected "
                f"{expected_hash}, got {actual_hash}."
            )
        return path

    def verify_runtime_model(
        self,
        runtime_model_path: str | Path,
        *,
        require_declared: bool = True,
    ) -> Path:
        """Bind the MJCF used at runtime to the Manifest model hash."""

        runtime_path = Path(runtime_model_path).expanduser().resolve()
        if not runtime_path.is_file():
            raise FileNotFoundError(f"Runtime robot model is missing: {runtime_path}")
        expected_hash = self.model_sha256
        if expected_hash is None:
            if require_declared:
                raise ValueError(
                    "Reference Manifest does not declare model and model_sha256; "
                    "strict runtime model binding is not possible."
                )
            return runtime_path
        actual_hash = sha256_file(runtime_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Runtime robot model hash mismatch for {runtime_path}: expected "
                f"{expected_hash}, got {actual_hash}."
            )
        return runtime_path

    def load_references(
        self,
        *,
        verify_hashes: bool = True,
        verify_declared_model: bool = True,
    ) -> tuple[DexterousReference, ...]:
        if verify_hashes and verify_declared_model:
            self.verify_model()
        references: list[DexterousReference] = []
        for entry in self.motions:
            path = self.resolve_motion_path(entry)
            if verify_hashes:
                actual_hash = sha256_file(path)
                if actual_hash != entry.sha256.lower():
                    raise ValueError(
                        f"Reference hash mismatch for {path}: expected "
                        f"{entry.sha256}, got {actual_hash}."
                    )
            reference = load_dexterous_reference_npz(path)
            if reference.frame_count != entry.frames:
                raise ValueError(
                    f"Reference frame count mismatch for {path}: expected "
                    f"{entry.frames}, got {reference.frame_count}."
                )
            if reference.joint_names != self.joint_names:
                raise ValueError(f"Reference joint order differs in {path}.")
            if (
                reference.left_wrist_frame_name != self.left_wrist_frame_name
                or reference.right_wrist_frame_name != self.right_wrist_frame_name
            ):
                raise ValueError(f"Reference wrist frame names differ in {path}.")
            if reference.robot_name != self.robot_name:
                raise ValueError(f"Reference robot_name differs in {path}.")
            if not np.isclose(reference.fps, self.reference_fps, atol=1.0e-6):
                raise ValueError(f"Reference fps differs in {path}.")
            references.append(reference)
        return tuple(references)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "dataset_name": self.dataset_name,
            "robot_name": self.robot_name,
            "reference_fps": self.reference_fps,
            "joint_names": list(self.joint_names),
            "left_wrist_frame_name": self.left_wrist_frame_name,
            "right_wrist_frame_name": self.right_wrist_frame_name,
            "motion_count": len(self.motions),
            "total_frames": sum(item.frames for item in self.motions),
            "motions": [
                {
                    "name": item.name,
                    "path": item.path,
                    "sha256": item.sha256,
                    "frames": item.frames,
                    "object_names": list(item.object_names),
                }
                for item in self.motions
            ],
            "metadata": self.metadata,
        }
        if self.model is not None:
            result["model"] = self.model
            result["model_sha256"] = self.model_sha256
        return result

    def save(self, output_path: str | Path) -> Path:
        output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        self.source_path = output.resolve()
        return self.source_path


def load_dexterous_reference_manifest(
    path: str | Path,
) -> DexterousReferenceManifest:
    """Load and validate one ILTools Reference manifest."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dexterous reference manifest is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dexterous reference manifest must be a JSON object.")
    raw_motions = payload.get("motions")
    if not isinstance(raw_motions, list):
        raise ValueError("Dexterous reference manifest must contain motions.")
    motions = tuple(
        ReferenceManifestEntry(
            name=str(item["name"]),
            path=str(item["path"]),
            sha256=str(item["sha256"]).lower(),
            frames=int(item["frames"]),
            object_names=tuple(str(name) for name in item.get("object_names", ())),
        )
        for item in raw_motions
        if isinstance(item, dict)
    )
    if len(motions) != len(raw_motions):
        raise ValueError("Each manifest motion must be a JSON object.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Manifest metadata must be a JSON object.")
    return DexterousReferenceManifest(
        schema=str(payload.get("schema", "")),
        dataset_name=str(payload.get("dataset_name", "")),
        robot_name=str(payload.get("robot_name", "")),
        reference_fps=float(payload.get("reference_fps", 0.0)),
        joint_names=tuple(str(name) for name in payload.get("joint_names", ())),
        left_wrist_frame_name=str(payload.get("left_wrist_frame_name", "")),
        right_wrist_frame_name=str(payload.get("right_wrist_frame_name", "")),
        motions=motions,
        model=(None if payload.get("model") is None else str(payload["model"])),
        model_sha256=(
            None
            if payload.get("model_sha256") is None
            else str(payload["model_sha256"]).lower()
        ),
        metadata=metadata,
        source_path=source,
    )


def create_dexterous_reference_manifest(
    reference_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    dataset_name: str,
    model_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Create a relative, hash-bound Manifest from compatible NPZ files."""

    paths = tuple(Path(path).expanduser().resolve() for path in reference_paths)
    if not paths:
        raise ValueError("At least one reference path is required.")
    references = tuple(load_dexterous_reference_npz(path) for path in paths)
    for reference in references:
        reference.verify_scene_assets(require_hashes=True)
    first = references[0]
    for path, reference in zip(paths[1:], references[1:], strict=True):
        if reference.robot_name != first.robot_name:
            raise ValueError(f"Reference robot_name differs in {path}.")
        if reference.joint_names != first.joint_names:
            raise ValueError(f"Reference joint order differs in {path}.")
        if (
            reference.left_wrist_frame_name != first.left_wrist_frame_name
            or reference.right_wrist_frame_name != first.right_wrist_frame_name
        ):
            raise ValueError(f"Reference wrist frame names differ in {path}.")
        if not np.isclose(reference.fps, first.fps, atol=1.0e-6):
            raise ValueError(f"Reference fps differs in {path}.")

    output = Path(output_path).expanduser().resolve()
    entries = tuple(
        ReferenceManifestEntry(
            name=reference.sequence_id,
            path=os.path.relpath(path, start=output.parent),
            sha256=sha256_file(path),
            frames=reference.frame_count,
            object_names=reference.object_names,
        )
        for path, reference in zip(paths, references, strict=True)
    )
    model: str | None = None
    model_hash: str | None = None
    if model_path is not None:
        resolved_model = Path(model_path).expanduser().resolve()
        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"Manifest robot model is missing: {resolved_model}"
            )
        model = os.path.relpath(resolved_model, start=output.parent)
        model_hash = sha256_file(resolved_model)
    manifest = DexterousReferenceManifest(
        dataset_name=dataset_name,
        robot_name=first.robot_name,
        reference_fps=first.fps,
        joint_names=first.joint_names,
        left_wrist_frame_name=first.left_wrist_frame_name,
        right_wrist_frame_name=first.right_wrist_frame_name,
        motions=entries,
        model=model,
        model_sha256=model_hash,
        metadata=dict(metadata or {}),
        source_path=output,
    )
    return manifest.save(output)


def load_dexterous_reference_set(
    path: str | Path,
    *,
    verify_hashes: bool = True,
    runtime_model_path: str | Path | None = None,
    require_model_hash: bool = False,
) -> tuple[DexterousReference, ...]:
    """Load one NPZ or a Manifest and optionally bind its runtime MJCF."""

    source = Path(path).expanduser().resolve()
    is_manifest = source.suffix.lower() == ".json"
    if require_model_hash and runtime_model_path is None:
        raise ValueError(
            "require_model_hash=True requires an explicit runtime_model_path."
        )
    if not is_manifest:
        if runtime_model_path is not None or require_model_hash:
            raise ValueError(
                "Runtime model hash binding requires a JSON Reference Manifest."
            )
        return (load_dexterous_reference_npz(source),)

    manifest = load_dexterous_reference_manifest(source)
    if runtime_model_path is not None:
        manifest.verify_runtime_model(
            runtime_model_path, require_declared=require_model_hash
        )
    return manifest.load_references(
        verify_hashes=verify_hashes,
        verify_declared_model=runtime_model_path is None,
    )


__all__ = [
    "CollisionClearanceQualification",
    "CollisionAssetDependency",
    "ContactSequence",
    "DexterousReference",
    "DexterousReferenceManifest",
    "MANIFEST_SCHEMA_VERSION",
    "REFERENCE_SCHEMA_VERSION",
    "ReferenceManifestEntry",
    "ScenePhysics",
    "TRAINING_QUALIFICATION_SCHEMA_VERSION",
    "TrainingQualification",
    "build_urdf_collision_asset_dependencies",
    "create_dexterous_reference_manifest",
    "derive_object_twists_w",
    "load_dexterous_reference_manifest",
    "load_dexterous_reference_npz",
    "load_dexterous_reference_set",
    "save_dexterous_reference_npz",
    "sha256_file",
    "verify_training_qualification",
]
