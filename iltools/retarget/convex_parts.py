"""Convex decomposition of an object mesh, cached on disk.

A hollow object cannot be represented by one convex shape. Its convex hull
fills the cavity, so a hand reaching inside reads as centimetres of
penetration that is not there. The released video-to-data spawn asks Isaac for
``convex_decomposition`` for exactly this reason.

Any tool that reasons about the same contact outside Isaac has to use the same
decomposition, or it measures a different object. This module produces the
convex parts once and caches them next to the source mesh, keyed by the mesh
content and the decomposition settings, so a measurement and the simulation
agree.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConvexDecompositionConfig:
    """Settings for the decomposition, recorded in the cache key."""

    threshold: float = 0.05
    """Concavity below which CoACD stops splitting. Lower means more parts."""

    max_convex_hull: int = -1
    """Cap on the number of parts; -1 leaves it to the concavity threshold."""

    seed: int = 0

    def key(self) -> str:
        return json.dumps(
            {
                "threshold": float(self.threshold),
                "max_convex_hull": int(self.max_convex_hull),
                "seed": int(self.seed),
            },
            sort_keys=True,
        )


def _cache_dir(mesh_path: Path, config: ConvexDecompositionConfig) -> Path:
    digest = hashlib.sha256()
    digest.update(mesh_path.read_bytes())
    digest.update(config.key().encode("utf-8"))
    return mesh_path.parent / f".convex_parts_{digest.hexdigest()[:16]}"


def convex_part_paths(
    mesh_path: str | Path,
    *,
    config: ConvexDecompositionConfig | None = None,
) -> tuple[Path, ...]:
    """Return one OBJ per convex part of ``mesh_path``, decomposing if needed.

    The result is cached beside the mesh under a directory named for the mesh
    content and the settings, so changing either produces a new cache rather
    than a stale hit.
    """

    import coacd
    import trimesh

    settings = config or ConvexDecompositionConfig()
    mesh_path = Path(mesh_path).expanduser().resolve()
    cache = _cache_dir(mesh_path, settings)
    existing = sorted(cache.glob("part_*.obj"))
    if existing:
        return tuple(existing)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    parts = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
        threshold=float(settings.threshold),
        max_convex_hull=int(settings.max_convex_hull),
        seed=int(settings.seed),
    )
    cache.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, (vertices, faces) in enumerate(parts):
        part = trimesh.Trimesh(
            vertices=np.asarray(vertices), faces=np.asarray(faces), process=False
        )
        path = cache / f"part_{index:03d}.obj"
        part.export(str(path))
        written.append(path)
    if not written:
        raise ValueError(f"Convex decomposition of {mesh_path} produced no parts.")
    return tuple(written)


__all__ = ["ConvexDecompositionConfig", "convex_part_paths"]
