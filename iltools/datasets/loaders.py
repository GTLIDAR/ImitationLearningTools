"""Lazy dataset-loader registry.

This module intentionally stores loader targets as import strings so optional
dataset backends do not become import-time package requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class DatasetLoaderSpec:
    """Import target for a dataset loader class."""

    module: str
    class_name: str
    optional_dependency: str | None = None


_DATASET_LOADERS: dict[str, DatasetLoaderSpec] = {
    "lafan1": DatasetLoaderSpec(
        module="iltools.datasets.lafan1.loader",
        class_name="Lafan1CsvLoader",
    ),
    "lafan1_csv": DatasetLoaderSpec(
        module="iltools.datasets.lafan1.loader",
        class_name="Lafan1CsvLoader",
    ),
    "lerobot": DatasetLoaderSpec(
        module="iltools.datasets.lerobot.loader",
        class_name="LeRobotLoader",
        optional_dependency="lerobot",
    ),
    "loco_mujoco": DatasetLoaderSpec(
        module="iltools.datasets.loco_mujoco.loader",
        class_name="LocoMuJoCoLoader",
        optional_dependency="loco-mujoco",
    ),
}


def register_dataset_loader(
    name: str,
    *,
    module: str,
    class_name: str,
    optional_dependency: str | None = None,
) -> None:
    """Register or override a dataset loader import target."""
    key = _normalize_name(name)
    _DATASET_LOADERS[key] = DatasetLoaderSpec(
        module=module,
        class_name=class_name,
        optional_dependency=optional_dependency,
    )


def get_dataset_loader_spec(name: str) -> DatasetLoaderSpec | None:
    """Return the registered loader spec for ``name`` if one exists."""
    return _DATASET_LOADERS.get(_normalize_name(name))


def load_dataset_loader(name: str) -> type[Any]:
    """Import and return a registered loader class.

    Raises:
        KeyError: if ``name`` is not registered.
        ImportError: if the loader module or class cannot be imported.
    """
    key = _normalize_name(name)
    spec = _DATASET_LOADERS.get(key)
    if spec is None:
        raise KeyError(f"Unknown dataset loader: {name}")

    try:
        module = import_module(spec.module)
    except ImportError as exc:
        dependency = spec.optional_dependency or spec.module
        raise ImportError(
            f"Dataset loader '{key}' requires optional dependency '{dependency}'."
        ) from exc

    try:
        loader_cls = getattr(module, spec.class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Dataset loader '{key}' module '{spec.module}' does not define "
            f"'{spec.class_name}'."
        ) from exc

    return loader_cls


def registered_dataset_loaders() -> tuple[str, ...]:
    """Return registered dataset loader names."""
    return tuple(sorted(_DATASET_LOADERS))


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")
