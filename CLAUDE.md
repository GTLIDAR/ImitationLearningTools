# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install:**
```bash
pip install -e .
# With LocoMuJoCo support:
pip install -e ".[loco-mujoco]"
```

**Run all tests:**
```bash
pytest
```

**Run a single test file:**
```bash
pytest tests/datasets/test_lafan1_csv_loader.py
```

**Run tests excluding slow/visual:**
```bash
pytest -m "not slow and not visual"
```

**Run visual tests (requires display):**
```bash
pytest --visualize -m visual
```

**CLI:**
```bash
iltools load loco_mujoco
iltools load amass
```

## Architecture

The package (`iltools`) is organized into four modules:

### `iltools/core/`
Foundational data types:
- `Trajectory`: dataclass holding observations, actions, rewards, infos, and dt
- `DatasetMeta`: Pydantic model describing a dataset (name, source, version, joint/body/site names, trajectory count and lengths)

### `iltools/datasets/`
All dataset I/O and management:
- `BaseLoader` (abstract): defines the loader interface all concrete loaders implement
- `Lafan1CsvLoader`: loads LAFAN1 motion capture CSVs; supports configurable input FPS, frame subsampling, and motion grouping
- `LocoMuJoCoLoader`: integrates with the `loco-mujoco` library; supports multiple sub-datasets (default, LAFAN1, AMASS) and writes to Zarr
- `TrajectoryDatasetManager` (`manager.py`): the main runtime class for RL training—manages a pool of pre-loaded trajectories across multiple parallel environments; supports assignment strategies (random, sequential, round-robin, curriculum) and uses compiled PyTorch functions for performance
- `utils.py`: helpers to build TorchRL `TensorDictReplayBuffer` from Zarr datasets

### `iltools/retarget/`
Motion retargeting:
- `BaseRetarget` (abstract): interface for retargeting implementations
- `PinocchioRetarget`: kinematics-based retargeting using Pinocchio

### `iltools/cli/`
Typer-based CLI (`iltools` command). Commands map to dataset loaders; `retarget` is a placeholder.

### Data Flow
Datasets are stored on disk as **Zarr v3** archives. Loaders parse raw data (CSVs, loco-mujoco envs) and write Zarr. At training time, `TrajectoryDatasetManager` loads Zarr slices into TensorDicts for batched environment resets and trajectory tracking.

## Test Markers
- `slow`: long-running tests (skipped by default with `-m "not slow"`)
- `visual`: tests that open a MuJoCo viewer (require `--visualize` flag to run)
