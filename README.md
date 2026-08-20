
# imitation-learning-tools

Scaffold for a modular imitation learning toolkit with dataset loaders, retargeting, and CLI utilities.


## Installation

We recommend using the `conda-forge` channel for managing dependencies. For a faster experience, use `mamba` to create the environment and `uv` to install packages.

1.  **Create and activate a conda environment:**
    ```bash
    mamba create -n iltools python=3.11 -c conda-forge
    conda activate iltools
    ```

2.  **Install `uv`:**
    ```bash
    pip install uv
    ```

3.  **Install the project:**

    Base install (without Loco-MuJoCo):
    ```bash
    uv pip install -e .
    ```

    If you need Loco-MuJoCo support (dataset loader, tests, viewers), install the optional extra:
    ```bash
    uv pip install -e .[loco-mujoco]
    ```
    Alternatively with pip:
    ```bash
    pip install -e .[loco-mujoco]
    ```
    *Note: Using the `-e` flag installs the project in "editable" mode, which is recommended for development.*

## Dataset Structure

The structure of a dataset is as follows:

```text
Dataset/
├── motion1/
│   ├── trajectory1/
│   │   ├── observations/
│   │   │   ├── qpos
│   │   │   ├── qvel
│   │   │   └── ...
│   │   ├── actions/
│   │   │   ├── target_joint_pos
│   │   │   └── ...
│   │   ├── rewards
│   │   └── infos
│   ├── trajectory2/
│   │   ├── observations/
│   │   │   ├── qpos
│   │   │   ├── qvel
│   │   │   └── ...
│   │   ├── actions/
│   │   │   ├── target_joint_pos
│   │   │   └── ...
│   │   ├── rewards
│   │   └── infos
│   └── ...
├── motion2/
│   ├── trajectory1/
│   │   ├── observations/
│   │   │   ├── qpos
│   │   │   ├── qvel
│   │   │   └── ...
│   │   ├── actions/
│   │   │   ├── target_joint_pos
│   │   │   └── ...
│   │   ├── rewards
│   │   └── infos
│   └── ...
└── ...
```

E.g. We currently support ```loco-mujoco``` dataset with various motions such as ```default-walk``` with ```1``` trajectory. Install the optional extra to enable this loader.

### Ego-Exo4D retargeting

The retargeting package accepts Ego-Exo4D EgoPose `annotation3D` JSON files.
It keeps frame numbers and coordinate-frame metadata explicit. Use
`load_ego_pose_trajectory` or `load_ego_pose_bundle` for body and hand files,
then apply `transform_keypoint_trajectory` before model-based IK. The package
provides three bounded paths:

- `KeypointRetargeter` for landmark-angle joint targets.
- `MujocoKeypointRetargeter` for MJCF hinge/slide joints.
- `PinocchioKeypointRetargeter` for scalar joints in a Pinocchio model.

All model-based paths require keypoints marked with
`infos["coordinate_frame"] == "robot"`; the loader does not guess a human to
robot transform. Each retargeted trajectory records target joint names and
derives `qvel` when `dt` is available. `save_joint_reference_npz` writes an
intermediate named `qpos`, `qvel`, `joint_names`, and `fps` archive; it is not
the Vega/Wuji task input. Build the full task contract with
`dexterous_reference_from_trajectory`, save it with
`save_dexterous_reference_npz`, and create strict multi-motion JSON Manifests
with `create_dexterous_reference_manifest`. Full References include wrist,
object, hand-frame, scene-asset hash/scale, and contact data.

For this workspace, run the retargeting tests through the repository Pixi
environment. Do not install a second dependency environment inside the
submodule:

```bash
pixi run python -m pytest -q ImitationLearningTools/tests/retarget
```

### Dexterous Reference contract

Use `load_dexterous_reference_set` to preload one NPZ or one JSON Manifest
before an Isaac scene is constructed:

```python
from iltools.core import load_dexterous_reference_set

references = load_dexterous_reference_set(
    "references.json",
    runtime_model_path="vega_wuji.xml",
    require_model_hash=True,
)
```

Each `DexterousReference` has these fixed-shape arrays:

- `qpos`, `qvel`: `[T, J]`, aligned with `joint_names`.
- `fixed_root_pose_w`: `[7]`.
- `left_wrist_pose_w`, `right_wrist_pose_w`: `[T, 7]`, for the MuJoCo
  sites named by `left_wrist_frame_name` and `right_wrist_frame_name`.
- `object_poses_w`: `[T, O, 7]`, aligned with `object_names`,
  `object_asset_paths`, and `object_radii`.
- `object_twists_w`: `[T, O, 6]`, aligned with `object_names`; each sample is
  the object-pose/root-frame velocity as world-frame linear XYZ followed by
  world-frame angular XYZ. This field is optional on legacy input and is
  always materialized in memory and on save.
  When absent, ILTools derives it deterministically at the Reference FPS from
  position differences and shortest-path WXYZ quaternion differences. Interior
  frames average adjacent interval velocities and endpoints use their sole
  adjacent interval.
- Per-side hand frame poses: `[T, K, 7]`, aligned with hand frame names.
- Static support surface poses: `[K, 7]`, aligned with support surface names
  and asset paths.
- Optional `ScenePhysics`: per-object mass, center of mass, diagonal inertia,
  static friction, dynamic friction, and restitution; plus per-support static
  friction, dynamic friction, and restitution.
- Optional padded contacts: vectors `[T, S, C, 3]` and active/object-index
  masks `[T, S, C]`.

Training uses a second, fail-closed gate. Add a typed
`TrainingQualification` to the Reference and call
`verify_training_qualification` before you give the Reference to a training
runtime:

```python
from iltools.core import (
    CollisionAssetDependency,
    CollisionClearanceQualification,
    ScenePhysics,
    TrainingQualification,
    build_urdf_collision_asset_dependencies,
    verify_training_qualification,
)

reference.collision_asset_dependencies = (
    build_urdf_collision_asset_dependencies(
        reference.object_asset_paths[0],
        asset_role="object",
        asset_index=0,
    )
)

reference.scene_physics = ScenePhysics(
    object_mass_kg=[0.35],
    object_center_of_mass_m=[[0.0, 0.0, 0.01]],
    object_diagonal_inertia_kg_m2=[[0.001, 0.001, 0.001]],
    object_static_friction=[0.8],
    object_dynamic_friction=[0.6],
    object_restitution=[0.05],
    support_static_friction=[0.9],
    support_dynamic_friction=[0.7],
    support_restitution=[0.02],
)
reference.training_qualification = TrainingQualification(
    runtime_qualified=True,
    isaac_runtime_qualified=True,
    inspection_only=False,
    contact_geometry_provenance="contact reconstruction audit run 42",
    collision_clearance=CollisionClearanceQualification(
        qualified=True,
        method="signed-distance replay",
        scope="robot, object, and support geometry for all frames",
        provenance="Isaac Newton audit run 42",
        checked_frame_count=reference.frame_count,
        minimum_signed_distance_m=-0.0001,
        penetration_tolerance_m=0.0002,
    ),
)
verify_training_qualification(reference)
```

The gate verifies the declared scene asset hashes against the local files.
For each URDF scene asset, it also parses every
`collision/geometry/mesh filename` URI and requires a typed
`CollisionAssetDependency` with the current SHA-256 of that local mesh. Visual
meshes, material files, and textures are not part of this physics dependency
contract. Relative dependency URIs resolve from the URDF directory.

Without OpenUSD/pxr, ILTools cannot prove the dependency closure of an
external-reference USD. Strict training therefore accepts a self-contained
ASCII `.usda`, but rejects an ASCII USDA with asset-reference tokens and
accepts `.usd` only when it is UTF-8 USDA text with no asset-reference tokens.
The conservative text scan can also reject asset tokens in comments or visual
properties; it never treats an unparsed texture as proof of collision closure.
Binary `.usd`, `.usdc`, and `.usdz` are rejected. General and inspection
loading remains backward compatible; an old NPZ without
`collision_asset_dependencies_json` loads with an empty dependency tuple, but
a URDF-backed Reference then fails the strict training gate.

The gate requires typed `ScenePhysics`, so a training runtime does not silently
use simulator mass, inertia, friction, or restitution defaults. It also
requires at least one active contact slot, valid object indices, non-zero
contact points, unit contact normals, real contact provenance, and a passing
collision and clearance record that covers all Reference frames. Free-form
metadata does not satisfy this gate. In particular, an inspection Reference with
`runtime_qualified=False`, `isaac_runtime_qualified=False`, unavailable contact
geometry, or `inspection_only=True` is not a training Reference.

The typed record is stored in the v1 Reference NPZ as JSON without pickle. The
training qualification schema is v2, which makes dependency hashes mandatory
at the strict boundary. Loading a v1 qualification or an old v1 Reference NPZ
without the optional dependency record still works for inspection. A general
data or inspection tool can use that Reference, but
`verify_training_qualification` rejects its dependency-unaware qualification.
An older dependency-unaware ILTools loader rejects the new v2 qualification,
which prevents a silent downgrade of strict verification.

With `require_model_hash=True`, loading fails unless the Manifest declares a model
hash and the selected runtime MJCF has the same SHA-256 digest.

All pose arrays use `[x, y, z, qw, qx, qy, qz]`. NPZ loading uses
`allow_pickle=False`. Manifests bind each motion and the optional robot model
to SHA-256 hashes and declare the common wrist-frame names.
`MujocoDualHandRetargeter` solves the two named wrist-site SE(3) targets and
records those site names. `dexterous_reference_from_trajectory` carries the
names into the runtime contract.

### LAFAN1 CSV Loader

You can also build a replay-ready Zarr dataset directly from prepared CSV (or NPZ) motions using
`Lafan1CsvLoader`. The CSV format follows the mimic preprocessing convention:

- columns `0:3`: root position
- columns `3:7`: root quaternion in `xyzw` (internally converted to `wxyz`)
- columns `7:`: joint positions

```python
from iltools.datasets.lafan1.loader import Lafan1CsvLoader

cfg = {
    "dataset": {
        "trajectories": {
            "lafan1_csv": [
                "/path/to/motions/walk_001.csv",
                {"path": "/path/to/motions/dance_002.csv", "input_fps": 60, "frame_range": [1, 1200]},
            ]
        }
    },
    "control_freq": 50,   # output sampling frequency
    "input_fps": 60,      # default input fps for csv files
}

loader = Lafan1CsvLoader(cfg=cfg, build_zarr_dataset=True, zarr_path="/tmp/lafan1_dataset.zarr")
print(loader.metadata)
```

To group multiple files as trajectories of the same motion (LocoMuJoCo-style motion grouping),
use an explicit motion `name` with `paths`:

```python
cfg = {
    "dataset": {
        "trajectories": {
            "lafan1_csv": [
                {"name": "dance_combo", "paths": ["/data/dance_01.csv", "/data/dance_02.csv"]},
                {"name": "walk_combo", "paths": ["/data/walk_01.csv", "/data/walk_02.csv"]},
            ]
        }
    },
    "control_freq": 50,
}
```

## Usage

Here is an example of how to use the `TrajectoryDatasetManager` to load and step through a dataset, inspired by `tests/datasets/test_integration.py`.

```python
import torch
from tensordict import TensorDict

from iltools.datasets.dataset_manager import TrajectoryDatasetManager

# Mock configuration for demonstration
class MockConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# 1. Configure the dataset manager
#    Replace 'path/to/your/dataset' with the actual path to your Zarr dataset directory.
#    This directory should contain 'trajectories.zarr' and 'metadata.json'.
cfg = MockConfig(
    dataset_path='path/to/your/dataset',
    assignment_strategy='random',  # or 'sequential', 'round_robin', 'curriculum'
    window_size=128
)

# 2. Initialize the manager for a specified number of environments
num_envs = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
manager = TrajectoryDatasetManager(cfg, num_envs, device)

# 3. Reset trajectories at the beginning of a training session
manager.reset_trajectories()

# 4. Fetch reference data in a loop (e.g., inside your RL environment's step function)
for _ in range(1000):  # Simulate 1000 steps
    # Get a batch of reference data for the current timestep
    reference_data = manager.get_reference_data()

    # The data is returned as a TensorDict for easy access
    # Shape: [num_envs, ...]
    com_positions = reference_data["com_pos"]
    joint_positions = reference_data["joint_pos"]

    # Your agent would use this data to compute actions...
    # print(f"Step {_}: COM Position Batch Shape: {com_positions.shape}")

    # To reset specific environments that have completed their episode:
    # done_env_ids = torch.tensor([2, 5], device=device) # Example IDs
    # manager.reset_trajectories(done_env_ids)

```

## Testing

To verify that the `loco_mujoco` dataset loader is working correctly, you can run the specific test file for it. This is currently the recommended test to run.

First, ensure you have the necessary test dependencies installed (including the optional extra):

```bash
uv pip install pytest numpy torch omegaconf zarr mujoco
uv pip install -e .[loco-mujoco]
```

Then, run the following command from the root of the project:

```bash
pytest tests/datasets/test_loco_mujoco_loader.py
```

To visually inspect the loaded trajectories, you can use the `--visualize` flag. This will open a MuJoCo viewer and replay the first trajectory.

```bash
pytest tests/datasets/test_loco_mujoco_loader.py --visualize
```
