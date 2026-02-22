
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
