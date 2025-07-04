import os
import json
import queue
import threading
from functools import lru_cache
from typing import Dict, Any, Optional, Union, List

import numpy as np
import torch
import zarr
from iltools_datasets.base_loader import BaseTrajectoryDataset


class LazyTrajectoryDataset(BaseTrajectoryDataset):
    """Dataset that reads from per-trajectory Zarr format.

    Each trajectory is stored as a separate Zarr group:
    trajectories.zarr/traj_XXXX/observations/{key}, actions/{key}
    """

    def __init__(self, data_dir: str, device: str = "cpu"):
        self.data_dir = data_dir
        self.device = device

        # Load global metadata
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self._metadata = json.load(f)

        self.lengths = self._metadata["trajectory_lengths"]

        self.observation_keys = self._metadata["observation_keys"]
        self.action_keys = self._metadata["action_keys"] or []
        self.dt_list = self._metadata["dt"]

        # Open Zarr store
        zarr_path = os.path.join(data_dir, "trajectories.zarr")
        store = zarr.DirectoryStore(zarr_path)
        self.root = zarr.open_group(store=store, mode="r")

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        returns a trajectory
        """
        return self.get_trajectory_slice(idx)

    def get_trajectory_slice(
        self, idx: int, start: Optional[int] = 0, end: Optional[int] = -1
    ) -> Dict[str, Any]:
        """Get a slice of a trajectory without loading the entire trajectory."""
        traj_group = self.root[f"traj_{idx:04d}"]

        # Load observations slice
        obs_dict = {}
        obs_group = traj_group["observations"]
        for key in self.observation_keys:
            obs_dict[key] = torch.tensor(
                np.array(obs_group[key][start:end]),
                dtype=torch.float32,
                device=self.device,
            )

        # Load actions slice (if present)
        act_dict = {}
        if self.action_keys and "actions" in traj_group:
            act_group = traj_group["actions"]
            for key in self.action_keys:
                act_dict[key] = torch.tensor(
                    np.array(act_group[key][start:end]),
                    dtype=torch.float32,
                    device=self.device,
                )

        return {
            **obs_dict,
            **act_dict,
            "dt": torch.tensor(
                float(self.dt_list[idx]), dtype=torch.float32, device=self.device
            ),
        }

    def validate_frequency_consistency(self) -> bool:
        """
        Validate that all samples have consistent dt values.
        Returns True if consistent, False otherwise.
        """
        if len(self) == 0:
            return True

        # Sample a few dt values to check consistency
        sample_indices = [0, min(len(self) - 1, 10), len(self) - 1]
        dt_values = []
        for idx in sample_indices:
            try:
                dt_val = float(self.dt_list[idx])
                dt_values.append(dt_val)
            except Exception:
                return False

        # Check if all dt values are approximately equal
        if len(dt_values) > 1:
            return all(abs(dt - dt_values[0]) < 1e-6 for dt in dt_values)
        return True

    def get_frequency_info(self) -> Dict[str, Any]:
        """
        Get frequency-related information from the dataset.
        """
        info: Dict[str, Any] = {
            "consistent_dt": self.validate_frequency_consistency(),
        }

        # Add sample dt value
        if len(self) > 0:
            try:
                sample_dt = float(self.dt_list[0])
                info["sample_dt"] = sample_dt
                info["sample_frequency"] = 1.0 / sample_dt if sample_dt > 0 else None
            except Exception:
                info["sample_dt"] = None
                info["sample_frequency"] = None

        return info

    def as_loader(self, **kwargs):
        raise NotImplementedError(
            "ZarrBackedTrajectoryDataset cannot be converted to a loader directly."
        )

    @property
    def metadata(self):
        from iltools_core.metadata_schema import DatasetMeta

        return DatasetMeta(**self._metadata)

    def get_window(self, idx: int, start: int, length: int) -> Dict[str, Any]:
        """
        Efficiently fetch a window of [start:start+length] for a given trajectory index.
        Returns a dict with 'observations' and 'actions', each a dict of tensors.
        If the requested window exceeds the trajectory length, it is truncated.
        """
        # Find the actual length of this trajectory
        traj_len = self.lengths[idx]
        end = min(start + length, traj_len)

        traj_group = self.root[f"traj_{idx:04d}"]

        # Load observations slice
        obs = {}
        obs_group = traj_group["observations"]
        for k in self.observation_keys:
            obs[k] = (
                torch.from_numpy(np.array(obs_group[k][start:end]))
                .float()
                .to(self.device)
            )

        # Load actions slice
        actions = {}
        if self.action_keys and "actions" in traj_group:
            act_group = traj_group["actions"]
            for k in self.action_keys:
                actions[k] = (
                    torch.from_numpy(np.array(act_group[k][start:end]))
                    .float()
                    .to(self.device)
                )

        return {"observations": obs, "actions": actions}

    def batch_get(
        self,
        traj_indices: Union[torch.Tensor, np.ndarray, List[int]],
        step_indices: Union[torch.Tensor, np.ndarray, List[int]],
        key: str,
        data_type: str = "observations",
    ) -> torch.Tensor:
        """
        Efficiently fetch a batch of (traj_idx, step_idx) pairs for a given key and data_type.
        Args:
            traj_indices: 1D torch.Tensor, np.ndarray, or list of trajectory indices [N]
            step_indices: 1D torch.Tensor, np.ndarray, or list of step indices [N]
            key: observation or action key
            data_type: 'observations' or 'actions'
        Returns:
            torch.Tensor of shape [N, ...] on the correct device
        """
        if data_type not in ["observations", "actions"]:
            raise ValueError(
                f"data_type must be 'observations' or 'actions', got {data_type}"
            )

        # Convert indices to numpy
        if isinstance(traj_indices, torch.Tensor):
            traj_indices = traj_indices.cpu().numpy()
        if isinstance(step_indices, torch.Tensor):
            step_indices = step_indices.cpu().numpy()

        # Collect data from per-trajectory format
        result = []
        for ti, si in zip(traj_indices, step_indices):
            if ti < 0 or ti >= len(self):
                raise IndexError(
                    f"Trajectory index {ti} out of bounds for dataset with {len(self)} trajectories"
                )

            traj_group = self.root[f"traj_{ti:04d}"]
            if data_type == "observations":
                data_group = traj_group["observations"]
            else:
                if "actions" not in traj_group:
                    raise KeyError(f"Actions not available for trajectory {ti}")
                data_group = traj_group["actions"]

            if key not in data_group:
                raise KeyError(
                    f"Key '{key}' not found in {data_type} for trajectory {ti}"
                )

            traj_data = data_group[key]
            if si < 0 or si >= traj_data.shape[0]:
                raise IndexError(
                    f"Step index {si} out of bounds for trajectory {ti} with length {traj_data.shape[0]}"
                )
            result.append(np.array(traj_data[si]))

        result = np.stack(result)
        return torch.from_numpy(result).to(self.device)
