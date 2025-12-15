import os
from typing import Any

import numpy as np
import zarr
from loco_mujoco.environments import LocoEnv
from loco_mujoco.task_factories import (
    AMASSDatasetConf,
    DefaultDatasetConf,
    ImitationFactory,
    LAFAN1DatasetConf,
)
from loco_mujoco.trajectory.handler import TrajectoryHandler
from omegaconf import DictConfig
from zarr.storage import LocalStore

from iltools_core.metadata_schema import DatasetMeta
from iltools_datasets.base_loader import (
    BaseDataset,
    BaseLoader,
)


class LocoMuJoCoLoader(BaseLoader):
    """
    Flexible loader for Loco-MuJoCo trajectories.
    Now supports multiple datasets/motions (default, lafan1, amass).

    The desired control dt is computed and set through num_substeps when initializing the environment.
    """

    def __init__(
        self,
        env_name: str,
        cfg: DictConfig,
        **kwargs,
    ):
        """cfg can from conf/ or from Isaaclab dataclass"""
        print("[LocoMuJoCoLoader] Initializing LocoMuJoCoLoader")
        self.cfg = cfg
        self.env_name = env_name
        self.dataset_dict = cfg.dataset.get(
            "trajectories", {"default": ["walk"], "amass": [], "lafan1": []}
        )
        print("[LocoMuJoCoLoader] Dataset dictionary:", self.dataset_dict)

        # self._setup_cache()

        self.env: LocoEnv = self._load_env(**kwargs)
        assert hasattr(self.env, "th") and isinstance(
            self.env.th, TrajectoryHandler
        ), "TrajectoryHandler not found in env"

        # Store original frequency info
        self.original_freq = self.env.th.traj.info.frequency
        self.effective_freq = getattr(self.cfg, "control_freq", self.original_freq)

        self._metadata = self._discover_metadata()

    def _setup_cache(self):
        cache_path = os.path.expanduser("~/.loco-mujoco-caches")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        os.environ["LOCO_MUJOCO_CACHE"] = cache_path

    def _load_env(self, **kwargs):
        # Only create confs if the list is non-empty, otherwise set to None
        default_conf = (
            DefaultDatasetConf(self.dataset_dict["default"])
            if self.dataset_dict["default"]
            else None
        )
        lafan1_conf = (
            LAFAN1DatasetConf(self.dataset_dict["lafan1"])
            if self.dataset_dict["lafan1"]
            else None
        )
        amass_conf = (
            AMASSDatasetConf(self.dataset_dict["amass"])
            if self.dataset_dict["amass"]
            else None
        )
        env = ImitationFactory.make(
            self.env_name,
            default_dataset_conf=default_conf,  # type: ignore
            lafan1_dataset_conf=lafan1_conf,  # type: ignore
            amass_dataset_conf=amass_conf,  # type: ignore
            **kwargs,
        )

        self.num_traj = len(env.th.traj.data.split_points) - 1  # type: ignore

        self.split_points = env.th.traj.data.split_points  # type: ignore

        return env

    def _discover_metadata(self) -> DatasetMeta:
        """
        TODO: this is useless for now.
        """
        traj_info: dict[str, Any] = self.env.th.traj.info.to_dict()  # type: ignore

        traj_data_keys = [
            "qpos",
            "qvel",
            "xpos",
            "xquat",
            "cvel",
            "subtree_com",
            "site_xpos",
            "site_xmat",
        ]

        traj_data_lengths: list[int] = [
            self.env.th.traj.data.split_points[i + 1]  # type: ignore
            - self.env.th.traj.data.split_points[i]  # type: ignore
            for i in range(len(self.env.th.traj.data.split_points) - 1)  # type: ignore
        ]  # type: ignore

        return DatasetMeta(
            name="loco_mujoco",
            source="loco_mujoco",
            version="1.0.1",
            citation="TODO",
            num_trajectories=self.env.th.n_trajectories,  # type: ignore
            keys=traj_data_keys,
            trajectory_lengths=traj_data_lengths,
            dt=1.0 / traj_info["frequency"],
            joint_names=traj_info["joint_names"],
            body_names=traj_info["body_names"],
            site_names=traj_info["site_names"],
            metadata=traj_info["metadata"],
        )

    @property
    def metadata(self) -> DatasetMeta:
        return self._metadata

    def __len__(self) -> int:
        return self.num_traj

    def __getitem__(self, idx: int) -> None:
        return None

    def as_dataset(self, **kwargs) -> BaseDataset:
        raise NotImplementedError(
            "LocoMuJoCoLoader.as_dataset is no longer supported. "
            "Export trajectories to Zarr via `save(...)` and build a replay "
            "buffer using `iltools_datasets.offline.OfflineDataset` and "
            "`build_replay_from_zarr` instead."
        )

    def save(self, path: str, **kwargs) -> None:
        """
        Saves the dataset to a directory. The dataset format is a Zarr store.
        The structure is as follows:
        Dataset/
        ├── motion1/  # e.g., default_walk
        │   ├── trajectory_0/
        │   │   ├── qpos/
        │   │   ├── qvel/
        │   │   ├── xpos/
        │   │   ├── xquat/
        │   │   ├── cvel/
        │   │   ├── subtree_com/
        │   │   ├── site_xpos/
        │   │   ├── site_xmat/
        │   ├── trajectory_1/
        │   └── ...
        └── ...

        However, the loco-mujoco dataset only assigns one trajectory to each motion.

        Transition export (optional):
          If `export_transitions=True` is passed via kwargs, this method will attempt
          to export per-step transition tuples (obs, action, next_obs) using the
          environment's observation function and the controller actions recorded
          by Loco-MuJoCo. The arrays will be saved under keys provided by
          `obs_key_name` (default: 'obs') and `action_key_name` (default: 'action').

          Notes:
            - This requires Loco-MuJoCo to expose the necessary APIs to reconstruct
              observations and actions for each step. If unavailable, a
              NotImplementedError is raised.
            - The Zarr → Replay pipeline in `replay_export.py` will pick up these
              keys automatically and include an additional `lmj_observation` view
              in the replay TensorDicts.
        """

        chunk_size: int = kwargs.get("chunk_size", 10)
        shard_size: int = kwargs.get("shard_size", 100)
        if not os.path.exists(path):
            os.makedirs(path)
        store = LocalStore(path)
        root = zarr.group(store=store, overwrite=False)
        locomujoco_group = root.create_group("loco_mujoco")

        # Need groupings because trajectories are in flat order in loco-mujoco, rather than grouped by dataset type
        groupings: list[str] = []
        for dataset_type, motions in self.dataset_dict.items():
            if not motions:
                continue
            for motion in motions:
                trajectory_name = f"{dataset_type}_{motion}"
                groupings.append(trajectory_name)

        # Save info, which is the same for all trajectories
        traj_info: dict[str, Any] = self.env.th.traj.info.to_dict()  # type: ignore
        for key, value in traj_info.items():
            if key in ["model", "metadata"]:
                continue
            locomujoco_group.attrs[key] = value

        # Save trajectories
        for idx, trajectory_name in enumerate(groupings):
            motion_group = locomujoco_group.create_group(trajectory_name)
            # only one trajectory per motion
            # TODO: consider splitting into multiple trajectories per motion
            trajectory_group = motion_group.create_group("trajectory_0")
            traj_start = self.env.th.traj.data.split_points[idx]  # type: ignore
            traj_end = self.env.th.traj.data.split_points[idx + 1]  # type: ignore

            for key, value in self.env.th.traj.to_dict().items():  # type: ignore
                if (
                    value is None
                    or isinstance(value, list)
                    or isinstance(value, int)
                    or isinstance(value, float)
                    or key
                    not in [
                        "qpos",
                        "qvel",
                        "xpos",
                        "xquat",
                        "cvel",
                        "subtree_com",
                        "site_xpos",
                        "site_xmat",
                    ]
                ):
                    continue
                # create chunked array
                sliced_value = np.array(value[traj_start:traj_end])
                chunks: list[int] = [chunk_size] + list(sliced_value.shape[1:])
                shards: list[int] = [shard_size] + list(sliced_value.shape[1:])
                trajectory_data = trajectory_group.create_dataset(
                    key,
                    shape=sliced_value.shape,
                    dtype=sliced_value.dtype,
                    chunks=chunks,
                    shards=shards,
                )
                trajectory_data[:] = sliced_value

            # Optionally export (obs, action, next_obs) per-step transitions
            if kwargs.get("export_transitions", False):
                obs_key_name = kwargs.get("obs_key_name", "obs")
                next_obs_key_name = kwargs.get("next_obs_key_name", "next_obs")
                action_key_name = kwargs.get("action_key_name", "action")

                # Ensure transitions are available; if not, create them
                transitions = self.env.th.traj.transitions  # type: ignore
                if transitions is None:
                    transitions = self.env.create_dataset()

                # Convert to numpy if needed
                try:
                    observations = np.asarray(transitions.observations)
                    next_observations = np.asarray(transitions.next_observations)
                    actions = (
                        np.asarray(transitions.actions)
                        if transitions.actions.size > 0
                        else None
                    )
                    absorbings = (
                        np.asarray(transitions.absorbings)
                        if transitions.absorbings.size > 0
                        else None
                    )
                    dones = (
                        np.asarray(transitions.dones)
                        if transitions.dones.size > 0
                        else None
                    )
                except Exception:
                    # transitions may be jax arrays; convert via provided helpers
                    try:
                        observations = transitions.to_np().observations  # type: ignore
                        next_observations = transitions.to_np().next_observations  # type: ignore
                        actions = transitions.to_np().actions if transitions.actions.size > 0 else None  # type: ignore
                        absorbings = transitions.to_np().absorbings if transitions.absorbings.size > 0 else None  # type: ignore
                        dones = transitions.to_np().dones if transitions.dones.size > 0 else None  # type: ignore
                    except Exception as e:  # pragma: no cover - unexpected type
                        raise RuntimeError(
                            "Failed to convert transitions to numpy arrays"
                        ) from e

                # Slice out this trajectory's segment: number of transitions is (traj_len-1)
                traj_len = traj_end - traj_start
                # Compute offset as sum_{k < idx} (len_k - 1). We accumulate on the fly.
                if idx == 0:
                    prev_transitions = 0
                else:
                    # Recompute prior transitions from split_points for robustness
                    prev_transitions = 0
                    for k in range(idx):
                        s = self.env.th.traj.data.split_points[k]  # type: ignore
                        e = self.env.th.traj.data.split_points[k + 1]  # type: ignore
                        prev_transitions += max(0, int(e - s) - 1)

                n_this = max(0, int(traj_len) - 1)
                if n_this <= 0:
                    continue

                obs_slice = observations[prev_transitions : prev_transitions + n_this]
                next_obs_slice = next_observations[
                    prev_transitions : prev_transitions + n_this
                ]
                actions_slice = (
                    None
                    if actions is None
                    else actions[prev_transitions : prev_transitions + n_this]
                )
                absorb_slice = (
                    None
                    if absorbings is None
                    else absorbings[prev_transitions : prev_transitions + n_this]
                )
                dones_slice = (
                    None
                    if dones is None
                    else dones[prev_transitions : prev_transitions + n_this]
                )

                # Write to Zarr under trajectory group
                traj_obs = trajectory_group.create_dataset(
                    obs_key_name, shape=obs_slice.shape, dtype=obs_slice.dtype
                )
                traj_obs[:] = obs_slice
                traj_next_obs = trajectory_group.create_dataset(
                    next_obs_key_name,
                    shape=next_obs_slice.shape,
                    dtype=next_obs_slice.dtype,
                )
                traj_next_obs[:] = next_obs_slice
                if actions_slice is not None:
                    traj_act = trajectory_group.create_dataset(
                        action_key_name,
                        shape=actions_slice.shape,
                        dtype=actions_slice.dtype,
                    )
                    traj_act[:] = actions_slice
                if absorb_slice is not None:
                    traj_abs = trajectory_group.create_dataset(
                        "absorbing", shape=absorb_slice.shape, dtype=absorb_slice.dtype
                    )
                    traj_abs[:] = absorb_slice
                if dones_slice is not None:
                    traj_done = trajectory_group.create_dataset(
                        "done", shape=dones_slice.shape, dtype=dones_slice.dtype
                    )
                    traj_done[:] = dones_slice
