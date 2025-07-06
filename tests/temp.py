import pytest
import numpy as np
import torch
import os
import mujoco
from typing import Optional
from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets import VectorizedTrajectoryDataset
from iltools_core.metadata_schema import DatasetMeta
from omegaconf import DictConfig

# def test_loco_mujoco_loader(tmp_path, **kwargs):
#     loader = LocoMuJoCoLoader(
#         env_name="UnitreeG1",
#         cfg=DictConfig(
#             {
#                 "dataset": {
#                     "trajectories": {
#                         "default": ["walk", "squat"],
#                         "lafan1": ["dance2_subject4", "walk1_subject1"],
#                         "amass": [],
#                     }
#                 }
#             }
#         ),
#         **kwargs,
#     )
#     assert loader.env_name == "UnitreeG1"
#     assert loader.env is not None
#     assert loader.env.th is not None
#     assert loader.env.th.traj is not None
#     assert loader.env.th.traj.info is not None
#     assert loader.env.th.traj.data is not None
#     loader.save(tmp_path / "test.zarr")


def test_vectorized_trajectory_dataset(tmp_path, **kwargs):
    loader = LocoMuJoCoLoader(
        env_name="UnitreeG1",
        cfg=DictConfig(
            {
                "dataset": {
                    "trajectories": {
                        "default": ["walk", "squat"],
                        "lafan1": [],
                        "amass": [],
                    }
                }
            }
        ),
        **kwargs,
    )
    loader.save(tmp_path / "test.zarr")
    dataset = VectorizedTrajectoryDataset(
        zarr_path=tmp_path / "test.zarr",
        num_envs=10,
        cfg=DictConfig(
            {
                "window_size": 10,
            }
        ),
    )

    a = dataset.fetch(idx=[0] * 10, key="qpos")
    print(a.shape)
