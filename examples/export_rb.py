#!/usr/bin/env python
# coding: utf-8

# # Export ReplayBuffer from LoCo-MuJoCo (UnitreeG1, default:walking)
# 
# - Loads Unitree G1 in LoCo-MuJoCo and selects the 'walking' motion from the default dataset (alias: 'walk').
# - Uses `TrajectoryDatasetManager` to create Zarr if missing and to step reference data.
# - Builds a TorchRL memmap-backed replay buffer from the saved Zarr.
# - Demonstrates step-wise sequential sampling and random minibatch sampling.
# 

# In[1]:


import os, gc, json, sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

# Repo import for local package
repo_root = Path.cwd().parent
if (repo_root / 'src').exists():
    sys.path.insert(0, str(repo_root/ 'src'))

from iltools_datasets.loco_mujoco.loader import LocoMuJoCoLoader
from iltools_datasets.manager import TrajectoryDatasetManager
from iltools_datasets.replay_export import build_replay_from_zarr
from iltools_datasets.replay_manager import EnvAssignment

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', DEVICE)


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Configure LoCo-MuJoCo (G1 walking)

# In[3]:


# LoCo-MuJoCo base cfg (see tests for example usage)
basic_cfg = DictConfig({
    'dataset': { 'trajectories': { 'default': ['walk'], 'amass': [], 'lafan1': [] } },
    'control_freq': 50.0,
    'window_size': 4,
    'sim': { 'dt': 0.001 },
    'decimation': 20,
})

# Resolve joint names via loader metadata to satisfy manager mapping
tmp_loader = LocoMuJoCoLoader(env_name='UnitreeG1', cfg=basic_cfg)
joint_names = list(tmp_loader.metadata.joint_names)
print('Found', len(joint_names), 'joints')
print("joint_names", joint_names)
del tmp_loader
joint_names = joint_names[1:] # no need for root joint

# Paths
DATA_DIR = Path(repo_root)/'examples'/ 'data'/ 'g1_default_walk'
DATA_DIR.mkdir(parents=True, exist_ok=True)
ZARR_PATH = DATA_DIR/ 'trajectories.zarr'

# Manager cfg (creates Zarr if missing using LocoMuJoCoLoader)
mgr_cfg = DictConfig({
    'dataset_path': str(DATA_DIR),
    'dataset': { 'trajectories': { 'default': ['walk'], 'amass': [], 'lafan1': [] } },
    'loader_type': 'loco_mujoco',
    'loader_kwargs': { 'env_name': 'UnitreeG1', 'cfg': basic_cfg },
    'assignment_strategy': 'sequential',
    'window_size': 4,
    'target_joint_names': joint_names,
    'reference_joint_names': joint_names,
})


# In[4]:


manager = TrajectoryDatasetManager(cfg=mgr_cfg, num_envs=8, device="cuda:0")
manager.reset_trajectories()
ref0 = manager.get_reference_data()
print("Reference data keys:", list(ref0.keys()))
print("Zarr ready at:", ZARR_PATH)


# ## Build memmap-backed replay buffer from Zarr

# In[5]:


# Export replay using qpos as observation; action auto-detected if present
replay_mgr = build_replay_from_zarr(
    zarr_path=str(ZARR_PATH),
    scratch_dir=str(DATA_DIR/ 'memmap'),
    obs_keys=['qpos'],
    concat_obs_to_key='observation',
    include_terminated=True, include_truncated=True,
)
print('Replay transitions available:', len(replay_mgr.buffer))


# In[6]:


buffer = replay_mgr.buffer
buffer


# In[ ]:





# ## Step-wise sequential sampler

# In[7]:


# Build a simple assignment: all 8 envs read the first segment (task 0, traj 0)
asg = [EnvAssignment(task_id=0, traj_id=0, step=i) for i in range(8)]
replay_mgr.set_assignment(asg)
# Sample twice; each env advances one step and wraps per-trajectory
b1 = replay_mgr.buffer.sample()
b2 = replay_mgr.buffer.sample()
print('Sequential batch1 shape:', {k: tuple(v.shape) for k,v in b1.items() if isinstance(v, torch.Tensor)})
print('Sequential batch2 observation head:', b2['observation'][:2])


# ## Random minibatch sampler

# In[8]:


# Switch to uniform random minibatch sampler (without replacement)
replay_mgr.set_uniform_sampler(batch_size=1024, without_replacement=True)
rb_batch = replay_mgr.buffer.sample()
print('Random minibatch size:', rb_batch.batch_size)
print('Keys:', list(rb_batch.keys(True)))


# ## Replay Buffer Examples and Tests

# In[9]:


# Inspect segment metadata and presence of auxiliary keys
print('Num segments:', len(replay_mgr.segments))
print('First 3 segments:', [ (s.task_id, s.traj_id, s.length) for s in replay_mgr.segments[:3] ])
batch = replay_mgr.buffer.sample()
print('Sampled keys:', list(batch.keys(True)))
print('Has terminated?', 'terminated' in batch, 'Has truncated?', 'truncated' in batch)
print('Batch sizes:', {k: tuple(v.shape) for k,v in batch.items() if hasattr(v, 'shape')})


# ### Step-wise Sequential Sampler (assignment and wraparound)

# In[10]:


# Assign all envs to the first segment and stagger start steps to show progression
first_seg = replay_mgr.segments[0]
num_envs = 6
asg = [EnvAssignment(task_id=first_seg.task_id, traj_id=first_seg.traj_id, step=i) for i in range(num_envs)]
replay_mgr.set_assignment(asg)
b1 = replay_mgr.buffer.sample()
b2 = replay_mgr.buffer.sample()
print('Sequential sampler shapes:', b1['observation'].shape, b2['observation'].shape)
print('First env obs head b1/b2:', b1['observation'][0, :4], b2['observation'][0, :4])
# Note: Actual values depend on dataset; we demonstrate API and per-call advancement.


# ### Random Minibatch Sampler

# In[11]:


# Switch to uniform minibatching (without replacement)
replay_mgr.set_uniform_sampler(batch_size=512, without_replacement=True)
rb_batch = replay_mgr.buffer.sample()
print('Uniform minibatch size:', rb_batch.batch_size)
# With replacement (sampler=None)
replay_mgr.set_uniform_sampler(batch_size=128, without_replacement=False)
rb_batch_rep = replay_mgr.buffer.sample()
print('With-replacement minibatch size:', rb_batch_rep.batch_size)


# ### Device Transform (prefetch to GPU if available)

# In[12]:


# Force reload of the replay_manager module to pick up latest changes
import importlib
import iltools_datasets.replay_manager
importlib.reload(iltools_datasets.replay_manager)

target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Target device:', target_device)
replay_mgr.set_device_transform(target_device)
bs = replay_mgr.buffer.sample()
print('Device:', bs['observation'].device)
print('Device types match:', bs['observation'].device.type == target_device.type)
if bs['observation'].device.type == target_device.type:
    print('✅ Device transform ok')
else:
    print('❌ Device transform failed - data still on CPU')
    print('This might be due to buffer recreation losing transforms. Check replay_manager.py')


# ### Synthetic Tests (deterministic observations)

# In[13]:


from iltools_datasets.replay_manager import ExpertReplayManager, ExpertReplaySpec
from iltools_datasets.replay_memmap import build_trajectory_td, Segment

def _mk_traj(task_id: int, traj_id: int, T: int, obs_dim: int = 3, act_dim: int = 1):
    t = torch.arange(T, dtype=torch.float32).unsqueeze(-1)
    obs = torch.cat([torch.full_like(t, float(task_id)), torch.full_like(t, float(traj_id)), t], dim=1)
    nxt = obs + 0.5
    act = torch.zeros(T, act_dim)
    return build_trajectory_td(observation=obs, action=act, next_observation=nxt)

# Build small tasks set
tasks = {0: [_mk_traj(0,0, T=3)], 1: [_mk_traj(1,0, T=2)]}
tmp_dir = str((Path.cwd()/'_tmp_memmap').absolute())
mgr2 = ExpertReplayManager(ExpertReplaySpec(tasks=tasks, scratch_dir=tmp_dir, device='cpu', sample_batch_size=4))

# Sequential assignment for 3 envs
asg = [EnvAssignment(0,0,0), EnvAssignment(1,0,0), EnvAssignment(0,0,2)]
mgr2.set_assignment(asg)
out = mgr2.buffer.sample()
obs = out['observation']
assert torch.allclose(obs[0], torch.tensor([0.0,0.0,0.0]))
assert torch.allclose(obs[1], torch.tensor([1.0,0.0,0.0]))
assert torch.allclose(obs[2], torch.tensor([0.0,0.0,2.0]))
out2 = mgr2.buffer.sample()
obs2 = out2['observation']
assert torch.allclose(obs2[0], torch.tensor([0.0,0.0,1.0]))
assert torch.allclose(obs2[1], torch.tensor([1.0,0.0,1.0]))
assert torch.allclose(obs2[2], torch.tensor([0.0,0.0,0.0]))
print('✅ Sequential sampler synthetic test passed')

# Uniform samplers
mgr2.set_uniform_sampler(batch_size=5, without_replacement=True)
u1 = mgr2.buffer.sample(); assert u1.batch_size[0] == 5
mgr2.set_uniform_sampler(batch_size=3, without_replacement=False)
u2 = mgr2.buffer.sample(); assert u2.batch_size[0] == 3
print('✅ Uniform sampler tests passed')


# ### Zarr Dataset Summary

# In[14]:


import zarr, os
if 'ZARR_PATH' not in globals():
    print('ZARR_PATH is undefined. Run the configuration cells first.')
else:
    zp = str(ZARR_PATH)
    if not os.path.exists(zp):
        print('Zarr not found at', zp, '- run earlier cells to create it.')
    else:
        root = zarr.open_group(zp, mode='r')
        print('Zarr:', zp)
        for ds_name in getattr(root, 'group_keys', lambda: list(root.keys()))():
            ds_group = root[ds_name]
            try:
                motions = list(ds_group.group_keys())
            except Exception:
                motions = [k for k in ds_group.keys() if isinstance(ds_group[k], zarr.hierarchy.Group)]
            print(f'- Dataset source: {ds_name} (motions: {len(motions)})')
            for motion in motions:
                mg = ds_group[motion]
                try:
                    trajs = list(mg.group_keys())
                except Exception:
                    trajs = [k for k in mg.keys() if isinstance(mg[k], zarr.hierarchy.Group)]
                lengths = []
                for traj in trajs:
                    tg = mg[traj]
                    # Prefer 'qpos' to determine T, else first array key
                    arr_key = 'qpos' if 'qpos' in tg else next((k for k in tg.keys() if isinstance(tg[k], zarr.core.Array)), None)
                    T = int(tg[arr_key].shape[0]) if arr_key is not None else -1
                    lengths.append(T)
                total_T = sum(max(0, T) for T in lengths)
                print(f'  • Motion: {motion:>20} | trajs: {len(trajs):3d} | mean T: { (sum(lengths)/len(lengths)) if lengths else 0:.1f} | total T: {total_T}')

