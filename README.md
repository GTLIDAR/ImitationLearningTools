
# imitation-learning-tools

Scaffold for a modular imitation learning toolkit with dataset loaders, retargeting, and CLI utilities.

This is **only a skeleton**—fill in each module with real logic as you develop.

The structure of a dataset is as follows:

```
Dataset/
├── Motion1/
│   ├── Trajectory1/
│   │   ├── observations.npy
│   │   ├── qpos.npy
│   │   └── qvel.npy
│   ├── actions.npy
│   ├── rewards.npy
│   └── infos.pkl
│   └── Trajectory2/
│       ├── observations.npy
│       ├── qpos.npy
│       └── qvel.npy
├── Motion2/
│   └── Trajectory1/
│       ├── observations.npy
│       ├── qpos.npy
│       └── qvel.npy
└── Motion3/
    └── Trajectory1/
        ├── observations.npy
        ├── qpos.npy
        └── qvel.npy
```
