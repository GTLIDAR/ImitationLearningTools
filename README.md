
# imitation-learning-tools

Scaffold for a modular imitation learning toolkit with dataset loaders, retargeting, and CLI utilities.

This is **only a skeleton**—fill in each module with real logic as you develop.

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
