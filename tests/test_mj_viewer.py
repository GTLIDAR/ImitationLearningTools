import pytest

# Check if installation was succesful, skip if not available.
try:
    print("Checking that the installation succeeded:")
    import mujoco
    mujoco.MjModel.from_xml_string("<mujoco/>")
except Exception:
    pytest.skip("mujoco not installed or unusable in this environment", allow_module_level=True)

# Other imports and helper functions
import time
import itertools
import numpy as np
import os

import mediapy as media
import matplotlib.pyplot as plt

xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)
with mujoco.Renderer(model) as renderer:
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=scene_option)
            pixels = renderer.render()
            frames.append(pixels)

# media.show_video(frames, fps=framerate)
# Create log directory if it doesn't exist and save the video there
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
output_path = os.path.join(log_dir, "test_mj_viewer.mp4")
media.write_video(output_path, frames, fps=framerate)
