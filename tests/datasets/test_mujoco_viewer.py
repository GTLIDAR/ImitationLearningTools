import os
import sys

import pytest

# Try to import mujoco and mujoco.viewer
try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None
    pytest.skip("mujoco not installed", allow_module_level=True)


def has_display():
    if sys.platform.startswith("linux"):
        return "DISPLAY" in os.environ and bool(os.environ["DISPLAY"])
    return True  # Assume non-Linux platforms have a display


@pytest.mark.visual
@pytest.mark.skipif(not has_display(), reason="No DISPLAY found for GUI/visual test.")
def test_mujoco_viewer_basic():
    """
    Test that the mujoco viewer can open a simple model (a box) and render it.
    This is for manual/visual inspection. Close the window to finish the test.
    """
    xml = """
    <mujoco model="box">
      <worldbody>
        <body name="box" pos="0 0 0.1">
          <geom type="box" size="0.1 0.1 0.1" rgba="0 0.5 1 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch(model, data)
    if viewer is not None:
        with viewer:
            print(
                "A MuJoCo viewer window should open with a blue box. Close it to finish the test."
            )
            # The context manager blocks until the window is closed
    else:
        print("MuJoCo viewer not available (no display)")


@pytest.mark.visual
@pytest.mark.skipif(not has_display(), reason="No DISPLAY found for GUI/visual test.")
def test_mujoco_viewer_pendulum():
    """
    Test that the mujoco viewer can open a simple pendulum model and render it.
    This is for manual/visual inspection. Close the window to finish the test.
    """
    xml = """
    <mujoco model="pendulum">
      <worldbody>
        <body name="pendulum" pos="0 0 0">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 1" size="0.05" rgba="1 0.5 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch(model, data)
    if viewer is not None:
        with viewer:
            print(
                "A MuJoCo viewer window should open with an orange pendulum. Close it to finish the test."
            )
            # The context manager blocks until the window is closed
    else:
        print("MuJoCo viewer not available (no display)")
