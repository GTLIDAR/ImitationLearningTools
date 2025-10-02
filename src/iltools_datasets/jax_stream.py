from typing import Callable, Dict, Optional

import jax
import numpy as np


class DoubleBufferStreamer:
    """Simple synchronous streamer for fixed-shape windows to device for JAX.

    Fetches windows via a callable (numpy dict with fixed shapes)
    and transfers them to device on-demand without buffering or threading.
    """

    def __init__(
        self,
        window_fn: Callable[[], Dict[str, np.ndarray]],
        device: Optional[jax.Device] = None,
    ):
        self._window_fn = window_fn
        self._device = device or jax.devices()[0]

    def start(self):
        """No-op for compatibility with original API."""
        pass

    def next_window(self) -> Dict[str, jax.Array]:
        """Fetch and transfer next window to device synchronously."""
        window = self._window_fn()  # numpy dict
        # Transfer to device
        return {k: jax.device_put(v, self._device) for k, v in window.items()}

    def stop(self):
        """No-op for compatibility with original API."""
        pass
