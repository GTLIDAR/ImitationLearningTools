import threading
from typing import Callable, Dict, Iterable, Optional

import jax
import jax.numpy as jp
import numpy as np


class DoubleBufferStreamer:
    """Double-buffered prefetch of fixed-shape windows to device for JAX.

    Host thread fetches windows via a callable (numpy dict with fixed shapes)
    and places them into two alternating device buffers. The jitted step
    consumes from one buffer while the host fills the other.
    """

    def __init__(
        self,
        window_fn: Callable[[], Dict[str, np.ndarray]],
        device: Optional[jax.Device] = None,
    ):
        self._window_fn = window_fn
        self._device = device
        self._buffers = [None, None]
        self._ready = [threading.Event(), threading.Event()]
        self._stop = threading.Event()
        self._producer = threading.Thread(target=self._run, daemon=True)
        self._produce_idx = 0
        self._consume_idx = 0

    def start(self):
        self._producer.start()

    def _run(self):
        while not self._stop.is_set():
            buf_idx = self._produce_idx % 2
            # Wait until consumer clears this buffer
            if self._ready[buf_idx].is_set():
                self._stop.wait(0.0005)
                continue
            window = self._window_fn()  # numpy dict
            # Transfer to device
            dev = self._device or jax.devices()[0]
            self._buffers[buf_idx] = {
                k: jax.device_put(v, dev) for k, v in window.items()
            }
            self._ready[buf_idx].set()
            self._produce_idx += 1

    def next_window(self) -> Dict[str, jax.Array]:
        buf_idx = self._consume_idx % 2
        # Wait for current buffer to be ready
        self._ready[buf_idx].wait()
        out = self._buffers[buf_idx]
        # Mark consumed
        self._ready[buf_idx].clear()
        self._consume_idx += 1
        return out

    def stop(self):
        self._stop.set()
        self._producer.join(timeout=1.0)
