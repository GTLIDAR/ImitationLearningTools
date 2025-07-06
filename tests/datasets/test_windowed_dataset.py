import pytest
import tempfile
import shutil
import os
import json
from typing import Dict, Any
import asyncio
import time

import numpy as np
import torch
import zarr
from tensordict import TensorDict

from iltools_datasets.storage import VectorizedTrajectoryDataset
from iltools_datasets.base_loader import BaseLoader
from iltools_core.trajectory import Trajectory
from iltools_core.metadata_schema import DatasetMeta


class TestAsyncLRUCache:
    """Test the async LRU cache component."""

    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        cache = _AsyncLRUCache(capacity=3)

        # Test empty cache
        assert cache.get((0, 0)) is None

        # Test put and get
        dummy_tensor = TensorDict({"test": torch.randn(5, 10)}, batch_size=[5])
        cache.put((0, 0), dummy_tensor)
        retrieved = cache.get((0, 0))

        assert retrieved is not None
        assert torch.allclose(retrieved["test"], dummy_tensor["test"])

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        cache = _AsyncLRUCache(capacity=2)

        # Fill cache to capacity
        tensor1 = TensorDict({"data": torch.ones(5, 10)}, batch_size=[5])
        tensor2 = TensorDict({"data": torch.ones(5, 10) * 2}, batch_size=[5])
        tensor3 = TensorDict({"data": torch.ones(5, 10) * 3}, batch_size=[5])

        cache.put((0, 0), tensor1)
        cache.put((0, 1), tensor2)

        # Access first item to make it recently used
        cache.get((0, 0))

        # Add third item - should evict (0, 1)
        cache.put((0, 2), tensor3)

        assert cache.get((0, 0)) is not None  # Still there
        assert cache.get((0, 1)) is None  # Evicted
        assert cache.get((0, 2)) is not None  # New item

    def test_async_prefetch(self):
        """Test async prefetching functionality."""
        cache = _AsyncLRUCache(capacity=10)

        def mock_load_func(traj_idx: int, start_idx: int) -> TensorDict:
            """Mock loading function that simulates work."""
            time.sleep(0.01)  # Simulate I/O
            return TensorDict(
                {"data": torch.ones(5, 10) * (traj_idx + start_idx)}, batch_size=[5]
            )

        # Submit async prefetch
        cache.prefetch_async((1, 5), mock_load_func)

        # Should return None immediately (not ready)
        result = cache.get((1, 5))
        if result is None:
            # Wait a bit and try again
            time.sleep(0.02)
            result = cache.get((1, 5))

        assert result is not None
        expected_value = 1 + 5  # traj_idx + start_idx
        assert torch.allclose(result["data"], torch.ones(5, 10) * expected_value)


class TestWindowedTrajectoryDataset:
    """Test the WindowedTrajectoryDataset with all optimizations."""

    @pytest.fixture
    def sample_zarr_dataset(self):
        """Create a sample Zarr dataset for testing."""
        temp_dir = tempfile.mkdtemp()
        zarr_path = os.path.join(temp_dir, "trajectories.zarr")

        # Create sample data
        num_trajectories = 5
        max_traj_length = 100

        # Create zarr arrays
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store=store, overwrite=True)

        # Generate trajectory lengths
        traj_lengths = [50, 75, 100, 60, 80]

        # Create observation data
        obs_group = root.create_group("observations")
        qpos_data = obs_group.zeros(
            "qpos", shape=(num_trajectories, max_traj_length, 30), dtype=np.float32
        )
        qvel_data = obs_group.zeros(
            "qvel", shape=(num_trajectories, max_traj_length, 29), dtype=np.float32
        )

        # Create action data
        act_group = root.create_group("actions")
        action_data = act_group.zeros(
            "actions", shape=(num_trajectories, max_traj_length, 12), dtype=np.float32
        )

        # Fill with meaningful data
        for traj_idx in range(num_trajectories):
            length = traj_lengths[traj_idx]

            # Fill qpos with trajectory-specific patterns
            for t in range(length):
                # COM position (first 3 elements)
                qpos_data[traj_idx, t, :3] = [t * 0.1, 0, 0.5 + 0.01 * np.sin(t * 0.1)]
                # COM orientation (next 4 elements) - identity quaternion
                qpos_data[traj_idx, t, 3:7] = [1, 0, 0, 0]
                # Joint angles (remaining elements)
                qpos_data[traj_idx, t, 7:] = np.sin(t * 0.1 + np.arange(23) * 0.1) * 0.5

            # Fill qvel with velocity patterns
            for t in range(length):
                qvel_data[traj_idx, t, :3] = [0.1, 0, 0.001 * np.cos(t * 0.1)]
                qvel_data[traj_idx, t, 3:6] = [0, 0, 0.01 * np.sin(t * 0.1)]
                qvel_data[traj_idx, t, 6:] = (
                    np.cos(t * 0.1 + np.arange(23) * 0.1) * 0.05
                )

            # Fill actions
            for t in range(length):
                action_data[traj_idx, t, :] = (
                    np.sin(t * 0.2 + np.arange(12) * 0.2) * 0.1
                )

        # Create metadata
        metadata = {
            "num_trajectories": num_trajectories,
            "trajectory_lengths": traj_lengths,
            "observation_keys": ["qpos", "qvel"],
            "action_keys": ["actions"],
            "window_size": 32,
            "export_control_freq": 50.0,
            "original_frequency": 100.0,
            "effective_frequency": 50.0,
            "dt": [0.02 for _ in range(num_trajectories)],
        }

        metadata_path = os.path.join(temp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        yield temp_dir, traj_lengths

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_dataset_initialization(self, sample_zarr_dataset):
        """Test dataset initialization and shape detection."""
        data_dir, traj_lengths = sample_zarr_dataset

        dataset = WindowedTrajectoryDataset(
            data_dir=data_dir,
            window_size=32,
            device="cpu",
            cache_capacity=100,
            prefetch_ahead=2,
        )

        assert dataset.window_size == 32
        assert len(dataset.lengths) == 5
        assert dataset.lengths == traj_lengths
        assert "qpos" in dataset.observation_keys
        assert "qvel" in dataset.observation_keys
        assert "actions" in dataset.action_keys

        # Check shape detection
        assert dataset._obs_shapes["qpos"] == (30,)
        assert dataset._obs_shapes["qvel"] == (29,)
        assert dataset._act_shapes["actions"] == (12,)

    def test_single_key_extraction(self, sample_zarr_dataset):
        """Test optimized single key extraction."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(data_dir, window_size=16)

        batch_size = 10
        traj_indices = np.random.randint(0, 5, batch_size)
        step_indices = np.random.randint(0, 20, batch_size)

        # Test single key extraction
        qpos_batch = dataset.batch_get(
            traj_indices, step_indices, key="qpos", data_type="observations"
        )

        assert isinstance(qpos_batch, torch.Tensor)
        assert qpos_batch.shape == (batch_size, 30)

        # Verify data correctness
        for i in range(batch_size):
            traj_idx, step_idx = traj_indices[i], step_indices[i]
            # Check COM position progression
            expected_x = step_idx * 0.1
            assert abs(qpos_batch[i, 0].item() - expected_x) < 1e-5

    def test_small_batch_cached_loading(self, sample_zarr_dataset):
        """Test small batch loading with caching."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(data_dir, window_size=16, cache_capacity=50)

        batch_size = 4  # Small batch
        traj_indices = np.array([0, 1, 2, 3])
        step_indices = np.array([10, 15, 20, 5])

        # First call - should populate cache
        data1 = dataset.batch_get(traj_indices, step_indices)

        assert isinstance(data1, TensorDict)
        assert data1.batch_size == (batch_size,)
        assert "observations/qpos" in data1
        assert "observations/qvel" in data1
        assert "actions/actions" in data1

        # Verify shapes
        assert data1["observations/qpos"].shape == (batch_size, 16, 30)
        assert data1["observations/qvel"].shape == (batch_size, 16, 29)
        assert data1["actions/actions"].shape == (batch_size, 16, 12)

        # Second call with same indices - should use cache
        start_time = time.time()
        data2 = dataset.batch_get(traj_indices, step_indices)
        cache_time = time.time() - start_time

        # Should be much faster due to caching
        assert cache_time < 0.01  # Very fast
        assert torch.allclose(data1["observations/qpos"], data2["observations/qpos"])

    def test_large_batch_vectorized_loading(self, sample_zarr_dataset):
        """Test large batch vectorized loading."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(data_dir, window_size=16)

        batch_size = 64  # Large batch
        traj_indices = np.random.randint(0, 5, batch_size)
        step_indices = np.random.randint(0, 30, batch_size)

        start_time = time.time()
        data = dataset.batch_get(traj_indices, step_indices)
        vectorized_time = time.time() - start_time

        assert isinstance(data, TensorDict)
        assert data.batch_size == (batch_size,)
        assert data["observations/qpos"].shape == (batch_size, 16, 30)

        # Verify data correctness for a few samples
        for i in range(min(5, batch_size)):
            traj_idx, step_idx = traj_indices[i], step_indices[i]
            qpos_window = data["observations/qpos"][i]  # (16, 30)

            # Check first timestep COM position
            expected_x = step_idx * 0.1
            assert abs(qpos_window[0, 0].item() - expected_x) < 1e-5

            # Check that COM position progresses in the window
            if step_idx + 15 < traj_lengths[traj_idx]:  # Full window available
                expected_x_end = (step_idx + 15) * 0.1
                assert abs(qpos_window[15, 0].item() - expected_x_end) < 1e-5

        print(
            f"Vectorized loading time for {batch_size} samples: {vectorized_time:.4f}s"
        )

    def test_async_prefetching(self, sample_zarr_dataset):
        """Test async prefetching functionality."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(
            data_dir, window_size=16, prefetch_ahead=3, cache_capacity=100
        )

        # Start with a batch that should trigger prefetching
        traj_indices = np.array([0, 1, 2])
        step_indices = np.array([10, 15, 20])

        # Load current data (should trigger async prefetch of future windows)
        current_data = dataset.batch_get(traj_indices, step_indices)

        # Small delay to let async prefetch work
        time.sleep(0.05)

        # Now access future windows - should be faster due to prefetching
        future_step_indices = step_indices + 2

        start_time = time.time()
        future_data = dataset.batch_get(traj_indices, future_step_indices)
        prefetch_time = time.time() - start_time

        assert isinstance(future_data, TensorDict)
        assert prefetch_time < 0.01  # Should be very fast due to prefetching

        # Verify data correctness
        for i in range(len(traj_indices)):
            traj_idx = traj_indices[i]
            future_step = future_step_indices[i]
            expected_x = future_step * 0.1
            actual_x = future_data["observations/qpos"][i, 0, 0].item()
            assert abs(actual_x - expected_x) < 1e-5

    def test_trajectory_boundary_handling(self, sample_zarr_dataset):
        """Test handling of trajectory boundaries and padding."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(data_dir, window_size=32)

        # Test near end of trajectory
        traj_idx = 0  # Length 50
        step_idx = 45  # Window would go from 45 to 77, but traj is only 50 long

        data = dataset.batch_get(np.array([traj_idx]), np.array([step_idx]))

        assert data["observations/qpos"].shape == (1, 32, 30)

        # Check that data is properly padded (last frame repeated)
        qpos_window = data["observations/qpos"][0]  # (32, 30)

        # Last valid frame should be at index 4 (step 49)
        # Frames 5-31 should be copies of frame 4
        last_valid_frame = qpos_window[4]  # step 49
        for i in range(5, 32):
            assert torch.allclose(qpos_window[i], last_valid_frame, atol=1e-6)

    def test_performance_comparison(self, sample_zarr_dataset):
        """Test performance differences between different batch sizes."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(data_dir, window_size=16)

        # Small batch (should use cache)
        small_batch_size = 8
        small_traj = np.random.randint(0, 5, small_batch_size)
        small_step = np.random.randint(0, 30, small_batch_size)

        start_time = time.time()
        small_data = dataset.batch_get(small_traj, small_step)
        small_time = time.time() - start_time

        # Large batch (should use vectorized)
        large_batch_size = 128
        large_traj = np.random.randint(0, 5, large_batch_size)
        large_step = np.random.randint(0, 30, large_batch_size)

        start_time = time.time()
        large_data = dataset.batch_get(large_traj, large_step)
        large_time = time.time() - start_time

        # Vectorized should be more efficient per sample
        time_per_sample_small = small_time / small_batch_size
        time_per_sample_large = large_time / large_batch_size

        print(f"Small batch: {small_time:.4f}s ({time_per_sample_small:.6f}s/sample)")
        print(f"Large batch: {large_time:.4f}s ({time_per_sample_large:.6f}s/sample)")

        # Large batches should be more efficient per sample
        assert time_per_sample_large < time_per_sample_small * 2  # At most 2x worse

        # Verify data correctness
        assert small_data["observations/qpos"].shape == (small_batch_size, 16, 30)
        assert large_data["observations/qpos"].shape == (large_batch_size, 16, 30)

    def test_memory_efficiency(self, sample_zarr_dataset):
        """Test memory efficiency with preallocated buffers."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(data_dir, window_size=16)

        # Monitor memory usage during large batch operations
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process many batches
        for _ in range(10):
            batch_size = 64
            traj_indices = np.random.randint(0, 5, batch_size)
            step_indices = np.random.randint(0, 30, batch_size)

            data = dataset.batch_get(traj_indices, step_indices)
            del data  # Explicit cleanup

        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase after 10 large batches: {memory_increase:.1f} MB")

        # Should not have significant memory leaks
        assert memory_increase < 100  # Less than 100MB increase

    def test_device_handling(self, sample_zarr_dataset):
        """Test proper device handling for tensors."""
        data_dir, traj_lengths = sample_zarr_dataset

        # Test CPU device
        dataset_cpu = WindowedTrajectoryDataset(data_dir, device="cpu")
        data_cpu = dataset_cpu.batch_get(np.array([0]), np.array([10]))

        assert data_cpu["observations/qpos"].device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            dataset_cuda = WindowedTrajectoryDataset(data_dir, device="cuda")
            data_cuda = dataset_cuda.batch_get(np.array([0]), np.array([10]))

            assert data_cuda["observations/qpos"].device.type == "cuda"

    def test_concurrent_access(self, sample_zarr_dataset):
        """Test concurrent access to the dataset."""
        data_dir, traj_lengths = sample_zarr_dataset
        dataset = WindowedTrajectoryDataset(
            data_dir, window_size=16, cache_capacity=200
        )

        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def worker(worker_id):
            try:
                for _ in range(5):
                    batch_size = 16
                    traj_indices = np.random.randint(0, 5, batch_size)
                    step_indices = np.random.randint(0, 30, batch_size)

                    data = dataset.batch_get(traj_indices, step_indices)
                    results.put((worker_id, data.batch_size))

                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.put((worker_id, str(e)))

        # Start multiple workers
        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)

        # Wait for completion
        for t in threads:
            t.join()

        # Check results
        assert errors.empty(), f"Errors in concurrent access: {list(errors.queue)}"
        assert results.qsize() == 20  # 4 workers * 5 iterations each

        # Verify all results are correct
        while not results.empty():
            worker_id, batch_size = results.get()
            assert batch_size == (16,)


class DummyLoader(BaseLoader):
    def __init__(self, num_trajectories=3, traj_lengths=None):
        if traj_lengths is None:
            traj_lengths = [100, 150, 120]
        self.num_trajectories = num_trajectories
        self.traj_lengths = traj_lengths

        self._metadata = DatasetMeta(
            name="test_dataset",
            source="test",
            citation="test",
            version="1.0.0",
            observation_keys=["qpos", "qvel"],
            action_keys=["actions"],
            trajectory_lengths=traj_lengths,
            num_trajectories=num_trajectories,
            dt=[0.02 for _ in range(num_trajectories)],
        )

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, idx):
        length = self.traj_lengths[idx]
        return Trajectory(
            observations={
                "qpos": torch.randn(length, 10),
                "qvel": torch.randn(length, 10),
            },
            actions={
                "actions": torch.randn(length, 5),
            },
            dt=0.02,
        )


@pytest.fixture
def sample_zarr_dataset():
    """Create a sample per-trajectory Zarr dataset for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy loader
        loader = DummyLoader(num_trajectories=3, traj_lengths=[100, 150, 120])

        # Export to per-trajectory Zarr format
        export_trajectories_to_zarr_per_trajectory(loader, temp_dir)

        yield temp_dir


def test_windowed_dataset_initialization(sample_zarr_dataset):
    """Test that WindowedTrajectoryDataset initializes correctly."""
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset, window_size=32, stride=16, device="cpu", cache_size=10
    )

    assert dataset.window_size == 32
    assert dataset.stride == 16
    assert len(dataset.lengths) == 3
    assert dataset.lengths == [100, 150, 120]


def test_windowed_dataset_window_indexing(sample_zarr_dataset):
    """Test that window indexing works correctly."""
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset, window_size=32, stride=16, device="cpu"
    )

    # Calculate expected number of windows
    expected_windows = 0
    for length in [100, 150, 120]:
        max_start = max(0, length - 32 + 1)  # 69, 119, 89
        windows_in_traj = len(range(0, max_start, 16))
        expected_windows += windows_in_traj

    assert len(dataset) == expected_windows


def test_windowed_dataset_getitem(sample_zarr_dataset):
    """Test that __getitem__ returns correct windows."""
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset, window_size=32, stride=16, device="cpu"
    )

    # Get first window
    window = dataset[0]

    assert isinstance(window, TensorDict)
    assert "qpos" in window
    assert "qvel" in window
    assert "actions" in window
    assert "dt" in window
    assert "traj_idx" in window
    assert "start_idx" in window

    # Check shapes
    assert window["qpos"].shape == (32, 10)
    assert window["qvel"].shape == (32, 10)
    assert window["actions"].shape == (32, 5)

    # Check dt value
    assert torch.allclose(window["dt"], torch.tensor(0.02))


def test_windowed_dataset_trajectory_windows(sample_zarr_dataset):
    """Test getting all windows for a specific trajectory."""
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset, window_size=32, stride=16, device="cpu"
    )

    # Get all windows for trajectory 0 (length 100)
    windows = dataset.get_trajectory_windows(0)

    # Calculate expected number of windows for trajectory 0
    max_start = max(0, 100 - 32 + 1)  # 69
    expected_num_windows = len(range(0, max_start, 16))

    assert len(windows) == expected_num_windows

    # Check that all windows are from trajectory 0
    for window in windows:
        assert window["traj_idx"] == 0


def test_windowed_dataset_cache(sample_zarr_dataset):
    """Test caching functionality."""
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset, window_size=32, stride=16, device="cpu", cache_size=5
    )

    # Access some windows multiple times
    window1_first = dataset[0]
    window1_second = dataset[0]

    # Should be the same tensor (cached)
    assert torch.equal(window1_first["qpos"], window1_second["qpos"])

    # Clear cache
    dataset.clear_cache()


def test_windowed_dataset_info(sample_zarr_dataset):
    """Test dataset info method."""
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset, window_size=32, stride=16, device="cpu"
    )

    info = dataset.get_info()

    assert info["num_trajectories"] == 3
    assert info["window_size"] == 32
    assert info["stride"] == 16
    assert info["trajectory_lengths"] == [100, 150, 120]
    assert info["observation_keys"] == ["qpos", "qvel"]
    assert info["action_keys"] == ["actions"]
    assert len(info["dt_list"]) == 3


def test_windowed_dataset_edge_cases(sample_zarr_dataset):
    """Test edge cases like window size larger than trajectory."""
    # Window size larger than shortest trajectory
    dataset = WindowedTrajectoryDataset(
        sample_zarr_dataset,
        window_size=200,  # Larger than all trajectories
        stride=1,
        device="cpu",
    )

    # Should have no valid windows
    assert len(dataset) == 0


def test_export_and_load_consistency():
    """Test that export and load produce consistent results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create loader with specific data
        loader = DummyLoader(num_trajectories=2, traj_lengths=[50, 75])

        # Get original trajectory data
        orig_traj_0 = loader[0]
        orig_traj_1 = loader[1]

        # Export to Zarr
        export_trajectories_to_zarr_per_trajectory(loader, temp_dir)

        # Load with windowed dataset
        dataset = WindowedTrajectoryDataset(
            temp_dir,
            window_size=25,
            stride=25,  # Non-overlapping windows
            device="cpu",
        )

        # Get first window of trajectory 0
        window = dataset[0]

        # Compare with original data
        assert torch.allclose(window["qpos"], orig_traj_0.observations["qpos"][:25])
        assert torch.allclose(window["qvel"], orig_traj_0.observations["qvel"][:25])
        assert torch.allclose(window["actions"], orig_traj_0.actions["actions"][:25])
