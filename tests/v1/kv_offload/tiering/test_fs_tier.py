# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import errno
import logging
import mmap
import os
import runpy
import stat
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import vllm.v1.kv_offload.tiering.fs.io as io_mod
import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs import errors as errors_mod
from vllm.v1.kv_offload.tiering.fs.errors import (
    IOErrorClass,
    annotate,
    classify,
    errno_label,
    log_level_for,
)
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_BLOCKS = 8
_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(
    enable_kv_cache_events: bool = False,
    *,
    tp_size: int = 1,
    rank: int = 0,
    world_size: int | None = None,
    replicated_layout: bool = False,
    is_parallelism_agnostic: bool = False,
) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    if world_size is None:
        world_size = tp_size
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=is_parallelism_agnostic,
        ),
        replicated_layout=replicated_layout,
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    chunk_ids: list[int] | None = None,
    is_promotion: bool = False,
) -> TransferJob:
    if chunk_ids is None:
        chunk_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array(chunk_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=_CTX,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # A successful load must NOT touch the file: the delete path fires only on
    # a provable short read, so a good block stays on disk (guards against an
    # over-eager delete regressing to upstream's delete-on-any-error).
    for k in (key(1), key(2)):
        assert os.path.exists(tier.file_mapper.get_file_name(k))
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()

    # Verify queues are cleared and threads stopped
    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


@pytest.mark.parametrize("batch_size", [0, 1, 2, 5])
@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_load_data_integrity(fs_tier, monkeypatch, use_c_ext, batch_size):
    """Data written by store must be exactly recovered by load, for batches
    of any size -- including the empty batch."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)

    keys = [key(i) for i in range(batch_size)]
    store_chunk_ids = list(range(batch_size))
    load_chunk_ids = list(range(_NUM_BLOCKS - batch_size, _NUM_BLOCKS))
    expected = tensor[:batch_size].clone()

    tier.submit_store(make_job(1, keys, store_chunk_ids))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert store_results[0].success
    assert all(os.path.exists(tier.file_mapper.get_file_name(k)) for k in keys)

    # reset tensor to prove data is read from disk
    tensor[:] = 0.0

    # Load into a range disjoint by index from the store ids, to also
    # exercise loading a chunk into a different id than it was stored from.
    tier.submit_load(make_job(2, keys, load_chunk_ids, is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert load_results[0].success

    for i, cid in enumerate(load_chunk_ids):
        assert torch.allclose(tensor[cid], expected[i]), (
            f"Chunk {cid} data mismatch after store+load"
        )


def test_store_load_roundtrip_without_o_direct(tmp_path, monkeypatch):
    """Buffered fallback must round-trip data when O_DIRECT is unsupported.

    Simulates filesystems (e.g. overlayfs, some NFS) that reject O_DIRECT by
    forcing the capability probe to report it unavailable.
    """
    monkeypatch.setattr(
        "vllm.v1.kv_offload.tiering.fs.manager.probe_o_direct",
        lambda _dir: False,
    )
    tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    try:
        assert tier._use_o_direct is False

        keys = [key(0), key(1)]
        expected = tensor[:2].clone()
        tier.submit_store(make_job(1, keys, [0, 1]))
        assert all(r.success for r in drain(tier))

        tensor[:2] = 0.0
        tier.submit_load(make_job(2, keys, [2, 3], is_promotion=True))
        assert all(r.success for r in drain(tier))

        for i, bid in enumerate([2, 3]):
            assert torch.allclose(tensor[bid], expected[i])
    finally:
        tier.shutdown()


def test_wait_idle_blocks_until_tasks_complete():
    """wait_idle must not return while a task is still in flight."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    gate = threading.Event()
    pool.enqueue_store(job_id=1, n_tasks=1, tasks=[lambda: gate.wait(timeout=5.0)])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_out_of_bounds_block_id_smoke(fs_tier, monkeypatch, use_c_ext):
    """Smoke test: a block id beyond the primary tensor's block count must
    fail the job, for both the C extension and the Python fallback."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    out_of_bounds_bid = tensor.shape[0]  # one past the last valid block

    tier.submit_store(make_job(1, [key(1)], [out_of_bounds_bid]))
    store_results = drain(tier)
    assert len(store_results) == 1
    assert not store_results[0].success

    tier.submit_load(make_job(2, [key(1)], [out_of_bounds_bid], is_promotion=True))
    load_results = drain(tier)
    assert len(load_results) == 1
    assert not load_results[0].success


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_failed_load_corrects_verdict_and_removes_corrupt_file(
    fs_tier, monkeypatch, use_c_ext
):
    """Failed-load livelock regression, covering the whole contract.

    A successful promotion leaves the cached HIT and the on-disk block intact.
    A promotion that short-reads a truncated (corrupt) block fails, and in
    get_finished_jobs() the tier removes the corrupt file (stores are atomic,
    so a too-short file is genuine corruption) and marks the cached verdict
    False. The SAME request's next lookup is then a MISS served from cache with
    NO re-probe, so the scheduler cannot re-issue the doomed promotion.
    """
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))

    ctx = ReqContext(req_id="livelock-req")
    assert lookup_and_wait(tier, [key(1)], ctx=ctx) == [LookupResult.HIT]

    # A successful promotion must NOT touch the verdict or the file.
    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and results[0].success
    assert tier.lookup(key(1), ctx) == LookupResult.HIT
    assert os.path.exists(path)

    # Truncate below block_size so the next promotion short-reads.
    with open(path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(3, [key(1)], [0], is_promotion=True))
    results = drain(tier)  # get_finished_jobs() marks the verdict False here
    assert len(results) == 1 and not results[0].success

    # Corrupt file removed; the SAME request now misses from cache, no re-probe.
    assert not os.path.exists(path)
    lm = tier._lookup_manager
    assert tier.lookup(key(1), ctx) == LookupResult.MISS
    assert lm._lookup_batch == []

    # A FRESH request re-probes the tier (no cached verdict) and misses too,
    # since the corrupt file is gone -- the real batch_lookup re-probe path.
    fresh = ReqContext(req_id="fresh-after-short-read")
    assert lookup_and_wait(tier, [key(1)], ctx=fresh) == [LookupResult.MISS]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_partial_load_failure_keeps_loaded_blocks(
    fs_tier, monkeypatch, use_c_ext
):
    """A batched promotion stops at the first bad block and reports how many
    loaded before it (#50321). Corrupt the LAST block: the earlier blocks load
    fine, so the job reports successful_keys for them and marks only the failed
    tail a miss. The earlier keys stay HIT — including for the same request —
    while the corrupt block stays a MISS (its file was removed)."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # last one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))
    bad_path = tier.file_mapper.get_file_name(key(3))

    ctx = ReqContext(req_id="batch-req")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    # Corrupt only the last block, then load the whole batch as one job.
    with open(bad_path, "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    # (a) the job fails but reports the two blocks that loaded before the bad one.
    assert len(results) == 1 and not results[0].success
    assert tuple(results[0].successful_keys) == (key(1), key(2))

    # (b) Only the failed tail is a miss; the loaded blocks stay HIT on the same
    # request, and nothing was re-probed.
    lm = tier._lookup_manager
    assert [tier.lookup(k, ctx) for k in keys] == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]
    assert lm._lookup_batch == []

    # (c) A fresh request re-probes: the loaded blocks are still on disk (HIT),
    # only the corrupt block was removed (MISS).
    tier.on_request_finished(ctx)
    fresh = ReqContext(req_id="fresh-batch-req")
    assert lookup_and_wait(tier, keys, ctx=fresh) == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
    ]


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batched_load_first_block_fails_marks_whole_batch(
    fs_tier, monkeypatch, use_c_ext
):
    """When the FIRST block fails, nothing loaded before it: the job reports no
    successful_keys (None) and the whole batch is marked a miss for the
    request."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    keys = [key(1), key(2), key(3)]  # first one is the "bad" block
    tier.submit_store(make_job(1, keys, [0, 1, 2]))
    assert all(r.success for r in drain(tier))

    ctx = ReqContext(req_id="batch-first-fail")
    assert lookup_and_wait(tier, keys, ctx=ctx) == [LookupResult.HIT] * 3

    with open(tier.file_mapper.get_file_name(key(1)), "wb") as f:
        f.write(b"x" * 10)
    tier.submit_load(make_job(2, keys, [0, 1, 2], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success
    # Nothing loaded before the failure -> no partial success reported.
    assert results[0].successful_keys is None
    # The whole batch is a miss for this request.
    assert [tier.lookup(k, ctx) for k in keys] == [LookupResult.MISS] * 3


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_transient_load_failure_leaves_file(fs_tier, monkeypatch, use_c_ext):
    """A transient host error (here ELOOP on open) is NOT a short read: the job
    fails but the block file must survive untouched, on both the C and Python
    paths. Deleting on a transient error would turn a passing hiccup into
    permanent data loss."""
    import vllm.v1.kv_offload.tiering.fs.io as io_mod

    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))
    path = tier.file_mapper.get_file_name(key(1))
    with open(path, "rb") as f:
        original = f.read()

    # Make open() fail with ELOOP (fd < 0) without truncating the block. Not
    # chmod 000: CI runs as root, which bypasses permission bits, so open()
    # would succeed and the load would not fail at all.
    saved = path + ".saved"
    loop = path + ".loop"
    os.rename(path, saved)
    os.symlink(loop, path)
    os.symlink(path, loop)

    tier.submit_load(make_job(2, [key(1)], [0], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1 and not results[0].success

    # The path is left alone: a non-short-read error must not unlink.
    assert os.path.lexists(path)

    os.unlink(path)
    os.unlink(loop)
    os.rename(saved, path)
    with open(path, "rb") as f:
        assert f.read() == original


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    assert events[0].medium == Medium.STORAGE
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(
    fs_tier_with_events, monkeypatch
):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(1))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1)], [0]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events, monkeypatch):
    """A store job with any failed block emits no event for the whole job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(2))
    original_batch_store_block = mgr_mod.batch_store_block

    def flaky_batch_store_block(paths, *args, **kwargs):
        if failing_path in paths:
            raise OSError("injected store failure")
        return original_batch_store_block(paths, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "batch_store_block", flaky_batch_store_block)

    tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_chunks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == Medium.STORAGE]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()


def test_fs_tier_cross_tp_round_trip(tmp_path):
    """TP=2 replicated writer and TP=4 reader share namespace and bytes."""
    root = str(tmp_path)
    writer_tensor = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)
    expected = writer_tensor[0].clone()
    writer = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=2, world_size=2, rank=0, replicated_layout=True
        ),
        primary_kv_view=memoryview(writer_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        writer.submit_store(make_job(1, [key(7)], [0]))
        assert all(r.success for r in drain(writer))
        writer_base = writer.file_mapper.base_path
        writer_path = writer.file_mapper.get_file_name(key(7))
    finally:
        writer.shutdown()

    reader_tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    reader = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(
            tp_size=4, world_size=4, rank=3, replicated_layout=True
        ),
        primary_kv_view=memoryview(reader_tensor.numpy()),
        tier_type="fs",
        root_dir=root,
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        assert reader.file_mapper.base_path == writer_base
        assert reader.file_mapper.get_file_name(key(7)) == writer_path
        assert lookup_and_wait(reader, [key(7)]) == [LookupResult.HIT]
        reader.submit_load(make_job(2, [key(7)], [1], is_promotion=True))
        assert all(r.success for r in drain(reader))
        assert torch.allclose(reader_tensor[1], expected)
    finally:
        reader.shutdown()


@pytest.fixture
def vllm_logs():
    """Capture vLLM log records.

    ``caplog`` cannot see them: the ``vllm`` logger sets ``propagate = False``,
    so nothing reaches the root handler pytest installs.
    """

    class _Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.records.append(record)

    handler = _Capture()
    logger = logging.getLogger("vllm")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


# ---------------------------------------------------------------------------
# Errno classification (RFC #54363 item 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,is_load,expected",
    [
        (errno.ENOENT, True, "miss"),
        (errno.ENOENT, False, "unknown"),
        (errno.ESTALE, True, "transient"),
        (errno.EIO, True, "transient"),
        (errno.ENOSPC, False, "permanent"),
        (errno.EROFS, False, "permanent"),
        (errno.EACCES, True, "permanent"),
    ],
)
def test_classify_errno(code, is_load, expected):
    """A block that is simply gone is a miss, not a tier failure, and a
    condition an operator must fix is not the same as one that may clear."""
    exc = OSError(code, os.strerror(code))
    assert classify(exc, is_load=is_load).value == expected


def test_classifier_imports_without_optional_errno(monkeypatch):
    """A platform without EREMOTEIO must still be able to import the FS tier."""
    monkeypatch.delattr(errno, "EREMOTEIO", raising=False)
    namespace = runpy.run_path(errors_mod.__file__)
    result = namespace["classify"](OSError(errno.EIO, "injected"), is_load=True)
    assert result.value == "transient"


def test_unrecognised_errno_keeps_its_label():
    """Unknown errno values retain their labels without becoming transient."""
    exc = OSError(4095, "made up")
    assert classify(exc, is_load=True) is IOErrorClass.UNKNOWN
    assert errno_label(exc) == "errno_4095"

    no_errno = OSError("short read: expected 64, read 32")
    assert classify(no_errno, is_load=True) is IOErrorClass.UNKNOWN
    assert errno_label(no_errno) == "unknown"


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_load_miss_is_not_reported_as_an_error(
    fs_tier, monkeypatch, use_c_ext, vllm_logs
):
    """A block removed between lookup and load is ordinary on shared storage
    with an external evictor, so it must not look like an I/O failure."""
    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    keys = [key(i) for i in range(4)]
    tier.submit_store(make_job(1, keys, list(range(4))))
    drain(tier)
    lookup_and_wait(tier, keys)

    os.remove(tier.file_mapper.get_file_name(keys[1]))

    tier.submit_load(make_job(2, keys, list(range(4)), is_promotion=True))
    results = drain(tier)

    assert [r.success for r in results] == [False]
    assert tier.io_error_counts()[("miss", "ENOENT")] == 1
    assert not [r for r in vllm_logs if r.levelno >= logging.ERROR]


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root ignores the mode bits this test relies on"
)
def test_permanent_error_is_escalated_once(fs_tier, vllm_logs):
    """A volume that has stopped accepting writes must say so once, instead of
    repeating an indistinguishable error line for every job."""
    tier, tensor = fs_tier
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    # Block files live in a sibling of base_path, not inside it.
    base = f"{tier.file_mapper.base_path}_r{tier.file_mapper.rank}"
    os.makedirs(base, exist_ok=True)
    mode = os.stat(base).st_mode
    os.chmod(base, stat.S_IRUSR | stat.S_IXUSR)
    try:
        for job_id in range(3):
            tier.submit_store(make_job(10 + job_id, [key(100 + job_id)], [0]))
            drain(tier)
    finally:
        os.chmod(base, mode)

    warnings = [r for r in vllm_logs if r.levelno == logging.WARNING]
    errors = [r for r in vllm_logs if r.levelno >= logging.ERROR]
    assert len(warnings) == 1, "the operator-actionable line must appear once"
    assert len(errors) == 1, "repeats of a known permanent fault must be quiet"
    assert sum(tier.io_error_counts().values()) == 3


def test_a_second_permanent_fault_is_still_escalated(fs_tier, vllm_logs, monkeypatch):
    """Suppressing one permanent errno must not hide a different errno."""
    tier, _ = fs_tier
    codes = [errno.EACCES, errno.EACCES, errno.ENOSPC, errno.ENOSPC]
    failures = iter(codes)

    def stub(*args, **kwargs):
        code = next(failures)
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(mgr_mod, "batch_store_block", stub)
    for job_id, _ in enumerate(codes):
        tier.submit_store(make_job(job_id, [key(job_id)], [0]))
        drain(tier)

    warned = [r.getMessage() for r in vllm_logs if r.levelno == logging.WARNING]
    assert len(warned) == 2, "one line per distinct permanent errno"
    assert "EACCES" in warned[0] and "ENOSPC" in warned[1]
    assert len([r for r in vllm_logs if r.levelno == logging.ERROR]) == 2
    assert tier.io_error_counts() == {
        ("permanent", "EACCES"): 2,
        ("permanent", "ENOSPC"): 2,
    }


def test_error_counts_survive_concurrent_failures(fs_tier):
    """Every pool thread updates the same counters, so a plain
    read-modify-write would lose failures under load.

    ``counts[k] = counts.get(k, 0) + 1`` is several bytecodes, and the
    interpreter may switch threads between them. At the default switch
    interval it usually does not, which would leave this test passing over an
    unsynchronised counter, so the interval is shortened to make the
    interleaving actually happen.
    """
    tier, _ = fs_tier
    threads_count, per_thread = 8, 2000

    def hammer() -> None:
        for _ in range(per_thread):
            tier._count_io_error(OSError(errno.EIO, "injected"), is_load=True)

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=hammer) for _ in range(threads_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(previous)

    assert tier.io_error_counts()[("transient", "EIO")] == threads_count * per_thread


def test_unclassified_error_still_reported_as_an_error():
    """The pool must not go quiet for a failure no one classified."""
    assert log_level_for(RuntimeError("not an OSError")) == logging.ERROR


def test_repeat_of_a_known_permanent_fault_is_quiet():
    """The escalation message promises later occurrences drop to debug."""
    exc = OSError(errno.EROFS, "read-only")
    annotate(exc, IOErrorClass.PERMANENT, first=True)
    assert log_level_for(exc) == logging.ERROR
    annotate(exc, IOErrorClass.PERMANENT, first=False)
    assert log_level_for(exc) == logging.DEBUG


# ---------------------------------------------------------------------------
# Transient error retry (RFC #54363 item 4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_retry_budget_is_rejected(tmp_path, budget):
    """Reject invalid budgets before creating files or starting workers."""
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    root = tmp_path / "unused"
    with pytest.raises(ValueError, match="io_retry_budget_seconds"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(root),
            io_retry_budget_seconds=budget,
        )
    assert not root.exists()


@pytest.fixture
def fs_tier_retry(tmp_path, monkeypatch):
    """Enable retries with a deterministic clock, independent of CI load."""
    # Replace only the manager's reference, not the shared time module used
    # by thread_pool and drain(). Tests of the time gate override this clock.
    monkeypatch.setattr(mgr_mod, "time", SimpleNamespace(monotonic=lambda: 0.0))
    tensor = _page_aligned_zero_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=2,
        n_write_threads=2,
        io_retry_budget_seconds=0.05,
    )
    try:
        yield tier, tensor
    finally:
        tier.shutdown()


def _failing_io(code, times, *, is_load=True):
    """Fail before I/O a fixed number of times, then run the real operation."""
    state = {"n": 0}
    real = io_mod.batch_load_block if is_load else io_mod.batch_store_block

    def stub(paths, view, offsets, block_size, use_o_direct=True):
        if state["n"] < times:
            state["n"] += 1
            exc = OSError(code, os.strerror(code))
            if is_load:
                exc.num_succeeded = 0  # type: ignore[attr-defined]
            raise exc
        return real(paths, view, offsets, block_size, use_o_direct)

    return stub


def _store_then_lookup(tier, tensor, count=4):
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    keys = [key(i) for i in range(count)]
    tier.submit_store(make_job(1, keys, list(range(count))))
    drain(tier)
    lookup_and_wait(tier, keys)
    return keys


def test_transient_failure_is_retried_and_recovers(fs_tier_retry, monkeypatch):
    """A fault that clears must not cost the request a recompute."""
    tier, tensor = fs_tier_retry
    keys = _store_then_lookup(tier, tensor)
    monkeypatch.setattr(mgr_mod, "batch_load_block", _failing_io(errno.EIO, times=1))

    tier.submit_load(make_job(2, keys, list(range(4)), is_promotion=True))
    results = drain(tier)

    assert [r.success for r in results] == [True]
    assert tier.io_retry_counts() == {("load", "EIO"): 1}
    assert tier.io_error_counts() == {("transient", "EIO"): 1}


@pytest.mark.parametrize("elapsed", [0.05, 0.08])
@pytest.mark.parametrize("is_load", [True, False])
def test_slow_failure_is_not_retried(fs_tier_retry, monkeypatch, elapsed, is_load):
    """At or beyond the failed-attempt budget, no retry is submitted."""
    tier, tensor = fs_tier_retry
    keys = _store_then_lookup(tier, tensor)
    ticks = iter([0.0, elapsed])
    monkeypatch.setattr(
        mgr_mod, "time", SimpleNamespace(monotonic=lambda: next(ticks, elapsed))
    )
    monkeypatch.setattr(
        mgr_mod,
        "batch_load_block" if is_load else "batch_store_block",
        _failing_io(errno.EIO, times=1, is_load=is_load),
    )

    submit = tier.submit_load if is_load else tier.submit_store
    submit(make_job(2, keys, list(range(4)), is_promotion=is_load))
    results = drain(tier)

    assert [r.success for r in results] == [False]
    assert tier.io_retry_counts() == {}


@pytest.mark.parametrize("is_load", [True, False])
def test_retry_budget_accumulates_failed_attempts(fs_tier_retry, monkeypatch, is_load):
    """Two individually cheap failures can exhaust the cumulative budget."""
    tier, tensor = fs_tier_retry
    keys = _store_then_lookup(tier, tensor)
    ticks = iter([0.0, 0.03, 0.03, 0.06])
    monkeypatch.setattr(
        mgr_mod, "time", SimpleNamespace(monotonic=lambda: next(ticks, 0.06))
    )
    monkeypatch.setattr(
        mgr_mod,
        "batch_load_block" if is_load else "batch_store_block",
        _failing_io(errno.EIO, times=2, is_load=is_load),
    )

    submit = tier.submit_load if is_load else tier.submit_store
    submit(make_job(2, keys, list(range(4)), is_promotion=is_load))

    assert [r.success for r in drain(tier)] == [False]
    direction = "load" if is_load else "store"
    assert tier.io_retry_counts() == {(direction, "EIO"): 1}
    assert tier.io_error_counts() == {("transient", "EIO"): 2}


@pytest.mark.parametrize("code", [errno.ENOENT, errno.EROFS, 4095])
@pytest.mark.parametrize("is_load", [True, False])
def test_nontransient_failure_is_not_retried(fs_tier_retry, monkeypatch, code, is_load):
    """Miss, permanent and unknown errors each terminate the task."""
    tier, tensor = fs_tier_retry
    keys = _store_then_lookup(tier, tensor)
    monkeypatch.setattr(
        mgr_mod,
        "batch_load_block" if is_load else "batch_store_block",
        _failing_io(code, times=1, is_load=is_load),
    )

    submit = tier.submit_load if is_load else tier.submit_store
    submit(make_job(2, keys, list(range(4)), is_promotion=is_load))
    assert [r.success for r in drain(tier)] == [False]
    assert tier.io_retry_counts() == {}


@pytest.mark.parametrize("is_load", [True, False])
def test_retry_is_off_by_default(fs_tier, monkeypatch, is_load):
    """Disabled retries do not read a retry clock or repeat failed I/O."""
    tier, tensor = fs_tier
    clock = MagicMock(side_effect=AssertionError("retry clock is disabled"))
    monkeypatch.setattr(mgr_mod, "time", SimpleNamespace(monotonic=clock))
    keys = _store_then_lookup(tier, tensor)
    operation = "batch_load_block" if is_load else "batch_store_block"
    failing_io = MagicMock(side_effect=OSError(errno.EIO, "injected"))
    monkeypatch.setattr(mgr_mod, operation, failing_io)

    submit = tier.submit_load if is_load else tier.submit_store
    submit(make_job(2, keys, list(range(4)), is_promotion=is_load))
    results = drain(tier)

    assert [r.success for r in results] == [False]
    assert tier.io_retry_counts() == {}
    assert tier.io_error_counts() == {("transient", "EIO"): 1}
    failing_io.assert_called_once()
    clock.assert_not_called()


def _partial_load(fail_points, code=errno.EIO):
    """Load stub that really reads a prefix, then fails at a chosen block.

    ``fail_points`` is one prefix length per failure; afterwards the real
    implementation runs. Reading the prefix for real is what makes the resume
    observable: the blocks it copied must survive into the final result.
    """
    state = {"n": 0}
    real = io_mod.batch_load_block

    def stub(paths, view, offsets, block_size, use_o_direct=True):
        if state["n"] < len(fail_points):
            k = min(fail_points[state["n"]], len(paths))
            state["n"] += 1
            if k:
                real(paths[:k], view, offsets[:k], block_size, use_o_direct)
            exc = OSError(code, os.strerror(code))
            exc.num_succeeded = k  # type: ignore[attr-defined]
            raise exc
        return real(paths, view, offsets, block_size, use_o_direct)

    return stub


def _distinct_blocks(tensor, count):
    """Give each source block content no other block shares."""
    tensor[:] = _page_aligned_rand_tensor(_NUM_BLOCKS, _BLOCK_ELEMENTS)
    for i in range(count):
        tensor[i] += float(i + 1) * 1000.0
    return tensor[:count].clone()


@pytest.mark.parametrize("use_c_ext", [True, False])
@pytest.mark.parametrize("fail_points", [[2], [1, 1]])
def test_retry_resumes_without_losing_or_repeating_data(
    fs_tier_retry, monkeypatch, use_c_ext, fail_points
):
    """Every block must arrive in its own destination slot, whether it was
    copied before a failure or after the resume."""
    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)

    tier, tensor = fs_tier_retry
    expected = _distinct_blocks(tensor, 4)
    keys = [key(i) for i in range(4)]
    tier.submit_store(make_job(1, keys, list(range(4))))
    drain(tier)
    lookup_and_wait(tier, keys)

    # Load into slots the store never touched, so leftover source data cannot
    # stand in for a block the resume failed to copy.
    dest = [4, 5, 6, 7]
    tensor[4:8] = -7.0
    monkeypatch.setattr(mgr_mod, "batch_load_block", _partial_load(fail_points))

    tier.submit_load(make_job(2, keys, dest, is_promotion=True))
    results = drain(tier)

    assert [r.success for r in results] == [True]
    assert torch.equal(tensor[4:8], expected)


def test_partial_progress_accumulates_across_retries(fs_tier_retry, monkeypatch):
    """When the retries run out, the keys reported as loaded must count every
    block copied, not just the ones from the last attempt."""
    tier, tensor = fs_tier_retry
    expected = _distinct_blocks(tensor, 4)
    keys = [key(i) for i in range(4)]
    tier.submit_store(make_job(1, keys, list(range(4))))
    drain(tier)
    lookup_and_wait(tier, keys)

    tensor[4:8] = -7.0
    # One block per attempt, and the attempt cap stops it before the last.
    monkeypatch.setattr(mgr_mod, "batch_load_block", _partial_load([1, 1, 1]))

    tier.submit_load(make_job(2, keys, [4, 5, 6, 7], is_promotion=True))
    results = drain(tier)

    assert [r.success for r in results] == [False]
    assert results[0].successful_keys == tuple(keys[:3])
    assert torch.equal(tensor[4:7], expected[:3])


@pytest.mark.parametrize("prefix", [0, 1])
def test_non_oserror_preserves_only_confirmed_progress(
    fs_tier_retry, monkeypatch, prefix, vllm_logs
):
    """Unexpected errors must not mark unread slots as successfully loaded."""
    tier, tensor = fs_tier_retry
    expected = _distinct_blocks(tensor, 2)
    keys = [key(0), key(1)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))
    lookup_and_wait(tier, keys)
    tensor[4:6] = -7.0
    partial = _partial_load([prefix])
    calls = 0

    def fail(paths, view, offsets, block_size, use_o_direct=True):
        nonlocal calls
        calls += 1
        if prefix and calls == 1:
            partial(paths, view, offsets, block_size, use_o_direct)
        raise RuntimeError("unexpected load failure")

    monkeypatch.setattr(mgr_mod, "batch_load_block", fail)
    tier.submit_load(make_job(2, keys, [4, 5], is_promotion=True))
    results = drain(tier)

    assert len(results) == 1 and not results[0].success
    assert results[0].successful_keys == (tuple(keys[:prefix]) or None)
    assert calls == prefix + 1
    assert tier.io_retry_counts() == ({("load", "EIO"): 1} if prefix else {})
    assert any(
        r.levelno == logging.ERROR and "unexpected load failure" in r.getMessage()
        for r in vllm_logs
    )

    # A subsequent transfer still completes on the same pool.
    monkeypatch.setattr(mgr_mod, "batch_load_block", io_mod.batch_load_block)
    tier.submit_load(make_job(3, keys, [4, 5], is_promotion=True))
    assert [r.success for r in drain(tier)] == [True]
    assert torch.equal(tensor[4:6], expected)


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_store_retry_completes_the_batch(fs_tier_retry, monkeypatch, use_c_ext):
    """A store restarts the whole batch, so the blocks the failed attempt had
    already written must still be the right blocks afterwards."""
    if use_c_ext and not io_mod._HAS_FSIO_C:
        pytest.skip("fs_io_C extension not built")
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", use_c_ext)
    tier, tensor = fs_tier_retry
    expected = _distinct_blocks(tensor, 4)
    keys = [key(i) for i in range(4)]

    state = {"n": 0}
    real_store = io_mod.batch_store_block

    def stub(paths, view, offsets, block_size, use_o_direct=True):
        if state["n"] == 0:
            state["n"] += 1
            real_store(paths[:2], view, offsets[:2], block_size, use_o_direct)
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real_store(paths, view, offsets, block_size, use_o_direct)

    monkeypatch.setattr(mgr_mod, "batch_store_block", stub)
    tier.submit_store(make_job(1, keys, list(range(4))))
    results = drain(tier)

    assert [r.success for r in results] == [True]
    assert tier.io_retry_counts() == {("store", "EIO"): 1}

    monkeypatch.setattr(mgr_mod, "batch_store_block", real_store)
    lookup_and_wait(tier, keys)
    tensor[4:8] = -7.0
    tier.submit_load(make_job(2, keys, [4, 5, 6, 7], is_promotion=True))
    assert [r.success for r in drain(tier)] == [True]
    assert torch.equal(tensor[4:8], expected)


def test_short_read_is_not_retried_on_the_python_path(fs_tier_retry, monkeypatch):
    """A short read has already deleted the file it was reading. Repeating it
    would find nothing and record detected corruption as an ordinary miss."""
    monkeypatch.setattr(io_mod, "_HAS_FSIO_C", False)
    tier, tensor = fs_tier_retry
    _distinct_blocks(tensor, 2)
    keys = [key(i) for i in range(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    drain(tier)
    lookup_and_wait(tier, keys)

    path = tier.file_mapper.get_file_name(keys[1])
    with open(path, "r+b") as handle:
        handle.truncate(os.path.getsize(path) // 2)

    tier.submit_load(make_job(2, keys, [4, 5], is_promotion=True))
    results = drain(tier)

    assert [r.success for r in results] == [False]
    assert tier.io_retry_counts() == {}
    assert tier.io_error_counts() == {("unknown", "unknown"): 1}


@pytest.mark.parametrize("is_load", [True, False])
def test_a_fault_that_never_clears_gives_up(fs_tier_retry, monkeypatch, is_load):
    """The attempt cap stops repeated fast failures even with budget left."""
    tier, tensor = fs_tier_retry
    keys = _store_then_lookup(tier, tensor)
    monkeypatch.setattr(
        mgr_mod,
        "batch_load_block" if is_load else "batch_store_block",
        _failing_io(errno.EIO, times=99, is_load=is_load),
    )

    submit = tier.submit_load if is_load else tier.submit_store
    submit(make_job(2, keys, list(range(4)), is_promotion=is_load))
    results = drain(tier)

    assert [r.success for r in results] == [False]
    direction = "load" if is_load else "store"
    assert tier.io_retry_counts() == {(direction, "EIO"): 2}
    assert tier.io_error_counts() == {("transient", "EIO"): 3}
