# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Object store secondary tier implementation."""

import contextlib
import ctypes
import json
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from vllm.distributed.nixl_utils import NixlWrapper as nixl_agent
from vllm.distributed.nixl_utils import nixl_agent_config
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.manifest import reconcile_manifest
from vllm.v1.kv_offload.tiering.async_lookup import AsyncLookupManager
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobResult,
    RequestOffloadingContext,
    ScheduleEndContext,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.obj.config import ObjStoreConfig

if TYPE_CHECKING:
    from nixl._api import nixl_prepped_dlist_handle, nixl_xfer_handle

    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)

NIXL_WRITE = "WRITE"
NIXL_READ = "READ"
NIXL_PROC = "PROC"
NIXL_DONE = "DONE"

# Device ID for CPU DRAM descriptors. DRAM is not a multi-device resource so
# the device ID is always 0.
NIXL_DEV_ID: int = 0

# Fields for NIXL OBJ descriptors: (addr, len, dev_id, obj_key).
# For existence probes addr and len are placeholders — no data is read.
# dev_id=0 is reserved for probes; transfers start from 1.
_PROBE_ADDR: int = 0
_PROBE_LEN: int = 1
_PROBE_DEV_ID: int = 0

# The manifest is stored as one fixed-size object so it can be read back
# without a separate size query: JSON tolerates the trailing padding.
_MANIFEST_BYTES: int = 1 << 16
_MANIFEST_TIMEOUT_S: float = 30.0


class TransferEntry(NamedTuple):
    xfer_handle: "nixl_xfer_handle"
    files_desc: object
    obj_handle: "nixl_prepped_dlist_handle"


class ObjAsyncLookupManager(AsyncLookupManager):
    """Async lookup manager for ObjectStoreSecondaryTierManager.

    Batches existence probes into a single query_memory() call so the
    background thread issues one round-trip per step instead of one per key.
    """

    def __init__(
        self,
        tier: "ObjectStoreSecondaryTierManager",
        tier_type: str,
    ) -> None:
        super().__init__(tier_type=tier_type)
        self._tier = tier

    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        descriptors = [
            (
                _PROBE_ADDR,
                _PROBE_LEN,
                _PROBE_DEV_ID,
                self._tier._file_mapper.get_file_name(k),
            )
            for k in keys
        ]
        results = self._tier._agent.query_memory(descriptors, "OBJ", "OBJ")
        return (r is not None for r in results)


class ObjectStoreSecondaryTierManager(SecondaryTierManager):
    """Secondary tier that offloads KV cache blocks to an S3-compatible store.

    Handles CPU DRAM <-> S3 transfers only. GPU <-> CPU is managed by the
    primary tier. Object keys are formed as ``{prefix}/{hash_shard}/{hash}.bin``.
    """

    medium: ClassVar[Medium] = Medium.STORAGE

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        store_config: dict,
        prefix: str = "",
        io_threads: int = 4,
        enable_kv_events: bool = False,
        locality: str | None = None,
    ):
        """
        Args:
            offloading_spec: Offloading configuration.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            store_config: Object store connection parameters (see ObjStoreConfig).
            prefix: Key prefix prepended to all object keys.
            io_threads: Number of NIXL I/O threads.
            enable_kv_events: Emit BlockStored KV events for blocks
                successfully stored to this tier. Effective only when KV
                cache events are enabled globally (kv_events_config).
            locality: Whether this tier's storage is LOCAL or REMOTE relative
                to the publishing vLLM instance.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        self.locality = Locality(locality) if locality is not None else None

        self.events: list[OffloadingEvent] | None = None
        if enable_kv_events:
            if offloading_spec.kv_events_config.enable_kv_cache_events:
                self.events = []
            else:
                logger.warning(
                    "enable_kv_events is set on secondary tier '%s' but KV "
                    "cache events are disabled globally; the tier will not "
                    "emit events.",
                    tier_type,
                )
        # Keys of in-flight store jobs, tracked only when events are enabled.
        self._store_job_keys: dict[JobId, list[OffloadKey]] = {}
        # Keys of in-flight load (promotion) jobs, so a failed download can
        # mark its own cached lookup verdicts False (see get_finished_jobs).
        self._load_job_keys: dict[JobId, list[OffloadKey]] = {}

        agent_config = nixl_agent_config(backends=[])
        self._agent = nixl_agent("ObjAgent", agent_config)
        obj_config = ObjStoreConfig(**store_config)
        params = {**obj_config.to_nixl_params(), "num_threads": str(io_threads)}
        self._agent.create_backend("OBJ", params)
        self._transfers: dict[int, TransferEntry] = {}
        # Buffered results awaiting the next get_finished_jobs() call:
        # submission-time failures + poll-time completions accumulated
        # during drain_jobs().
        self._pending_results: list[JobResult] = []
        self._primary_reg = None
        self._block_size_bytes: int = 0
        root_dir = f"{prefix}/" if prefix else ""
        # Opt in; FileMapper enables it only for a parallelism-invariant block.
        self._file_mapper = FileMapper.from_offloading_spec(
            root_dir, offloading_spec, parallel_agnostic=True
        )
        self._next_obj_dev_id: int = 1  # dev_id=0 is reserved for _exists() probes
        # Reconciled on first use, not here: the prefix-cache hash seed the
        # namespace must agree on is only settled by init_none_hash, which
        # runs after the tiers are constructed. None means "not yet".
        self._namespace_usable: bool | None = None

        self._probe_connectivity()

        base_addr = ctypes.addressof(ctypes.c_char.from_buffer(primary_kv_view))
        assert primary_kv_view.strides is not None
        stride = primary_kv_view.strides[0]
        self._primary_reg = self._agent.register_memory(
            [(base_addr, primary_kv_view.nbytes, NIXL_DEV_ID, "")], "DRAM"
        )
        self._block_size_bytes = stride
        all_blocks = [
            (base_addr + i * stride, stride, NIXL_DEV_ID)
            for i in range(len(primary_kv_view))
        ]
        # NIXL_INIT_AGENT marks this as the local side; make_prepped_xfer requires
        # local_xfer_side tagged with NIXL_INIT_AGENT and remote_xfer_side tagged
        # with the peer agent name ("ObjAgent").
        self._dram_prepped_handle: nixl_prepped_dlist_handle = (
            self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", all_blocks, "DRAM")
        )

        self._lookup_manager = ObjAsyncLookupManager(
            tier=self, tier_type=self.tier_type
        )

    def _probe_connectivity(self) -> None:
        """Verify object store connectivity at startup via a NIXL lookup probe.

        Performs a single exists() check against a synthetic key that will
        never exist. A True/False result confirms the bucket is reachable;
        an exception indicates misconfigured obj store params and raises RuntimeError.
        """
        probe_key = "__nixl_probe__/connectivity_test"
        try:
            self._exists(probe_key)
            logger.info("Object store tier connectivity probe succeeded")
        except Exception as e:
            raise RuntimeError(
                f"Object store tier connectivity probe failed — check bucket, "
                f"endpoint_override, and scheme. If using explicit credentials "
                f"verify access_key and secret_key; otherwise ensure the AWS "
                f"SDK default credential chain is configured (IAM role, env "
                f"vars, credential file). Error: {e}"
            ) from e

    @property
    def _usable(self) -> bool:
        """Whether this run may read and write the namespace.

        False leaves the tier inert: it serves misses and stores nothing, so
        an incompatible namespace costs a cold prefill, not correctness. The
        reconciliation is one round trip, on the first lookup only.
        """
        if self._namespace_usable is None:
            self._namespace_usable = self._sync_manifest()
        return self._namespace_usable

    def _sync_manifest(self) -> bool:
        usable, to_write = reconcile_manifest(
            self._file_mapper,
            self.tier_type,
            self._file_mapper.base_path,
            self._read_manifest(),
        )
        if to_write is not None:
            self._write_manifest(to_write)
        return usable

    def _read_manifest(self) -> dict | None:
        """The manifest stored in this namespace, or None if there is none."""
        manifest_key = self._file_mapper.get_config_file_path()
        if not self._exists(manifest_key):
            return None
        buf = (ctypes.c_char * _MANIFEST_BYTES)()
        if not self._manifest_xfer(NIXL_READ, buf, manifest_key):
            logger.warning(
                "Could not read the KV offload manifest at '%s'; treating the "
                "namespace as unrecorded.",
                manifest_key,
            )
            return None
        try:
            return json.loads(bytes(buf).rstrip(b"\x00 \t\r\n").decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            logger.warning(
                "Rewriting unreadable KV offload manifest at '%s': %s",
                manifest_key,
                exc,
            )
            return None

    def _write_manifest(self, manifest: dict) -> None:
        """Record what wrote this namespace, so the next run can check it."""
        manifest_key = self._file_mapper.get_config_file_path()
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        if len(payload) > _MANIFEST_BYTES:
            logger.warning(
                "KV offload manifest for '%s' needs %d bytes but the object is "
                "%d; leaving the namespace unrecorded.",
                manifest_key,
                len(payload),
                _MANIFEST_BYTES,
            )
            return
        buf = (ctypes.c_char * _MANIFEST_BYTES)()
        buf.raw = payload.ljust(_MANIFEST_BYTES)
        if not self._manifest_xfer(NIXL_WRITE, buf, manifest_key):
            # Losing the manifest costs the next run its check, not this run
            # its cache, so a read-only bucket must not fail startup.
            logger.warning(
                "Could not write the KV offload manifest at '%s'; the next run "
                "will not be able to check this namespace.",
                manifest_key,
            )

    def _manifest_xfer(self, op: str, buf, manifest_key: str) -> bool:
        """Move the fixed-size manifest buffer to or from the object store.

        Synchronous, unlike the block path: the verdict it feeds gates the
        first lookup, and it happens once per process.

        Returns:
            True when the transfer completed.
        """
        addr = ctypes.addressof(buf)
        dev_id = self._next_obj_dev_id
        self._next_obj_dev_id += 1
        dram_desc = obj_desc = dram_handle = obj_handle = xfer = None
        try:
            dram_desc = self._agent.register_memory(
                [(addr, _MANIFEST_BYTES, NIXL_DEV_ID, "")], "DRAM"
            )
            obj_desc = self._agent.register_memory(
                [(0, _MANIFEST_BYTES, dev_id, manifest_key)], "OBJ"
            )
            if dram_desc is None or obj_desc is None:
                return False
            dram_handle = self._agent.prep_xfer_dlist(
                "NIXL_INIT_AGENT", [(addr, _MANIFEST_BYTES, NIXL_DEV_ID)], "DRAM"
            )
            obj_handle = self._agent.prep_xfer_dlist("ObjAgent", obj_desc.trim())
            if not dram_handle or not obj_handle:
                return False
            xfer = self._agent.make_prepped_xfer(op, dram_handle, [0], obj_handle, [0])
            if not xfer:
                return False
            state = self._agent.transfer(xfer)
            deadline = time.monotonic() + _MANIFEST_TIMEOUT_S
            while state == NIXL_PROC:
                if time.monotonic() > deadline:
                    logger.warning("Timed out on manifest %s at '%s'", op, manifest_key)
                    return False
                time.sleep(0.01)
                state = self._agent.check_xfer_state(xfer)
            return state == NIXL_DONE
        except Exception:
            logger.warning(
                "Manifest %s failed at '%s'", op, manifest_key, exc_info=True
            )
            return False
        finally:
            for handle, release in (
                (xfer, self._agent.release_xfer_handle),
                (dram_handle, self._agent.release_dlist_handle),
                (obj_handle, self._agent.release_dlist_handle),
            ):
                if handle:
                    with contextlib.suppress(Exception):
                        release(handle)
            for desc in (dram_desc, obj_desc):
                if desc is not None:
                    with contextlib.suppress(Exception):
                        self._agent.deregister_memory(desc)

    def _exists(self, obj_key: str) -> bool:
        results = self._agent.query_memory(
            [(_PROBE_ADDR, _PROBE_LEN, _PROBE_DEV_ID, obj_key)], "OBJ", "OBJ"
        )
        return results[0] is not None

    def _submit_transfer(
        self,
        job_id: int,
        block_ids: Iterable[int],
        obj_keys: Iterable[str],
        op: str,
    ) -> None:
        """Submit an async transfer. op is 'WRITE' (store) or 'READ' (load)."""
        block_ids_list = [int(bid) for bid in block_ids]
        # The OBJ backend maps devId -> obj_key. All descriptors must have
        # unique devIds or later registrations overwrite earlier ones.
        nixl_files = [
            (0, self._block_size_bytes, dev_id, key)
            for dev_id, key in enumerate(obj_keys, self._next_obj_dev_id)
        ]
        self._next_obj_dev_id += len(nixl_files)

        files_desc = self._agent.register_memory(nixl_files, "OBJ")
        if files_desc is None:
            logger.warning("register_memory (OBJ) failed for job %d", job_id)
            self._pending_results.append(JobResult(job_id=job_id, success=False))
            return

        obj_handle = self._agent.prep_xfer_dlist("ObjAgent", files_desc.trim())
        if not obj_handle:
            logger.warning("prep_xfer_dlist (OBJ) failed for job %d", job_id)
            self._agent.deregister_memory(files_desc)
            self._pending_results.append(JobResult(job_id=job_id, success=False))
            return

        xfer_handle = self._agent.make_prepped_xfer(
            op,
            self._dram_prepped_handle,
            block_ids_list,
            obj_handle,
            list(range(len(nixl_files))),
        )
        if not xfer_handle:
            logger.warning("make_prepped_xfer failed for job %d", job_id)
            self._agent.release_dlist_handle(obj_handle)
            self._agent.deregister_memory(files_desc)
            self._pending_results.append(JobResult(job_id=job_id, success=False))
            return

        state = self._agent.transfer(xfer_handle)
        if state == "ERR":
            logger.warning("agent.transfer failed for job %d", job_id)
            self._agent.release_dlist_handle(obj_handle)
            self._agent.deregister_memory(files_desc)
            self._agent.release_xfer_handle(xfer_handle)
            self._pending_results.append(JobResult(job_id=job_id, success=False))
            return

        self._transfers[job_id] = TransferEntry(
            xfer_handle=xfer_handle,
            files_desc=files_desc,
            obj_handle=obj_handle,
        )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if not self._usable:
            return LookupResult.MISS
        result = self._lookup_manager.lookup(key, req_context)
        if result is None:
            return LookupResult.RETRY
        return LookupResult.HIT if result else LookupResult.MISS

    def submit_store(self, job_metadata: TransferJob) -> None:
        if not self._usable:
            # Fail the job so the caller is not left waiting, and leave the
            # foreign namespace untouched.
            self._pending_results.append(
                JobResult(job_id=job_metadata.job_id, success=False)
            )
            return
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        obj_keys = (self._file_mapper.get_file_name(k) for k in job_metadata.keys)
        self._submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, obj_keys, NIXL_WRITE
        )

    def submit_load(self, job_metadata: TransferJob) -> None:
        self._load_job_keys[job_metadata.job_id] = list(job_metadata.keys)
        obj_keys = (self._file_mapper.get_file_name(k) for k in job_metadata.keys)
        self._submit_transfer(
            job_metadata.job_id, job_metadata.block_ids, obj_keys, NIXL_READ
        )

    def on_request_finished(self, req_context: ReqContext) -> None:
        self._lookup_manager.cleanup(req_context.req_id)

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._lookup_manager.flush()

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def _poll_active_transfers(self) -> None:
        """Poll all in-flight transfers once; move newly-completed (success or
        failure) into ``_pending_results`` and release their NIXL handles."""
        for job_id, entry in list(self._transfers.items()):
            try:
                state = self._agent.check_xfer_state(entry.xfer_handle)
            except Exception as exc:
                success = False
                logger.warning("check_xfer_state raised for job %d: %s", job_id, exc)
            else:
                if state == NIXL_PROC:
                    continue
                if state == NIXL_DONE:
                    success = True
                else:
                    success = False
                    logger.warning("transfer failed job=%d state=%s", job_id, state)

            transfer_time = None
            if success:
                telemetry = self._agent.get_xfer_telemetry(entry.xfer_handle)
                transfer_time = telemetry.xferDuration / 1e6

            try:
                self._agent.release_xfer_handle(entry.xfer_handle)
            except Exception as exc:
                # Keep the entry until NIXL confirms that the transfer handle
                # can be released. The transfer may still access primary-tier
                # memory, so publishing its result would allow unsafe reuse.
                logger.warning("release_xfer_handle failed for job %d: %s", job_id, exc)
                continue

            # Once the transfer handle is released, these remaining cleanup
            # failures must not suppress the job completion. They can leak
            # NIXL metadata, but cannot leave an active data transfer behind.
            try:
                self._agent.release_dlist_handle(entry.obj_handle)
            except Exception as exc:
                logger.warning(
                    "release_dlist_handle failed for job %d: %s", job_id, exc
                )
            try:
                self._agent.deregister_memory(entry.files_desc)
            except Exception as exc:
                logger.warning("deregister_memory failed for job %d: %s", job_id, exc)

            del self._transfers[job_id]
            self._pending_results.append(
                JobResult(
                    job_id=job_id,
                    success=success,
                    transfer_time=transfer_time,
                )
            )

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Poll transfers; a failed promotion marks its cached verdicts False
        here (scheduler thread)."""
        self._poll_active_transfers()
        results = self._pending_results
        self._pending_results = []
        for result in results:
            if self.events is not None:
                keys = self._store_job_keys.pop(result.job_id, None)
                if result.success and keys:
                    self.events.append(
                        OffloadingEvent(
                            keys=keys,
                            medium=self.medium,
                            removed=False,
                            locality=self.locality,
                        )
                    )
            # Mark only the keys that did not load as a miss; the request
            # recomputes them. The miss is per-request (cleared when the request
            # finishes), so other requests still HIT the blocks that loaded
            # fine. nixl reports the batch as a whole (successful_keys is None),
            # so today this marks all keys; the subtraction keeps it correct if
            # partial results are ever reported.
            load_keys = self._load_job_keys.pop(result.job_id, None)
            if load_keys is not None and not result.success:
                successful = set(result.successful_keys or ())
                failed = [k for k in load_keys if k not in successful]
                self._lookup_manager.mark_miss(failed)
        return results

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def drain_jobs(self) -> None:
        """Block until every submitted transfer has completed or failed.

        nixl exposes only ``check_xfer_state`` (poll-based), so this loops
        until ``_transfers`` is empty. Results accumulate in
        ``_pending_results`` and are surfaced by the next
        ``get_finished_jobs()`` call.
        """
        start = time.monotonic()
        warned = False
        while self._transfers:
            self._poll_active_transfers()
            if not self._transfers:
                break
            if not warned and time.monotonic() - start > 5.0:
                logger.warning(
                    "ObjectStoreSecondaryTierManager.drain_jobs: still "
                    "draining after 5s (%d transfers in flight); a stuck "
                    "transfer will block the engine.",
                    len(self._transfers),
                )
                warned = True
            time.sleep(0.001)

    def shutdown(self) -> None:
        self._lookup_manager.shutdown()
        for job_id, entry in self._transfers.items():
            try:
                self._agent.release_xfer_handle(entry.xfer_handle)
            except Exception as exc:
                logger.warning("release_xfer_handle failed for job %d: %s", job_id, exc)
            try:
                self._agent.release_dlist_handle(entry.obj_handle)
            except Exception as exc:
                logger.warning(
                    "release_dlist_handle failed for job %d: %s", job_id, exc
                )
            try:
                self._agent.deregister_memory(entry.files_desc)
            except Exception as exc:
                logger.warning("deregister_memory failed for job %d: %s", job_id, exc)
        self._transfers.clear()
        if self._dram_prepped_handle is not None:
            try:
                self._agent.release_dlist_handle(self._dram_prepped_handle)
            except Exception as exc:
                logger.warning("failed to release DRAM prepped handle: %s", exc)
            self._dram_prepped_handle = None
        if self._primary_reg is not None:
            try:
                self._agent.deregister_memory(self._primary_reg)
            except Exception as exc:
                logger.warning("failed to deregister primary buffer: %s", exc)
            self._primary_reg = None
