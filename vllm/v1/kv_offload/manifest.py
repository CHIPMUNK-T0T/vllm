# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconciling a run against the manifest stored in its namespace.

A storage namespace is named by a hash of the fields that must never alias.
Fields that a run can legitimately differ on -- and still be unable to read
the bytes -- cannot go in that hash: adding one renames the namespace, and
with no reclamation in this subsystem the data already there is orphaned
rather than reused. They are recorded in the manifest and checked here.

The policy is shared by every persistent tier; the I/O is not, so the caller
reads and writes the manifest itself.
"""

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import UNSHAREABLE_NONE_HASH_SEED
from vllm.v1.kv_offload.file_mapper import MANIFEST_VERSION, FileMapper

logger = init_logger(__name__)


def reconcile_manifest(
    file_mapper: FileMapper,
    tier_type: str,
    namespace: str,
    stored: dict | None,
) -> tuple[bool, dict | None]:
    """Decide whether this run may use a namespace, and what to record in it.

    Args:
        file_mapper: The mapper naming this run's namespace.
        tier_type: Tier identifier, for diagnostics.
        namespace: Human-readable namespace location, for diagnostics.
        stored: The manifest found in the namespace, or None when there is
            none or it could not be read.

    Returns:
        `(usable, to_write)`. `usable` is False when the namespace belongs to
        a run whose bytes this one cannot interpret; the caller is expected to
        go inert rather than raise, so an incompatible namespace costs a cold
        prefill and not correctness. `to_write` is the manifest to publish, or
        None when the stored one already records the right thing.
    """
    compat = file_mapper.resolved_compat()
    if compat.get("hash_seed") == UNSHAREABLE_NONE_HASH_SEED:
        # vLLM warns that block hashes are irreproducible, but not that a
        # persistent tier keeps writing blocks under them: every restart adds
        # a generation of objects nothing will ever look up again.
        logger.warning(
            "KV offload tier '%s' cannot reuse '%s' across restarts: the "
            "prefix-cache hash seed is random per process, so this run "
            "matches nothing already stored and nothing it stores will be "
            "found again. Set PYTHONHASHSEED to a shared value, or use a "
            "cryptographic prefix_caching_hash_algo.",
            tier_type,
            namespace,
        )

    if stored is not None:
        mismatches = file_mapper.compat_mismatches(stored)
        if mismatches:
            logger.error(
                "Disabling KV offload tier '%s': '%s' was written with %s. "
                "Those blocks cannot be reinterpreted by this run, so it will "
                "prefill cold. Give each configuration its own root_dir to "
                "cache both.",
                tier_type,
                namespace,
                ", ".join(
                    f"{name}={found!r}, not {mine!r}"
                    for name, (found, mine) in sorted(mismatches.items())
                ),
            )
            return False, None
        if stored.get("manifest_version") == MANIFEST_VERSION:
            return True, None

    # Nothing on disk records what wrote the blocks already here, so adopting
    # the namespace leaves one unchecked reopen per namespace written before
    # this manifest existed. The alternative -- refusing every such namespace
    # -- discards those caches outright, which costs far more than the single
    # window it closes.
    return True, file_mapper.get_manifest()
