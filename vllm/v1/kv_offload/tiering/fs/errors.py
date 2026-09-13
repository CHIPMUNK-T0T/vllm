# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Filesystem I/O classification, retry policy and terminal-error logging.

Only known transient errnos are retried. Unknown errors, including detected
corruption, which both I/O paths raise without an errno, are counted but not
retried.
"""

import errno as _errno
import logging
import math
from dataclasses import dataclass
from enum import Enum

# Retry candidates, not a guarantee that the fault will clear. Resource
# exhaustion (ENOMEM, EMFILE, ENFILE) is left out: an immediate repeat from the
# same worker has no reason to succeed.
_TRANSIENT = frozenset(
    {
        _errno.EAGAIN,
        _errno.EWOULDBLOCK,
        _errno.EINTR,
        _errno.EIO,
        _errno.ESTALE,
        _errno.ETIMEDOUT,
    }
)

# Retrying the unchanged operation is not expected to help.
_PERMANENT = frozenset(
    {
        _errno.ENOSPC,
        _errno.EDQUOT,
        _errno.EROFS,
        _errno.EACCES,
        _errno.EPERM,
        _errno.EISDIR,
        _errno.ENOTDIR,
        _errno.ENAMETOOLONG,
        _errno.EFBIG,
        # Includes O_DIRECT alignment errors; retrying does not fix alignment.
        _errno.EINVAL,
        _errno.EOVERFLOW,
    }
)


class IOErrorClass(Enum):
    """How a filesystem tier I/O failure should be accounted for."""

    MISS = "miss"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RetryPolicy:
    """When a failed task may be repeated. Disabled by default.

    ``budget_seconds`` bounds the time already spent in failed attempts, not
    the next attempt, which may still block. It is a placeholder for a job
    deadline, which is what should bound retries once one exists.
    """

    # The benefit over recomputation is not yet measured.
    max_retries: int = 2
    budget_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget_seconds) or self.budget_seconds < 0:
            raise ValueError("budget_seconds must be finite and nonnegative")

    @property
    def enabled(self) -> bool:
        return self.budget_seconds > 0.0

    def allows(self, cls: IOErrorClass, attempts: int, spent: float) -> bool:
        """Whether a failure may be repeated.

        Args:
            cls: The class of the failure.
            attempts: Retries this task has already made.
            spent: Cumulative time of this task's failed attempts.
        """
        return (
            cls is IOErrorClass.TRANSIENT
            and attempts < self.max_retries
            and spent < self.budget_seconds
        )


# Preserve the original OSError, including its errno and num_succeeded.
_CLASS_ATTR = "vllm_io_class"
_FIRST_ATTR = "vllm_io_first"


def classify(exc: OSError, *, is_load: bool) -> IOErrorClass:
    """Classify a filesystem tier I/O failure by its errno.

    Args:
        exc: The error raised by either I/O implementation.
        is_load: Whether this is a load. ENOENT is a miss only on load;
            a missing store directory has no retry policy yet.

    Returns:
        The error class. Missing or unrecognised errnos are UNKNOWN and do not
        authorize retries, including Python short reads with no errno.
    """
    code = exc.errno
    if code is None:
        return IOErrorClass.UNKNOWN
    if is_load and code == _errno.ENOENT:
        return IOErrorClass.MISS
    if code in _PERMANENT:
        return IOErrorClass.PERMANENT
    if code in _TRANSIENT:
        return IOErrorClass.TRANSIENT
    return IOErrorClass.UNKNOWN


def errno_label(exc: OSError) -> str:
    """Return the errno name, preserving unrecognised numeric codes."""
    code = exc.errno
    if code is None:
        return "unknown"
    return _errno.errorcode.get(code, f"errno_{code}")


def annotate(exc: OSError, cls: IOErrorClass, *, first: bool) -> None:
    """Record the classification on the error for the thread pool to read.

    Args:
        exc: The error about to be re-raised to the pool.
        cls: Its class.
        first: Whether this permanent errno has not yet been reported.
    """
    setattr(exc, _CLASS_ATTR, cls)
    setattr(exc, _FIRST_ATTR, first)


def log_level_for(exc: BaseException) -> int:
    """Quiet misses and repeated permanent errors; keep other errors visible."""
    cls = getattr(exc, _CLASS_ATTR, None)
    if cls is None:
        return logging.ERROR
    if cls is IOErrorClass.MISS or (
        cls is IOErrorClass.PERMANENT and not getattr(exc, _FIRST_ATTR, True)
    ):
        return logging.DEBUG
    return logging.ERROR


def class_name(exc: BaseException) -> str:
    """Return the class of *exc* as a log/metric label."""
    cls = getattr(exc, _CLASS_ATTR, None)
    return cls.value if isinstance(cls, IOErrorClass) else "unclassified"
