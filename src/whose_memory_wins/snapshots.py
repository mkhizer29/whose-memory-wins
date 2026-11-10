"""One canonical way to hash and verify frozen retrieval snapshots.

Writers (controlled fixtures, normal-ingestion studies, memory-removal diagnostics) call
``finalize_snapshot`` as the LAST step, after every field is set. Readers call
``verify_snapshot_file``, which recomputes the hash from the file's actual content
(everything except the ``snapshot_hash`` field itself) and compares it with both the stored
hash and the hash recorded in the run manifest. Editing a retrieved sentence while leaving the
old stored hash in place is therefore detected.

Historical snapshots are never rewritten here: a mismatch is an error to report, not to repair.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import stable_hash

HASH_FIELD = "snapshot_hash"


class SnapshotIntegrityError(RuntimeError):
    """A retrieval snapshot is missing or its content does not match its hash."""
    pass


def snapshot_content_hash(snap: dict[str, Any]) -> str:
    """SHA-256 of the snapshot content excluding ``snapshot_hash``.

    The content is normalized through a JSON round trip first, so the hash computed in memory
    before writing equals the hash recomputed from the file after reading.
    """
    content = {k: v for k, v in snap.items() if k != HASH_FIELD}
    return stable_hash(json.loads(json.dumps(content, default=str)))


def finalize_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Set ``snapshot_hash`` from the final content. Call after the last field is changed."""
    snap[HASH_FIELD] = snapshot_content_hash(snap)
    return snap


def write_snapshot(path: Path, snap: dict[str, Any]) -> str:
    """Hash the final snapshot content, write it to ``path`` and return the hash."""
    finalize_snapshot(snap)
    Path(path).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return snap[HASH_FIELD]


def verify_snapshot_file(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    """Load a snapshot and verify its content. Returns the snapshot if it is intact.

    Raises ``SnapshotIntegrityError`` if the recomputed content hash differs from the stored hash
    or from ``expected_hash`` (the value recorded in the run manifest).
    """
    path = Path(path)
    if not path.exists():
        raise SnapshotIntegrityError(f"retrieval snapshot is missing: {path}")
    snap = json.loads(path.read_text(encoding="utf-8"))
    actual = snapshot_content_hash(snap)
    stored = snap.get(HASH_FIELD)
    if stored != actual:
        raise SnapshotIntegrityError(
            f"retrieval snapshot content does not match its stored hash: {path}\n"
            f"  stored {stored}\n  actual {actual}\n"
            "The file was edited after it was written, or it was written by an older harness version "
            "that changed fields after hashing. It is NOT repaired automatically; rebuild the fixture / "
            "ingestion study / diagnostic and create a new run.")
    if expected_hash is not None and actual != expected_hash:
        raise SnapshotIntegrityError(
            f"retrieval snapshot changed since the run manifest was created: {path}\n"
            f"  manifest {expected_hash}\n  actual   {actual}")
    return snap
