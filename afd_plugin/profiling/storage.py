# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded background JSONL output for CAM profiling records."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path


class JsonRecordWriter:
    """Transfer records through a bounded, lock-based queue to one writer.

    Written counters cover flushed batches; an IO failure can leave a partial
    final batch in the file, identified by the error in the snapshot.
    """

    def __init__(
        self,
        path: Path,
        max_records: int = 4096,
        max_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if max_records <= 0 or max_bytes <= 0:
            raise ValueError("max_records and max_bytes must be positive")
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=max_records)
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._accepting = True
        self._closed = False
        self._written_records = 0
        self._dropped_records = 0
        self._written_bytes = 0
        self._pending_records = 0
        self._error: str | None = None
        self._file = path.open("x", encoding="utf-8", newline="\n")
        self._thread = threading.Thread(
            target=self._run, name="afd-cam-profile-writer", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._file.close()
            raise

    def submit(self, record: dict) -> bool:
        """Transfer ownership; the caller must not mutate the submitted record."""
        with self._lock:
            if not self._accepting:
                self._dropped_records += 1
                return False
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped_records += 1
                return False
            self._pending_records += 1
            return True

    def close(self, timeout_s: float) -> bool:
        """Stop admission and wait at most timeout_s; inspect snapshot for errors."""
        with self._lock:
            self._accepting = False
            self._stopping.set()
        self._thread.join(timeout=timeout_s)
        return not self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "written_records": self._written_records,
                "dropped_records": self._dropped_records,
                "written_bytes": self._written_bytes,
                "error": self._error,
                "closed": self._closed,
                "pending_records": self._pending_records,
            }

    def _fail(self, reason: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = reason
            self._accepting = False
            self._stopping.set()

    def _drop(self, count: int) -> None:
        with self._lock:
            self._dropped_records += count
            self._pending_records -= count

    def _write_batch(self, records: list[dict]) -> None:
        lines = []
        batch_bytes = 0
        for record in records:
            try:
                line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                line_bytes = len(line.encode("utf-8"))
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")
                break
            if self._written_bytes + batch_bytes + line_bytes > self._max_bytes:
                self._fail(f"max_bytes exceeded: limit={self._max_bytes}")
                break
            lines.append(line)
            batch_bytes += line_bytes

        if lines:
            self._file.writelines(lines)
            self._file.flush()
            with self._lock:
                self._written_records += len(lines)
                self._written_bytes += batch_bytes
                self._pending_records -= len(lines)
        self._drop(len(records) - len(lines))

    def _run(self) -> None:
        batch = []
        try:
            while not (self._stopping.is_set() and self._queue.empty()):
                try:
                    batch = [self._queue.get(timeout=0.05)]
                except queue.Empty:
                    continue
                for _ in range(63):
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                self._write_batch(batch)
                batch = []
                if self._error is not None:
                    break
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            self._drop(len(batch))
        finally:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._drop(1)
            try:
                self._file.close()
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._closed = True
