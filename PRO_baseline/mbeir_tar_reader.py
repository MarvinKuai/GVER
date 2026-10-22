# -*- coding: utf-8 -*-
"""Random-access image reader over a merged ``mbeir_images.tar`` using a
pre-built offset index (see ``build_tar_index.py``).

Design goals
------------
1. **Multi-process DataLoader safe.**  The index dict is loaded once in
   the parent, but the ``open(tar, 'rb')`` file handle is created lazily
   *per process* on first access (avoids sharing an fd across forks,
   which can corrupt reads).  This matches how PyTorch DataLoader
   workers typically fork after the Dataset is constructed.

2. **O(1) random reads.**  A read is a single ``seek + read`` on the
   underlying tar file, followed by ``PIL.Image.open(BytesIO(...))``.

3. **Fallback to loose files.**  If the index does not know about a
   requested path (e.g. an image outside ``oven_images/``), we optionally
   fall back to reading the loose file under ``images_root``.  This keeps
   the reader compatible with datasets we did *not* pack into the tar.

Usage
-----
    reader = TarImageReader(tar_path, index_path)
    img = reader.read("mbeir_images/oven_images/05/oven_05021361.jpg")
    # -> PIL.Image (RGB)

Or as a factory for ``mbeir_data.py``::

    getter = reader.make_getter(rel_path)
    img = getter()  # deferred; safe to call inside DataLoader worker
"""
from __future__ import annotations

import io
import os
import pickle
import threading
from typing import Callable, Dict, Optional, Tuple

from PIL import Image


class TarImageReader:
    """Random-access image reader over a single .tar file."""

    def __init__(
        self,
        tar_path: str,
        index_path: str,
        loose_fallback_root: Optional[str] = None,
    ) -> None:
        if not os.path.isfile(tar_path):
            raise FileNotFoundError(f"tar not found: {tar_path}")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"index not found: {index_path}")

        self.tar_path = tar_path
        self.index_path = index_path
        self.loose_fallback_root = loose_fallback_root

        with open(index_path, "rb") as f:
            self._index: Dict[str, Tuple[int, int]] = pickle.load(f)

        # Per-process file handle: keyed by pid to survive DataLoader forks.
        self._fd_by_pid: Dict[int, "io.BufferedReader"] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # File handle management (multi-process safe)
    # ------------------------------------------------------------------
    def _get_fd(self) -> "io.BufferedReader":
        pid = os.getpid()
        fd = self._fd_by_pid.get(pid)
        if fd is None:
            with self._lock:
                fd = self._fd_by_pid.get(pid)
                if fd is None:
                    fd = open(self.tar_path, "rb")
                    self._fd_by_pid[pid] = fd
        return fd

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def __contains__(self, rel_path: str) -> bool:
        return rel_path in self._index

    def __len__(self) -> int:
        return len(self._index)

    def read_bytes(self, rel_path: str) -> bytes:
        """Return the raw file bytes for a member path inside the tar."""
        info = self._index.get(rel_path)
        if info is None:
            if self.loose_fallback_root:
                abs_path = os.path.join(self.loose_fallback_root, rel_path)
                if os.path.isfile(abs_path):
                    with open(abs_path, "rb") as f:
                        return f.read()
            raise KeyError(f"Not in tar index (and no fallback): {rel_path}")
        offset, size = info
        fd = self._get_fd()
        # NOTE: seek+read is not atomic w.r.t. threads sharing this fd,
        # but PyTorch DataLoader workers are separate *processes* (each
        # gets its own fd via _get_fd), so this is safe for standard use.
        # If threads within one process need to share, wrap with a lock.
        with self._lock:
            fd.seek(offset)
            data = fd.read(size)
        if len(data) != size:
            raise IOError(f"Short read: got {len(data)} bytes, expected {size} "
                          f"for {rel_path}")
        return data

    def read(self, rel_path: str, mode: str = "RGB") -> Image.Image:
        """Return a PIL.Image (RGB by default)."""
        data = self.read_bytes(rel_path)
        img = Image.open(io.BytesIO(data))
        if mode:
            img = img.convert(mode)
        return img

    def make_getter(self, rel_path: str, mode: str = "RGB") -> Callable[[], Image.Image]:
        """Return a zero-arg callable that reads the image on demand."""
        def _get() -> Image.Image:
            return self.read(rel_path, mode=mode)
        return _get

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            for fd in self._fd_by_pid.values():
                try:
                    fd.close()
                except Exception:
                    pass
            self._fd_by_pid.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience singleton for mbeir_data.py
# ---------------------------------------------------------------------------
_DEFAULT_READER: Optional[TarImageReader] = None


def get_default_reader() -> Optional[TarImageReader]:
    """Return a lazily-created global reader if env vars are set.

    Environment variables:
        MBEIR_TAR_PATH     -> path to merged mbeir_images.tar
        MBEIR_TAR_INDEX    -> path to oven_images index pickle
        MBEIR_LOOSE_ROOT   -> (optional) fallback root for loose files
    """
    global _DEFAULT_READER
    if _DEFAULT_READER is not None:
        return _DEFAULT_READER

    tar_path = os.environ.get("MBEIR_TAR_PATH")
    index_path = os.environ.get("MBEIR_TAR_INDEX")
    if not tar_path or not index_path:
        return None
    loose_root = os.environ.get("MBEIR_LOOSE_ROOT") or None
    _DEFAULT_READER = TarImageReader(tar_path, index_path, loose_fallback_root=loose_root)
    return _DEFAULT_READER


__all__ = ["TarImageReader", "get_default_reader"]
