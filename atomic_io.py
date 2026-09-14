"""Write artefacts so a failure never leaves a complete-looking file behind.

``run_pipeline.py`` decides whether a stage has already run by asking whether its
output files exist -- the ``file_exists`` guards around each step.  A stage that
dies part-way through writing therefore leaves a file carrying the final name,
which a later re-run accepts as finished work: it prints ``[SKIP]`` and builds on
truncated data, silently.

Reproduced before this module existed: stage 04 writing a two-line
``filed_flights.csv`` and then raising leaves the pipeline at exit 1, and a re-run
over that tree skips trajectory generation entirely.

Writing to a temporary name in the same directory and renaming into place on
success closes that.  ``os.replace`` is atomic within a filesystem, so the
destination either does not exist yet or is the complete file -- there is no
observable in-between.  The temporary lives in the destination's own directory
because a rename across filesystems is not atomic.

Deliberately NOT a completion-marker scheme: a marker file would land in the
output tree, and the regression harness fingerprints every file there, so markers
would change generator output.  These helpers add no file that survives a
successful run and change no byte of what is written.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

__all__ = ["atomic_to_csv", "atomic_open", "atomic_group", "PARTIAL_SUFFIX"]

#: appended to the destination name while the write is in flight
PARTIAL_SUFFIX = ".partial"


def _partial(path: Path) -> Path:
    return path.with_name(path.name + PARTIAL_SUFFIX)


def atomic_to_csv(df, path, **kwargs) -> None:
    """``df.to_csv(path, **kwargs)``, but the destination is never half-written."""
    path = Path(path)
    tmp = _partial(path)
    try:
        df.to_csv(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_open(path, mode: str = "w", **kwargs):
    """``open(path, "w")``, but the destination is never half-written.

    Text or binary write modes only -- there is nothing to append to a file that
    does not exist yet, so append modes are rejected rather than silently doing
    something surprising.
    """
    if "a" in mode or "r" in mode or "+" in mode:
        raise ValueError(f"atomic_open supports write modes only, got {mode!r}")
    path = Path(path)
    tmp = _partial(path)
    fh = open(tmp, mode, **kwargs)
    try:
        yield fh
    except BaseException:
        fh.close()
        tmp.unlink(missing_ok=True)
        raise
    else:
        fh.close()
        os.replace(tmp, path)


@contextmanager
def atomic_group():
    """Several files that must appear together, or not at all.

    Per-file atomicity is not always enough.  ``run_pipeline.py`` skips a stage when
    its output exists, but stage 04's guard checks only ``filed_flights.csv`` while
    the stage writes three files -- the other two, ``flights.csv`` and
    ``aircrafts.csv``, already exist from stage 01, so their presence says nothing
    about whether stage 04 ran.  Writing each file atomically would still allow
    ``filed_flights.csv`` to land, the stage to die, and a re-run to skip it, leaving
    a filed plan beside stage 01's un-rewritten ``flights.csv``.

    Every other stage's guard checks all of that stage's outputs, so a partial set
    heals itself: a missing file makes the guard re-run the stage.  Stage 04 is the
    exception, and this is for it.

    Writes go to ``.partial`` names and are renamed only once the block completes:

        with atomic_group() as g:
            g.to_csv(df, data_dir / "filed_flights.csv", index=False)
            with g.open(data_dir / "aircrafts.csv", newline="") as fh:
                ...

    The closing renames are separate ``os.replace`` calls rather than one atomic
    transaction -- POSIX offers no such thing across several files -- but nothing
    between them can fail, so the exposure is the rename loop rather than the whole
    computation that produced the data.
    """
    pending: list[tuple[Path, Path]] = []

    class _Group:
        def to_csv(self, df, path, **kwargs) -> None:
            path = Path(path)
            tmp = _partial(path)
            df.to_csv(tmp, **kwargs)
            pending.append((tmp, path))

        @contextmanager
        def open(self, path, mode: str = "w", **kwargs):
            if "a" in mode or "r" in mode or "+" in mode:
                raise ValueError(f"atomic_group.open supports write modes only, got {mode!r}")
            path = Path(path)
            tmp = _partial(path)
            fh = open(tmp, mode, **kwargs)
            try:
                yield fh
            finally:
                fh.close()
            pending.append((tmp, path))

    try:
        yield _Group()
    except BaseException:
        for tmp, _ in pending:
            tmp.unlink(missing_ok=True)
        raise
    else:
        for tmp, final in pending:
            os.replace(tmp, final)
