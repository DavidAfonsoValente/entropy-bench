"""Compare a freshly computed analysis against its committed artifact.

Shared by the ``--check`` gates in ``tools/``. Byte-comparing the JSON looks stricter but is
actually wrong: the correlation coefficients differ in their last floating-point bit between
CPython versions (measured 2026-08-29: -0.9345753609945784 against ...83), so a text comparison
reports a stale artifact purely from which interpreter ran the gate. A check that fails on the
interpreter is not a check. Integers, types and non-numeric values are still compared exactly, so
the tolerance cannot hide a real regression.
"""
from __future__ import annotations

import math


def _diff(committed, fresh, path="", tol=1e-9):
    """Paths where the committed audit disagrees with a fresh computation.

    Compares numerically rather than byte-for-byte. The correlation coefficients here differ in
    their last floating-point bit between interpreters (measured: -0.9345753609945784 against
    ...83 on two CPython versions), so a text comparison would report a stale artefact purely from
    which Python ran the gate -- a check that fails on the interpreter is not a check.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        out = []
        for key in set(committed) | set(fresh):
            if key not in committed or key not in fresh:
                out.append(f"{path}/{key} only in {'fresh' if key in fresh else 'committed'}")
            else:
                out += _diff(committed[key], fresh[key], f"{path}/{key}", tol)
        return out
    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return [f"{path} length {len(committed)} != {len(fresh)}"]
        return [d for i, (a, b) in enumerate(zip(committed, fresh))
                for d in _diff(a, b, f"{path}[{i}]", tol)]
    num = (int, float)
    both_num = (isinstance(committed, num) and isinstance(fresh, num)
                and not isinstance(committed, bool) and not isinstance(fresh, bool))
    if both_num:
        # Counts must match exactly. The tolerance exists only for accumulated float error, and
        # applying it to integers would let a cohort of 1,000,000,000 pass as 1,000,000,001. A
        # change of numeric type is also a real schema change, not drift.
        if isinstance(committed, int) != isinstance(fresh, int):
            return [f"{path}: type changed, {committed!r} != {fresh!r}"]
        if isinstance(committed, int):
            return [] if committed == fresh else [f"{path}: {committed} != {fresh}"]
        if math.isnan(committed) and math.isnan(fresh):
            return []
        if math.isinf(committed) or math.isinf(fresh):
            # abs(inf - inf) is NaN, which would compare as drift.
            return [] if committed == fresh else [f"{path}: {committed} != {fresh}"]
        scale = max(1.0, abs(committed), abs(fresh))
        return [] if abs(committed - fresh) <= tol * scale else [f"{path}: {committed} != {fresh}"]
    return [] if committed == fresh else [f"{path}: {committed!r} != {fresh!r}"]
