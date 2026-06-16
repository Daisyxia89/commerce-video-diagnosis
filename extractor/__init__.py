from __future__ import annotations


def run_extractor(*args, **kwargs):
    from .entry import run_extractor as _run_extractor

    return _run_extractor(*args, **kwargs)


__all__ = ["run_extractor"]
