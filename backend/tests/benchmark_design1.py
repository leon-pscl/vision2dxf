"""Compatibility entry point for the design-2 benchmark.

Kept for callers that still invoke the design-1 filename; it simply delegates to
``benchmark_design2.py``.
"""
from __future__ import annotations

from benchmark_design2 import main

if __name__ == "__main__":
    main()
