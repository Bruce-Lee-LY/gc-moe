"""Benchmark SM120 fused shared experts against Cake, or time GC alone."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_sm120_gc_moe import main


if __name__ == '__main__':
    main(shared=True)
