"""SM120 routed readiness with alternating inputs, output guards, and exact stats.

Run from the repository root with torchrun and one process per GPU.
Use --mode eager to exercise reuse without an outer CUDA Graph.
"""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_sm120_gc_moe import main


if __name__ == '__main__':
    main(readiness=True)
