"""Benchmark PPU W4A4 GC MoE against W4A4 MegaMoE."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.ppu_gc_moe_bench import main


if __name__ == '__main__':
    main(routed_precision='fp4_fp4')
