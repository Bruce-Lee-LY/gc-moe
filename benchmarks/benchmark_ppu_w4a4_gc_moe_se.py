"""Benchmark PPU fused shared experts against MegaMoE with serial PTPC SE."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.ppu_gc_moe_bench import main


if __name__ == '__main__':
    main(routed_precision='fp4_fp4', shared_fp8_fp8_ptpc=True)
