"""Check PPU fused shared experts against grouped GEMM and PTPC GEMM."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_ppu_gc_moe import main


if __name__ == '__main__':
    main(shared=True)
