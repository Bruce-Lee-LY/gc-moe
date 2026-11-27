"""PPU fused shared-expert readiness with alternating inputs and workspace reuse."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.ppu_gc_moe_cases import main


if __name__ == '__main__':
    main(routed_precision='fp4_fp4', shared_fp8_fp8_ptpc=True, readiness=True)
