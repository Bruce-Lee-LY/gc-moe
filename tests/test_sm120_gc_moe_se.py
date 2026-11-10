"""SM120 fused shared-expert accuracy against the SM100 reference."""
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_sm120_gc_moe import main


if __name__ == '__main__':
    main(shared=True)
