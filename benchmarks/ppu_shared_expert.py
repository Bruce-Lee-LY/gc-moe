"""Serial W8A8 PTPC shared expert for the PPU performance baseline."""

import ctypes
from pathlib import Path

import torch

from gc_moe.jit_kernels.impls.ppu_fp4_fp4_gc_moe import build_ppu_library


class SerialPTPCSharedExpert:
    def __init__(self, deep_gemm, w1, w2, capacity):
        source = Path(__file__).with_name('ppu_shared_fp8_swiglu.cu')
        self.library = build_ppu_library(Path.home() / '.cache/gc_moe/ppu', source)
        self.swiglu = self.library.gc_moe_shared_swiglu_fp8
        self.swiglu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_float, ctypes.c_void_p]
        self.swiglu.restype = ctypes.c_int
        self.gemm = deep_gemm.fp8_gemm_nt
        self.w1, self.w2 = w1, w2
        hidden, intermediate = w2[0].shape
        self.gate_up = w1[0].new_empty((capacity, 2 * intermediate), dtype=torch.bfloat16)
        self.mid = w1[0].new_empty((capacity, intermediate))
        self.scales = w1[1].new_empty((capacity, 1))
        self.output = w2[0].new_empty((capacity, hidden), dtype=torch.bfloat16)

    def bind(self, tokens, inputs, clamp):
        inputs = tuple(tensor[:tokens] for tensor in inputs)
        gate_up, mid, scales, output = (tensor[:tokens] for tensor in
                                        (self.gate_up, self.mid, self.scales, self.output))
        swiglu_args = (gate_up.data_ptr(), mid.data_ptr(), scales.data_ptr(),
                       tokens, mid.shape[1], clamp)

        def launch():
            self.gemm(inputs, self.w1, gate_up)
            stream = torch.cuda.current_stream(gate_up.device)
            status = self.swiglu(*swiglu_args, stream.cuda_stream)
            if status:
                raise RuntimeError(f'PPU benchmark SwiGLU launch failed with HGGC status {status}')
            self.gemm((mid, scales), self.w2, output)
            return output

        return launch
