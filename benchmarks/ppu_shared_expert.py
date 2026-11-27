"""Serial W8A8 PTPC shared expert for the PPU performance baseline."""

import torch


class SerialFP8FP8PTPCSharedExpert:
    def __init__(self, deep_gemm, w1, w2, capacity):
        from sglang.kernels.ops.elementwise.silu_mul_quant import silu_and_mul_post_per_token_quant_fp8

        self.swiglu = silu_and_mul_post_per_token_quant_fp8
        self.gemm = deep_gemm.fp8_gemm_nt
        self.w1, self.w2 = w1, w2
        hidden, intermediate = w2[0].shape
        self.gate_up = w1[0].new_empty((capacity, 2 * intermediate), dtype=torch.bfloat16)
        self.output = w2[0].new_empty((capacity, hidden), dtype=torch.bfloat16)

    def bind(self, tokens, inputs, clamp):
        inputs = tuple(tensor[:tokens] for tensor in inputs)
        gate_up, output = self.gate_up[:tokens], self.output[:tokens]

        def launch():
            self.gemm(inputs, self.w1, gate_up)
            quantized = self.swiglu(gate_up, swiglu_limit=clamp)
            self.gemm(quantized, self.w2, output)
            return output

        return launch
