"""CuTe DSL GC MoE public API."""

from importlib import import_module


# Load CUDA backends only when their public API is accessed. The SAIL
# environment can import the PPU runtime without the NVIDIA DSL.
_EXPORT_MODULES = {
    'GCSymmBuffer': '.gc',
    'fp8_fp4_gc_moe': '.gc',
    'get_block_m_for_gc_moe': '.gc',
    'get_symm_buffer_for_gc_moe': '.gc',
    'transform_weights_for_gc_moe': '.gc',
    'SM90GCSymmBuffer': '.sm90_gc',
    'get_symm_buffer_for_gc_moe_sm90': '.sm90_gc',
    'transform_weights_for_gc_moe_sm90': '.sm90_gc',
    'sm90_fp8_fp4_gc_moe': '.sm90_gc',
    'PPUGCSymmBuffer': '.ppu_gc',
    'get_symm_buffer_for_gc_moe_ppu': '.ppu_gc',
    'transform_weights_for_gc_moe_ppu': '.ppu_gc',
    'ppu_fp4_fp4_gc_moe': '.ppu_gc',
    'SM120GCSymmBuffer': '.sm120_gc',
    'get_symm_buffer_for_gc_moe_sm120': '.sm120_gc',
    'transform_weights_for_gc_moe_sm120': '.sm120_gc',
    'sm120_fp8_fp4_gc_moe': '.sm120_gc',
}
_BACKEND_MODULES = ('gc', 'sm90_gc', 'sm120_gc', 'ppu_gc')


def __getattr__(name):
    if name in _BACKEND_MODULES:
        return import_module(f'.{name}', __name__)
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_BACKEND_MODULES))


__all__ = [
    'SM90GCSymmBuffer',
    'get_symm_buffer_for_gc_moe_sm90',
    'transform_weights_for_gc_moe_sm90',
    'sm90_fp8_fp4_gc_moe',
    'PPUGCSymmBuffer',
    'get_symm_buffer_for_gc_moe_ppu',
    'transform_weights_for_gc_moe_ppu',
    'ppu_fp4_fp4_gc_moe',
    'SM120GCSymmBuffer',
    'get_symm_buffer_for_gc_moe_sm120',
    'transform_weights_for_gc_moe_sm120',
    'sm120_fp8_fp4_gc_moe',
    'GCSymmBuffer',
    'fp8_fp4_gc_moe',
    'get_block_m_for_gc_moe',
    'get_symm_buffer_for_gc_moe',
    'transform_weights_for_gc_moe',
]

__version__ = '0.6.1'
