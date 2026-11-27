"""GC MoE public API for CUDA and Sail."""

from importlib import import_module


# Resolve only the requested backend so a Sail installation does not load
# NVIDIA-specific dependencies, and vice versa.
_BACKENDS = {
    '.gc': (
        'GCSymmBuffer',
        'fp8_fp4_gc_moe',
        'get_block_m_for_gc_moe',
        'get_symm_buffer_for_gc_moe',
        'transform_weights_for_gc_moe',
    ),
    '.gc_k3': (
        'GCK3SymmBuffer',
        'fp8_fp4_gc_moe_k3',
        'get_block_m_for_gc_moe_k3',
        'get_symm_buffer_for_gc_moe_k3',
        'transform_weights_for_gc_moe_k3',
    ),
    '.sm90_gc': (
        'SM90GCSymmBuffer',
        'get_symm_buffer_for_gc_moe_sm90',
        'transform_weights_for_gc_moe_sm90',
        'sm90_fp8_fp4_gc_moe',
    ),
    '.sm120_gc': (
        'SM120GCSymmBuffer',
        'get_symm_buffer_for_gc_moe_sm120',
        'transform_weights_for_gc_moe_sm120',
        'sm120_fp8_fp4_gc_moe',
    ),
    '.ppu_gc': (
        'PPUGCSymmBuffer',
        'get_symm_buffer_for_gc_moe_ppu',
        'transform_weights_for_gc_moe_ppu',
        'ppu_fp4_fp4_gc_moe',
        'ppu_fp8_fp4_gc_moe',
    ),
}
_EXPORTS = {name: backend for backend, names in _BACKENDS.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if f'.{name}' in _BACKENDS:
        return import_module(f'.{name}', __name__)
    backend = _EXPORTS.get(name)
    if backend is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(backend, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys() | {name[1:] for name in _BACKENDS})


__version__ = '0.7.0'
