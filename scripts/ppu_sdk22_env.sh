# Source with the directory containing PPU_SDK, nccl22 and the Python environment.
ppu_gc_env_root=${1:?usage: source ppu_sdk22_env.sh ENVIRONMENT_DIRECTORY}
source "$ppu_gc_env_root/PPU_SDK/envsetup.sh" >/dev/null
source "$ppu_gc_env_root/nccl22/pccl/envsetup.sh" >/dev/null

ppu_python="$ppu_gc_env_root/venv"
if [[ -f "$ppu_python/bin/activate" ]]; then
    source "$ppu_python/bin/activate"
else
    export PATH="$ppu_python/bin:$PATH"
fi
ppu_torch_lib=$(python -c 'import sysconfig; print(sysconfig.get_path("platlib") + "/torch/lib")')
export PPU_HOME="$PPU_SDK" CUDA_HOME="$PPU_SDK/CUDA_SDK"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$PCCL_HOME/lib:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$ppu_python/lib:$ppu_torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -d /opt/conda310/lib ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda310/lib"
fi
export DG_JIT_HGCC_COMPILER="$CUDA_HOME/bin/hgcc"
export PYTHONUNBUFFERED=1
unset HGGC_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES DG_RANK_LAUNCH
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE
