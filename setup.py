"""Build the CuTe DSL package; GPU kernels are JIT compiled at runtime."""
import ast
import os
from pathlib import Path
import re
import subprocess

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent
GC_MOE_USE_LOCAL_VERSION = int(os.getenv('GC_MOE_USE_LOCAL_VERSION', '1')) == 1


def get_package_version():
    source = (ROOT / 'gc_moe' / '__init__.py').read_text(encoding='utf-8')
    match = re.search(r"^__version__\s*=\s*(.*)$", source, re.MULTILINE)
    if match is None:
        raise RuntimeError('gc_moe/__init__.py must define __version__')
    public_version = ast.literal_eval(match.group(1))
    if not GC_MOE_USE_LOCAL_VERSION:
        return public_version

    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if status:
            print('Warning: Git working directory is not clean; using +local.')
            revision = 'local'
        else:
            revision = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'local'

    cuda_local = os.getenv('GC_MOE_WHEEL_CUDA_VERSION', '').strip()
    if cuda_local:
        revision = f"{revision}.{cuda_local}"
    return f"{public_version}+{revision}"


if __name__ == '__main__':
    setup(
        name='gc_moe',
        version=get_package_version(),
        description='CuTe DSL Green Context MoE kernels',
        long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
        long_description_content_type='text/markdown',
        packages=find_namespace_packages(
            include=['gc_moe', 'gc_moe.*'],
            exclude=['*.__pycache__', '*.__pycache__.*'],
        ),
        python_requires='>=3.10',
        install_requires=[
            'torch>=2.11.0,<2.12',
            'nvidia-cutlass-dsl==4.8.1',
            'cuda-python>=13.2,<14',
            'cuda-bindings>=13.2,<14',
        ],
        zip_safe=False,
    )
