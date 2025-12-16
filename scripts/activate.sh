#!/bin/bash
# Activate MCRL environment with CUDA support
# Usage: source scripts/activate.sh

# Activate venv
source .venv/bin/activate

# Set CUDA library paths for pip-installed packages
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

echo "MCRL environment activated with CUDA support"
python -c "import jax; print(f'JAX backend: {jax.default_backend()}, devices: {jax.devices()}')"
