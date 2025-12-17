#!/bin/bash
# MCRL Setup Script
# Sets up the project with uv and installs all dependencies

set -e

echo "============================================"
echo "MCRL Setup"
echo "============================================"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    source $HOME/.local/bin/env
fi

echo "uv version: $(uv --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
uv venv .venv --python 3.11

# Activate (for this script)
source .venv/bin/activate

# Install base dependencies
echo ""
echo "Installing base dependencies..."
uv pip install -e ".[dashboard,train]"

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    
    echo ""
    echo "Installing JAX with CUDA support..."
    # Install CUDA JAX
    uv pip install -U "jax[cuda12]"
    
    # Set library path for CUDA libs installed via pip
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/cudnn/lib:$SITE_PACKAGES/nvidia/cuda_cupti/lib:$SITE_PACKAGES/nvidia/cuda_runtime/lib:$SITE_PACKAGES/nvidia/cublas/lib:$SITE_PACKAGES/nvidia/cufft/lib:$SITE_PACKAGES/nvidia/cusolver/lib:$SITE_PACKAGES/nvidia/cusparse/lib:$SITE_PACKAGES/nvidia/nccl/lib:$SITE_PACKAGES/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
    
    echo ""
    echo "CUDA library path set."
else
    echo "No NVIDIA GPU detected. Using CPU JAX."
    echo "Note: Training will be very slow without GPU."
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX backend: {jax.default_backend()}')
print(f'JAX devices: {jax.devices()}')
"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Activate the environment with CUDA support:"
echo "  source scripts/activate.sh"
echo ""
echo "Run preflight checks:"
echo "  python experiments/scripts/preflight.py"
echo ""
echo "Start training:"
echo "  python experiments/scripts/train.py --debug"
echo ""
echo "Launch dashboard:"
echo "  python -m mcrl.dashboard.server --port 3000"
