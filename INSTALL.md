# Installation Guide

This guide provides detailed installation instructions for the Graphs4CausalFoundationModels repository.

## Prerequisites

- **Python**: 3.9 or higher (3.10+ recommended for full compatibility)
- **Git**: With Git LFS support
- **Operating System**: Linux, macOS, or Windows

## Installation Steps

### 1. Install Git LFS

Git LFS is required to download the pre-trained model checkpoints.

#### On macOS with Homebrew:
```bash
brew install git-lfs
git lfs install
```

#### On macOS without Homebrew:
```bash
# Download and install manually
cd /tmp
curl -L https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-darwin-arm64-v3.4.1.zip -o git-lfs.zip
unzip -q git-lfs.zip
cd git-lfs-3.4.1
mkdir -p ~/bin
cp git-lfs ~/bin/
export PATH="$HOME/bin:$PATH"
git lfs install
```

#### On Linux:
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# RHEL/CentOS
sudo yum install git-lfs
git lfs install
```

### 2. Clone the Repository

```bash
git clone https://github.com/ArikReuter/Graphs4CausalFoundationModels.git
cd Graphs4CausalFoundationModels

# Pull LFS objects (model checkpoints)
git lfs pull
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 4. Install Dependencies

#### Option A: Install PyTorch first (recommended)

Choose the appropriate command for your system:

**For CUDA (Linux/Windows with NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU or macOS (including Apple Silicon):**
```bash
pip install torch torchvision
```

Then install other dependencies:
```bash
pip install -r requirements.txt
```

#### Option B: Install all at once (may auto-select CPU version)
```bash
pip install -r requirements.txt
```

### 5. Verify Installation

Run the test script to ensure everything works:

```bash
python test_pretrained_model.py
```

You should see:
```
✅ ALL TESTS PASSED!
The repository is installed correctly and the pre-trained model works!
```

## Troubleshooting

### Issue: Git LFS files not downloaded

**Symptoms**: Model checkpoint files are small pointer files instead of large model files.

**Solution**:
```bash
git lfs install
git lfs pull
```

### Issue: Import errors or missing modules

**Solution**: Make sure virtual environment is activated and all dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: CUDA errors on macOS

**Solution**: macOS doesn't support CUDA. Install CPU/MPS version of PyTorch:
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

### Issue: Python version compatibility

**Symptoms**: Errors about packages requiring Python 3.10+

**Solution**: 
- Upgrade to Python 3.10 or higher, OR
- The core functionality works with Python 3.9 despite some warnings

### Issue: Git ignoring virtual environment

**Solution**: The `.gitignore` has been updated to exclude `venv/`, `env/`, and `.venv/` directories.

## Quick Start After Installation

```python
from src.models.GraphConditionedInterventionalPFN_sklearn import GraphConditionedInterventionalPFNSklearn
import numpy as np

# Load pre-trained model
wrapper = GraphConditionedInterventionalPFNSklearn(
    config_path="experiments/checkpoints/full_conditioned_model/config.yaml",
    checkpoint_path="experiments/checkpoints/full_conditioned_model/model.pt",
)
wrapper.load()

# Prepare data (must have 50 features)
X_obs = np.random.randn(10, 50)  # 10 observational samples, 50 features
T_obs = np.random.randn(10, 1)   # Treatment values
Y_obs = np.random.randn(10, 1)   # Outcome values

X_intv = np.random.randn(5, 50)  # 5 interventional queries
T_intv = np.random.randn(5, 1)   # Intervention values

# Create adjacency matrix (52x52 for T, Y, and 50 features)
adjacency_matrix = np.zeros((52, 52))
# Add some edges: T -> X[0:3], X[0:5] -> Y
adjacency_matrix[0, 2:5] = 1
adjacency_matrix[2:7, 1] = 1

# Predict interventional outcomes
predictions = wrapper.predict(
    X_obs, T_obs, Y_obs,
    X_intv, T_intv,
    adjacency_matrix,
    prediction_type="mode"
)

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions: {predictions}")
```

## System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for repository and dependencies
- **GPU**: Optional, but recommended for training. Pre-trained models work on CPU.

## Support

If you encounter issues not covered here, please:
1. Check the [main README](README.md) for additional documentation
2. Open an issue on GitHub with details about your system and the error message

## Development Setup

For development, you may want to install additional tools:

```bash
pip install pytest black flake8 mypy
```
