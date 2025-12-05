# AstroDynamics Environment Setup Guide (macOS Apple Silicon)

This project relies on high-performance numerical libraries (`numpy`, `scipy`, `numba`, `heyoka`) that are best managed using **Conda**.
For Apple Silicon (M1/M2/M3) Macs, we strongly recommend **Miniforge**, which defaults to the `conda-forge` channel and provides optimized binaries for ARM64.

## 1. Install Miniforge

1.  Download the installer script:
    ```bash
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
    ```
2.  Run the installer:
    ```bash
    bash Miniforge3-MacOSX-arm64.sh
    ```
3.  Follow the prompts (accept license, confirm location). When asked to initialize Miniforge3, type `yes`.
4.  **Restart your terminal** for changes to take effect.

## 2. Create Project Environment

Create a dedicated environment for this project (e.g., `astro_env`) with Python 3.10 or 3.11 (Heyoka supports these well).

```bash
# Create environment
conda create -n astro_env python=3.11

# Activate environment
conda activate astro_env
```

## 3. Install Dependencies

Install the required scientific libraries. Using `conda` ensures that binary dependencies (like LLVM for Numba, or Boost/SPDLog for Heyoka) are correctly installed.

```bash
# Install core libraries and Heyoka
conda install -c conda-forge numpy scipy matplotlib skyfield numba heyoka.py jupyterlab

# Install other Python dependencies (if any are missing from conda)
pip install fastapi uvicorn pydantic
```

## 4. Verify Installation

Run the following Python commands to check if `heyoka` and `numba` are working correctly.

```bash
python -c "import heyoka; print(f'Heyoka version: {heyoka.__version__}')"
python -c "import numba; print(f'Numba version: {numba.__version__}')"
```

## 5. VS Code Configuration

To use this environment in VS Code:

1.  Open the Command Palette (`Cmd + Shift + P`).
2.  Type `Python: Select Interpreter`.
3.  Select the entry corresponding to `astro_env` (e.g., `~/miniforge3/envs/astro_env/bin/python`).

## 6. Running the Project

Now you can run the scripts using the optimized environment:

```bash
# Run Benchmark
python universe/benchmark.py

# Run Server
python universe/server.py
```
