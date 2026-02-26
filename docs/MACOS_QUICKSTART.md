# VFS-Wind on macOS (Apple Silicon) Quick Start Guide

This guide explains how to build and run VFS-Wind with GPU acceleration on macOS with Apple Silicon (M1/M2/M3/M4) using the OpenMP backend.

## Prerequisites

### 1. Install Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Required Dependencies

```bash
# Essential build tools
brew install cmake

# MPI (required for parallel execution)
brew install open-mpi

# PETSc (numerical library)
brew install petsc

# HYPRE (preconditioner library)
brew install hypre

# OpenMP (required for Kokkos OpenMP backend on macOS)
brew install libomp
```

### 3. Verify Installation

```bash
# Check versions
cmake --version      # Should be 3.16+
mpirun --version     # Should show OpenMPI
pkg-config --modversion PETSc  # Should show version

# Verify libomp is installed
ls /opt/homebrew/opt/libomp/lib/libomp.dylib
```

## Building VFS-Wind

### Option 1: Quick Build (Recommended)

```bash
# Navigate to VFS-Wind directory
cd /path/to/VFS-Wind

# Configure with GPU (OpenMP backend for Apple Silicon)
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON \
    -DKOKKOS_BACKEND=OPENMP \
    -DENABLE_TESTING=ON

# Build (use all available cores)
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Option 2: Debug Build

```bash
cmake -B build-debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DENABLE_GPU=ON \
    -DKOKKOS_BACKEND=OPENMP \
    -DENABLE_TESTING=ON

cmake --build build-debug -j$(sysctl -n hw.ncpu)
```

### Option 3: CPU-Only Build (No GPU/Kokkos)

```bash
cmake -B build-cpu \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=OFF

cmake --build build-cpu -j$(sysctl -n hw.ncpu)
```

## Running Tests

### Run All GPU Tests

```bash
cd build

# Set OpenMP environment (prevents thread binding warnings)
export OMP_PROC_BIND=false

# Run all GPU tests
ctest -L gpu --output-on-failure
```

### Run Individual Test Suites

```bash
# Smoke tests (basic GPU functionality)
OMP_PROC_BIND=false ./tests/test_gpu_smoke

# Kernel tests (convection, viscous, etc.)
OMP_PROC_BIND=false ./tests/test_gpu_kernels

# Phase 4 tests (LES, RANS, level-set)
OMP_PROC_BIND=false ./tests/test_gpu_phase4

# All tests with timing
OMP_PROC_BIND=false ctest -L gpu --output-on-failure -V
```

## Running Simulations

### Basic Simulation

```bash
cd build

# Run with 1 MPI process
mpirun -np 1 ./Source/vwis -xml ../examples/Test_01_3D_Sloshing/control.xml

# Run with 4 MPI processes
mpirun -np 4 ./Source/vwis -xml ../examples/Test_01_3D_Sloshing/control.xml
```

### With Timing Output

```bash
# Enable timing (see "Adding Timers" section below)
mpirun -np 1 ./Source/vwis -xml control.xml -log_view

# GPU timing with environment variable
VFSWIND_TIMING=1 mpirun -np 1 ./Source/vwis -xml control.xml
```

### OpenMP Thread Control

```bash
# Set number of OpenMP threads (default: all cores)
export OMP_NUM_THREADS=8

# Disable thread binding (recommended for macOS)
export OMP_PROC_BIND=false

# Run simulation
mpirun -np 1 ./Source/vwis -xml control.xml
```

## Performance Tips for Apple Silicon

### 1. Optimal Thread Count

Apple Silicon has performance (P) and efficiency (E) cores. For best performance:

```bash
# M4: 4P + 6E cores = 10 total, use P-cores only
export OMP_NUM_THREADS=4

# M4 Pro: 10P + 4E cores = 14 total
export OMP_NUM_THREADS=10

# M4 Max: 12P + 4E cores = 16 total
export OMP_NUM_THREADS=12
```

### 2. Memory Considerations

Apple Silicon has unified memory. Large simulations benefit from:

```bash
# Increase PETSc memory pool
export PETSC_OPTIONS="-malloc_hbw"
```

### 3. Disable Thread Migration

```bash
export OMP_PROC_BIND=false
export OMP_PLACES=cores
```

## Troubleshooting

### Error: "libomp.dylib not found"

```bash
# Reinstall libomp
brew reinstall libomp

# Or specify path manually in CMake
cmake -B build \
    -DENABLE_GPU=ON \
    -DKOKKOS_BACKEND=OPENMP \
    -DOpenMP_ROOT=/opt/homebrew/opt/libomp
```

### Error: "Kokkos not initialized"

Make sure Kokkos is initialized before using GPU functions. Check that `main.c` calls:
```c
VFSWind_Kokkos_Initialize(&argc, &argv);
// ... simulation ...
VFSWind_Kokkos_Finalize();
```

### Warning: "OMP: Warning #181: OMP_PROC_BIND"

This is harmless on macOS. Suppress with:
```bash
export OMP_PROC_BIND=false
```

### Slow Performance

1. Check thread count: `echo $OMP_NUM_THREADS`
2. Verify Release build: `cmake --build build --config Release`
3. Check for thermal throttling (Activity Monitor → CPU)

## Example: Channel Flow Benchmark

```bash
cd build

# Set optimal threads for M4
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=false

# Run channel flow test case
mpirun -np 1 ./Source/vwis -xml ../examples/Test_10_ChannelFlow_Retau3000/control.xml

# With timing
VFSWIND_TIMING=1 mpirun -np 1 ./Source/vwis -xml ../examples/Test_10_ChannelFlow_Retau3000/control.xml
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `cmake -B build -DENABLE_GPU=ON -DKOKKOS_BACKEND=OPENMP` | Configure with OpenMP GPU |
| `cmake --build build -j` | Build with all cores |
| `ctest --test-dir build -L gpu` | Run GPU tests |
| `OMP_NUM_THREADS=4 mpirun -np 1 ./Source/vwis -xml control.xml` | Run simulation |
| `VFSWIND_TIMING=1 ...` | Enable timing output |

## Next Steps

- See `docs/GPU_BUILD_GUIDE.md` for detailed build options
- See `docs/GPU_ARCHITECTURE.md` for architecture documentation
- See `examples/` for test cases
