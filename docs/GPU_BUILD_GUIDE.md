# VFS-Wind GPU Build Guide

This guide explains how to build VFS-Wind with GPU acceleration enabled.

## Prerequisites

### Required Software

1. **CMake** 3.16 or higher
2. **C++ Compiler** with C++17 support (GCC 9+, Clang 10+, or NVCC)
3. **MPI** (OpenMPI, MPICH, or vendor MPI)
4. **PETSc** 3.16+ (with GPU support recommended)
5. **HYPRE** (with GPU support optional)

### GPU-Specific Requirements

#### For NVIDIA GPUs (CUDA)
- CUDA Toolkit 11.0 or higher
- NVIDIA GPU with compute capability 7.0+ (Volta, Turing, Ampere, Hopper)

#### For AMD GPUs (HIP/ROCm)
- ROCm 5.0 or higher
- AMD GPU with CDNA or RDNA architecture

#### For Intel GPUs (SYCL)
- Intel oneAPI DPC++ compiler
- Intel GPU (Gen9+, Xe)

## Quick Start

### NVIDIA GPU Build

```bash
# Configure with CUDA backend
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON \
    -DKOKKOS_BACKEND=CUDA \
    -DKokkos_ARCH_AMPERE80=ON  # For A100 GPUs

# Build
cmake --build build -j$(nproc)

# Run GPU smoke test
./build/tests/test_gpu_smoke
```

### AMD GPU Build

```bash
# Configure with HIP backend
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON \
    -DKOKKOS_BACKEND=HIP \
    -DKokkos_ARCH_VEGA90A=ON  # For MI250X

cmake --build build -j$(nproc)
```

### CPU-Only Build (OpenMP)

```bash
# Configure with OpenMP backend (multi-threaded CPU)
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GPU=ON \
    -DKOKKOS_BACKEND=OPENMP

cmake --build build -j$(nproc)
```

### CPU-Only Build (No GPU)

```bash
# Standard build without GPU support
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_GPU` | OFF | Enable GPU acceleration via Kokkos |
| `KOKKOS_BACKEND` | CUDA | Backend: CUDA, HIP, SYCL, OPENMP, SERIAL |

### Kokkos Architecture Flags

When using `ENABLE_GPU=ON`, you can specify the target GPU architecture:

#### NVIDIA GPUs
| Flag | GPU Generation |
|------|----------------|
| `Kokkos_ARCH_VOLTA70` | V100 |
| `Kokkos_ARCH_TURING75` | RTX 20xx, T4 |
| `Kokkos_ARCH_AMPERE80` | A100, A30 |
| `Kokkos_ARCH_AMPERE86` | RTX 30xx, A10 |
| `Kokkos_ARCH_HOPPER90` | H100 |

#### AMD GPUs
| Flag | GPU Generation |
|------|----------------|
| `Kokkos_ARCH_VEGA906` | MI50, MI60 |
| `Kokkos_ARCH_VEGA90A` | MI200, MI250X |
| `Kokkos_ARCH_GFX90A` | MI210 |

## Running with GPU

### Basic Usage

```bash
# Run simulation on GPU (same command as CPU)
mpirun -np 4 ./build/Source/vwis -xml control.xml

# The GPU will be detected and used automatically
```

### Environment Variables

```bash
# Select specific GPU (for multi-GPU systems)
export CUDA_VISIBLE_DEVICES=0  # NVIDIA
export HIP_VISIBLE_DEVICES=0   # AMD

# Set GPU memory pool size (optional)
export KOKKOS_NUM_THREADS=1  # Usually 1 for GPU
```

### PETSc GPU Options

To enable PETSc GPU operations (vectors, matrices, solvers):

```bash
mpirun -np 4 ./build/Source/vwis -xml control.xml \
    -vec_type cuda \
    -mat_type aijcusparse \
    -pc_type hypre \
    -pc_hypre_boomeramg_device_level 1
```

## Verification

### Run GPU Smoke Test

```bash
# Build and run GPU tests
cmake --build build --target test_gpu_smoke
./build/tests/test_gpu_smoke
```

Expected output:
```
==============================================================
VFS-Wind GPU Acceleration Enabled
==============================================================
  Kokkos Version:     4.3.0
  Backend:            CUDA
  Execution Space:    Kokkos::Cuda
  Memory Space:       Kokkos::CudaSpace
  CUDA Device:        0
==============================================================

Running GPU smoke test...
GPU smoke test PASSED (1000 elements verified)
[==========] 7 tests from 1 test suite ran.
[  PASSED  ] 7 tests.
```

### Check GPU Memory

Add `-log_view` to see memory usage:

```bash
mpirun -np 1 ./build/Source/vwis -xml control.xml -log_view
```

## Troubleshooting

### Common Issues

#### 1. Kokkos not found

```
CMake Error: Could not find Kokkos
```

**Solution**: Kokkos will be automatically fetched. Ensure internet access or install Kokkos manually:

```bash
git clone https://github.com/kokkos/kokkos.git
cd kokkos
cmake -B build -DCMAKE_INSTALL_PREFIX=/path/to/install \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build build --target install
```

Then configure VFS-Wind with:
```bash
cmake -B build -DENABLE_GPU=ON -DKokkos_DIR=/path/to/install/lib/cmake/Kokkos
```

#### 2. CUDA not found

```
CMake Error: Could not find CUDA
```

**Solution**: Set CUDA path:
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

#### 3. GPU out of memory

```
CUDA error: out of memory
```

**Solution**:
- Use fewer MPI ranks per GPU
- Reduce problem size
- Enable unified memory (experimental):
  ```bash
  export KOKKOS_ENABLE_UNIFIED_MEMORY=1
  ```

#### 4. GPU kernel fails

```
PETSC ERROR: Caught signal number 11 SEGV
```

**Solution**:
- Rebuild with debugging: `-DCMAKE_BUILD_TYPE=Debug`
- Run with CUDA memcheck: `compute-sanitizer ./build/Source/vwis ...`
- Check GPU memory with `nvidia-smi`

#### 5. NaN errors at first timestep (dx_min=0, Inlet Area=0)

```
CFL 1.0 time step=0.000000, dx_min=0.00000
*** Max Ucat = nan
ERROR detected by Hypre: INFs and/or NaNs detected in input
```

**Solution**: This issue was caused by the grid file path being ignored when using
XML configuration with XYZ format. The code was hardcoded to look for "xyz.dat"
instead of the user-specified filename. This has been fixed in the `gpu_portability`
branch.

If you encounter this issue:
- Ensure your grid file exists with the correct name specified in control.xml
- Update to the latest `gpu_portability` branch
- See `docs/GPU_METRIC_SYNC_FIX.md` for details

**Verification**: After the fix, you should see `Reading mesh.xyz` (or your filename)
in the output, positive values for `dx_min` and `Inlet Area`, and the simulation
should complete without NaN errors.

#### 6. PETSc DMLocalToGlobal warnings with periodic BCs

```
[0]PETSC ERROR: No support for this operation for this object type
[0]PETSC ERROR: Available only for boundary none or with parallelism in z direction
```

**Cause**: PETSc's `DMLocalToGlobal` has restrictions when periodic boundary
conditions are combined with certain MPI domain decompositions. For example, with
`ii_periodic` (x-direction periodic) and the domain split in x-direction.

**Solution**: These are warnings that typically don't stop the simulation. To avoid
them, choose an MPI process count that results in a compatible domain decomposition:
- For channel flow with `ii_periodic` and `kk_periodic`: use NP=1 or NP=4 (not 2)
- Check the "DM Distribution" output (e.g., `2 1 1` means split 2x in x, 1x in y, 1x in z)
- Ensure the periodic direction has either 1 partition or the decomposition is in z

## Performance Tips

1. **Use one MPI rank per GPU** for best performance
2. **Enable GPU-aware MPI** if available:
   ```bash
   mpirun --mca mpi_cuda_support 1 ...
   ```
3. **Profile with nsys/nvprof**:
   ```bash
   nsys profile -o report mpirun -np 1 ./build/Source/vwis ...
   ```

## File Structure

```
Source/
├── les.c                        # LES solver (GPU-integrated)
├── rhs.c                        # Convection/Viscous (GPU-integrated)
├── k-omega.c                    # RANS k-omega (GPU-integrated)
├── level.c                      # Level-set (GPU-integrated)
├── main.c                       # Modified to call GPU init/finalize
└── gpu/
    ├── gpu_config.hpp           # GPU configuration and type aliases
    ├── gpu_dispatch.h           # C interface for GPU kernels
    ├── gpu_dispatch.cpp         # C++ dispatch implementation
    ├── gpu_solver.hpp           # GPU solver configuration (C++)
    ├── gpu_solver.h             # GPU solver configuration (C interface)
    ├── gpu_solver.cpp           # GPU solver implementation
    ├── kokkos_init.cpp          # Kokkos initialization/finalization
    ├── kokkos_init.h            # C interface header
    └── kernels/
        ├── kernels.hpp          # Unified kernel header
        ├── petsc_kokkos.hpp     # PETSc-Kokkos integration utilities
        ├── convection_kernel.hpp # QUICK scheme convection
        ├── viscous_kernel.hpp    # Viscous diffusion
        ├── pressure_gradient_kernel.hpp # Pressure gradient
        ├── ibm_types.hpp         # IBM GPU data structures
        ├── ibm_interpolation_kernel.hpp # IBM velocity interpolation
        ├── ibm_force_kernel.hpp  # IBM force spreading
        ├── ibm_kernels.hpp       # IBM unified header
        ├── levelset_kernel.hpp   # Level-set advection (WENO schemes)
        ├── les_kernel.hpp        # LES turbulence models
        ├── rans_kernel.hpp       # RANS k-omega models
        └── turbulence_kernels.hpp # Phase 4 unified header

tests/
└── gpu/
    ├── test_gpu_smoke.cpp       # GPU smoke tests (10 tests)
    ├── test_kernels.cpp         # Kernel unit tests (5 tests)
    ├── test_gpu_solver.cpp      # GPU solver tests (9 tests)
    ├── test_ibm_kernels.cpp     # IBM kernel tests (9 tests)
    └── test_phase4_kernels.cpp  # Level-set & turbulence tests (17 tests)
```

## Current Status

**Phase 0 (Infrastructure)**: Complete ✓
- CMake integration with Kokkos
- GPU initialization/finalization
- Smoke tests
- macOS OpenMP support (Homebrew libomp)

**Phase 1 (Core Kernels)**: Complete ✓
- Convection kernel (QUICK scheme)
- Viscous kernel (with LES/RANS support)
- Pressure gradient kernel
- PETSc-Kokkos integration utilities
- Kernel unit tests (5 tests passing)

**Phase 2 (Linear Solvers)**: Complete ✓
- GPU backend detection (CUDA, HIP, Kokkos, SYCL)
- PETSc GPU vector/matrix type configuration
- HYPRE GPU preconditioner options
- DM and KSP GPU configuration utilities
- C and C++ interface APIs
- GPU solver unit tests (9 tests passing)

**Phase 3 (IBM/FSI)**: Complete ✓
- GPU-friendly IBM data structures (SoA layout)
- IBM velocity interpolation kernel
- IBM force spreading kernel (with delta functions)
- Direct forcing method implementation
- No-slip and free-slip boundary conditions
- IBM kernel unit tests (9 tests passing)

**Phase 4 (Level-Set/Turbulence)**: Complete ✓
- Level-set advection with WENO3/WENO5 schemes
- Level-set reinitialization (signed distance)
- Two-phase property computation (Heaviside smoothing)
- LES: Static and Dynamic Smagorinsky models
- LES: van Driest wall damping
- RANS: k-omega Wilcox (Low-Re and High-Re)
- RANS: k-omega SST (Menter) with F1/F2 blending
- Phase 4 kernel unit tests (17 tests passing)

**Phase 5 (Solver Integration)**: Complete ✓
- GPU dispatch layer (gpu_dispatch.h/cpp)
- Integration into les.c (LES eddy viscosity)
- Integration into rhs.c (Convection, Viscous)
- Integration into k-omega.c (RANS k-omega RHS)
- Integration into level.c (Level-set advection)
- Automatic CPU fallback on GPU failure
- Extensive code documentation

**Total GPU Tests**: 50 passing

## Documentation

- **`docs/GPU_ARCHITECTURE.md`** - Detailed explanation of the GPU architecture,
  design decisions, and reasoning behind each choice. **Recommended reading** for
  developers who want to understand or extend the GPU code.
- **`docs/GPU_PORTING_PLAN.md`** - Original porting roadmap and milestones.
- **`docs/GPU_METRIC_SYNC_FIX.md`** - Documentation of the metric vector synchronization
  fix that resolved NaN errors with periodic boundary conditions.

## Using the GPU Kernels

The GPU kernels are header-only and can be used as follows:

```cpp
#include "gpu/kernels/kernels.hpp"

using namespace vfswind::gpu;
using namespace vfswind::gpu::kernels;

// Create domain info
KernelDomainInfo domain = createDomainInfo(da, reynolds_number);

// Allocate Kokkos Views
VectorView3D<> ucont("ucont", NZ, NY, NX);
VectorView3D<> ucat("ucat", NZ, NY, NX);
ScalarView3D<> nvert("nvert", NZ, NY, NX);
VectorView3D<> conv("conv", NZ, NY, NX);

// Copy PETSc data to Kokkos Views
copyPetscVectorToView(fda, lUcont, ucont);
copyPetscVectorToView(fda, lUcat, ucat);
copyPetscScalarToView(da, lNvert, nvert);

// Execute kernel
ConvectionKernel::execute(ucont, ucat, nvert, conv, domain);

// Copy result back to PETSc
copyViewToPetscVector(fda, conv, Conv);
```

## Using the GPU Solver

### C++ Interface

```cpp
#include "gpu/gpu_solver.hpp"

using namespace vfswind::gpu;

// Print GPU solver configuration
printGPUSolverInfo();

// Apply GPU solver options (call before KSPCreate)
GPUSolverConfig config;
applyGPUSolverOptions(config);

// Configure DM for GPU vectors/matrices
configureDMForGPU(dm, config);

// Configure KSP for GPU
configureKSPForGPU(ksp, config);
```

### C Interface

```c
#include "gpu/gpu_solver.h"

// Check GPU support
if (VFSWind_HasGPUSolverSupport()) {
    printf("GPU backend: %s\n", VFSWind_GetGPUBackendName());
}

// Apply GPU options (call early, before solver setup)
VFSWind_ApplyGPUSolverOptions();

// Configure DM for GPU
VFSWind_ConfigureDMForGPU(dm);

// Configure KSP for GPU
VFSWind_ConfigureKSPForGPU(ksp);

// Print configuration info
VFSWind_PrintGPUSolverInfo();
```

### Runtime GPU Solver Options

You can also configure GPU solvers via command-line options:

```bash
# NVIDIA CUDA
mpirun -np 4 ./build/Source/vwis -xml control.xml \
    -vec_type cuda \
    -mat_type aijcusparse \
    -dm_vec_type cuda \
    -dm_mat_type aijcusparse \
    -pc_type hypre \
    -pc_hypre_boomeramg_device_level 1

# AMD HIP/ROCm
mpirun -np 4 ./build/Source/vwis -xml control.xml \
    -vec_type hip \
    -mat_type aijhipsparse \
    -pc_type hypre

# Kokkos (auto-detect backend)
mpirun -np 4 ./build/Source/vwis -xml control.xml \
    -vec_type kokkos \
    -mat_type aijkokkos
```

## Running GPU Tests

```bash
# Build and run all GPU tests
cmake --build build --target run_gpu_tests

# Or run individual test suites
OMP_PROC_BIND=false ./build/tests/test_gpu_smoke      # 10 tests
OMP_PROC_BIND=false ./build/tests/test_gpu_kernels    # 5 tests
OMP_PROC_BIND=false ./build/tests/test_gpu_solver     # 9 tests
OMP_PROC_BIND=false ./build/tests/test_gpu_ibm        # 9 tests
OMP_PROC_BIND=false ./build/tests/test_gpu_phase4     # 17 tests

# Or use CTest
cd build && ctest -L gpu --output-on-failure
```

## Using Level-Set Kernels

```cpp
#include "gpu/kernels/turbulence_kernels.hpp"

using namespace vfswind::gpu;
using namespace vfswind::gpu::levelset;

// Advect level-set with WENO3
LevelsetKernel::advect(
    levelset, ucont, nvert, aj, dt, domain,
    LevelsetAdvectionKernel::Scheme::WENO3
);

// Reinitialize to signed distance function
int iters = LevelsetKernel::reinitialize(
    levelset, nvert, aj, h, domain, max_iter, tol
);

// Compute two-phase properties
LevelsetKernel::computeProperties(
    levelset, rho, mu, rho1, rho2, mu1, mu2, epsilon, domain
);
```

## Using Turbulence Model Kernels

### LES (Smagorinsky)

```cpp
#include "gpu/kernels/turbulence_kernels.hpp"

using namespace vfswind::gpu;

// Static Smagorinsky model
TurbulenceKernel::computeTurbulentViscosity(
    ucat, nvert, csi, eta, zet, aj, nu_t, domain,
    TurbulenceModel::LES_STATIC_SMAGORINSKY,
    Re_nu, Cs_constant
);

// Dynamic Smagorinsky model
TurbulenceKernel::computeTurbulentViscosity(
    ucat, nvert, csi, eta, zet, aj, nu_t, domain,
    TurbulenceModel::LES_DYNAMIC_SMAGORINSKY,
    Re_nu
);
```

### RANS (k-omega)

```cpp
#include "gpu/kernels/rans_kernel.hpp"

using namespace vfswind::gpu::rans;

// Compute SST blending function F1
KOmegaKernel::computeF1(k, omega, distance, nvert, F1, Re_nu, domain);

// Compute turbulent viscosity (SST model)
KOmegaKernel::computeTurbulentViscosity(
    k, omega, nvert, F1, nu_t, Re_nu,
    RANSModel::SST_MENTER, domain
);

// Compute k-omega RHS (production, dissipation, cross-diffusion)
KOmegaKernel::computeRHS(
    ucat, ucont, k, omega, nu_t, nvert, F1,
    csi, eta, zet, aj, k_rhs, omega_rhs,
    Re_nu, RANSModel::SST_MENTER, domain
);

// Apply wall boundary conditions
KOmegaKernel::applyWallBC(k, omega, distance, nvert, Re_nu, domain);
```
