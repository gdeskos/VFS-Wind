# VFS-Wind GPU Architecture

This document explains the GPU acceleration architecture for VFS-Wind, including the design decisions, tradeoffs, and reasoning behind each choice.

## Table of Contents

1. [Overview](#overview)
2. [Design Constraints](#design-constraints)
3. [Architecture Layers](#architecture-layers)
4. [Why This Architecture?](#why-this-architecture)
5. [Component Details](#component-details)
6. [Data Flow](#data-flow)
7. [Adding New GPU Kernels](#adding-new-gpu-kernels)
8. [Performance Considerations](#performance-considerations)

---

## Overview

VFS-Wind is a Computational Fluid Dynamics (CFD) solver written primarily in C, using PETSc for distributed data structures and linear algebra. The GPU acceleration uses Kokkos for portability across NVIDIA (CUDA), AMD (HIP), and Intel (SYCL) GPUs.

### The Challenge

The core challenge was: **How do we add GPU acceleration to a large C codebase without rewriting everything?**

The solution is a **layered dispatch architecture** that:
- Keeps existing C code intact
- Provides a clean C-callable interface for GPU functions
- Handles all C++/Kokkos complexity in a separate layer
- Allows gradual migration of compute-intensive functions to GPU

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                            │
│                                                                     │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│   │  les.c  │  │  rhs.c  │  │k-omega.c│  │ level.c │  │  ibm.c  │  │
│   │  (LES)  │  │(Conv/   │  │ (RANS)  │  │(Level-  │  │  (IBM)  │  │
│   │         │  │ Visc)   │  │         │  │  Set)   │  │         │  │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  │
│        │            │            │            │            │        │
│        └────────────┴─────┬──────┴────────────┴────────────┘        │
│                           │                                         │
│                           ▼                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    gpu_dispatch.h                            │   │
│   │                  (C Interface Layer)                         │   │
│   │                                                              │   │
│   │  • VFSWind_GPU_IsAvailable()                                │   │
│   │  • VFSWind_GPU_ComputeEddyViscosityLES()                    │   │
│   │  • VFSWind_GPU_ComputeConvection()                          │   │
│   │  • VFSWind_GPU_ComputeKOmegaRHS()                           │   │
│   │  • VFSWind_GPU_AdvectLevelset()                             │   │
│   │  • ...                                                       │   │
│   └─────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DISPATCH LAYER (C++)                           │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   gpu_dispatch.cpp                           │   │
│   │                                                              │   │
│   │  Responsibilities:                                           │   │
│   │  1. Extract domain info from PETSc DMDA                     │   │
│   │  2. Copy data: PETSc Vec → Kokkos View                      │   │
│   │  3. Call appropriate GPU kernel                              │   │
│   │  4. Copy results: Kokkos View → PETSc Vec                   │   │
│   │  5. Handle errors and provide CPU fallback                   │   │
│   └─────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       KERNEL LAYER (C++)                            │
│                                                                     │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│   │ les_kernel.hpp│  │rans_kernel.hpp│  │levelset_      │          │
│   │               │  │               │  │  kernel.hpp   │          │
│   │ • Static      │  │ • Wilcox      │  │ • WENO3/5     │          │
│   │   Smagorinsky │  │ • SST Menter  │  │ • Reinit      │          │
│   │ • Dynamic     │  │ • Wall BC     │  │ • Heaviside   │          │
│   └───────────────┘  └───────────────┘  └───────────────┘          │
│                                                                     │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐          │
│   │convection_    │  │viscous_       │  │ibm_kernels.hpp│          │
│   │  kernel.hpp   │  │  kernel.hpp   │  │               │          │
│   │               │  │               │  │ • Interpolate │          │
│   │ • QUICK       │  │ • Face fluxes │  │ • Force spread│          │
│   │ • Upwind      │  │ • Metrics     │  │ • Delta funcs │          │
│   └───────────────┘  └───────────────┘  └───────────────┘          │
│                             │                                       │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       KOKKOS RUNTIME                                │
│                                                                     │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│   │    CUDA     │  │     HIP     │  │   OpenMP    │                │
│   │  (NVIDIA)   │  │    (AMD)    │  │    (CPU)    │                │
│   └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Design Constraints

Before explaining the architecture, it's important to understand the constraints we faced:

### 1. Legacy C Codebase
- VFS-Wind has ~50,000 lines of C code
- Rewriting in C++ would take years and introduce bugs
- The code is well-tested and validated against experiments

### 2. PETSc Integration
- PETSc manages distributed arrays (DMDA), vectors (Vec), and solvers
- PETSc has its own memory layout (3D arrays with ghost cells)
- We can't replace PETSc—it handles MPI parallelism and linear solvers

### 3. GPU Portability
- Need to support NVIDIA, AMD, and Intel GPUs
- Different systems have different hardware
- Can't lock into a single vendor (CUDA-only)

### 4. Incremental Adoption
- Can't port everything at once
- Need to maintain CPU fallback for debugging
- Some operations (I/O, boundary conditions) stay on CPU

### 5. Performance
- Data transfer between CPU and GPU is expensive
- Need to minimize copies
- Keep data on GPU as long as possible

---

## Architecture Layers

### Layer 1: Application Layer (C)

**Files:** `les.c`, `rhs.c`, `k-omega.c`, `level.c`, `ibm.c`

This is the original VFS-Wind code. Each file contains solver routines that:
- Loop over grid cells
- Compute physical quantities (velocities, stresses, etc.)
- Update PETSc vectors

**Modification pattern:**
```c
void Compute_eddy_viscosity_LES(UserCtx *user)
{
    /* Check if GPU is available and use it */
#ifdef ENABLE_GPU
    if (VFSWind_GPU_IsAvailable()) {
        int err = VFSWind_GPU_ComputeEddyViscosityLES(
            da, fda, user->lUcat, user->lCsi, ...
        );
        if (err == 0) {
            goto wall_function_treatment;  /* Skip CPU path */
        }
        /* GPU failed, fall through to CPU */
    }
#endif

    /* Original CPU implementation (unchanged) */
    for (k = lzs; k < lze; k++) {
        for (j = lys; j < lye; j++) {
            for (i = lxs; i < lxe; i++) {
                /* ... computation ... */
            }
        }
    }

wall_function_treatment:
    /* Post-processing (runs on both paths) */
}
```

**Why this pattern?**
- **Minimal changes:** Only add GPU dispatch at function entry
- **Fallback:** If GPU fails, CPU code runs automatically
- **Conditional compilation:** `#ifdef ENABLE_GPU` means no overhead when GPU is disabled
- **Shared post-processing:** Some operations (like wall functions) run on CPU regardless

### Layer 2: C Interface (gpu_dispatch.h)

**Purpose:** Provide C-callable function declarations for GPU operations.

```c
#ifdef __cplusplus
extern "C" {
#endif

int VFSWind_GPU_IsAvailable(void);
const char* VFSWind_GPU_GetBackendName(void);

int VFSWind_GPU_ComputeEddyViscosityLES(
    DM da, DM fda,
    Vec lUcat, Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj, Vec lNvert, Vec lCs, Vec lNu_t
);

/* ... more functions ... */

#ifdef __cplusplus
}
#endif
```

**Why a separate header?**
- **C compatibility:** The `extern "C"` block ensures C linkage
- **Clean API:** C code only sees simple function declarations
- **No C++ leakage:** No templates, classes, or Kokkos types visible to C
- **Documentation:** Each function is documented with inputs/outputs

### Layer 3: Dispatch Implementation (gpu_dispatch.cpp)

**Purpose:** Bridge between PETSc (C) and Kokkos (C++).

```cpp
int VFSWind_GPU_ComputeEddyViscosityLES(
    DM da, DM fda, Vec lUcat, Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj, Vec lNvert, Vec lCs, Vec lNu_t
) {
    // 1. Check Kokkos is ready
    if (!Kokkos::is_initialized()) return -1;

    // 2. Extract domain info from PETSc
    KernelDomainInfo domain = extractDomainInfo(da);

    // 3. Allocate Kokkos Views
    VectorView3D<> ucat("ucat", domain.mz, domain.my, domain.mx);
    ScalarView3D<> nu_t("nu_t", domain.mz, domain.my, domain.mx);
    // ... more views ...

    // 4. Copy data from PETSc to Kokkos
    copyPetscToKokkos(fda, lUcat, ucat);
    copyPetscToKokkos(fda, lCsi, csi);
    // ... more copies ...

    // 5. Execute GPU kernel
    LESKernel::computeEddyViscosity(
        ucat, csi, eta, zet, aj, nvert, cs, nu_t, domain
    );

    // 6. Copy result back to PETSc
    copyKokkosToPetsc(da, nu_t, lNu_t);

    return 0;  // Success
}
```

**Why this layer?**
- **Encapsulation:** All PETSc↔Kokkos data transfer in one place
- **Error handling:** Can catch errors and return status codes
- **Abstraction:** Kernels don't know about PETSc
- **Testability:** Kernels can be tested independently (see test files)

### Layer 4: GPU Kernels (header-only C++)

**Purpose:** Actual parallel computation using Kokkos.

```cpp
struct LESKernel {
    static void computeEddyViscosity(
        const VectorView3D<>& ucat,
        const VectorView3D<>& csi,
        /* ... */
        ScalarView3D<>& nu_t,
        const KernelDomainInfo& domain
    ) {
        Kokkos::parallel_for(
            "LES_EddyViscosity",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {domain.lzs, domain.lys, domain.lxs},
                {domain.lze, domain.lye, domain.lxe}
            ),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Skip solid cells
                if (nvert(k, j, i) > 0.1) {
                    nu_t(k, j, i) = 0.0;
                    return;
                }

                // Compute strain rate tensor
                double S_mag = computeStrainRateMagnitude(
                    ucat, csi, eta, zet, k, j, i
                );

                // Smagorinsky model: nu_t = (Cs * delta)^2 * |S|
                double delta = pow(1.0 / aj(k, j, i), 1.0/3.0);
                nu_t(k, j, i) = cs(k, j, i) * delta * delta * S_mag;
            }
        );
    }
};
```

**Why header-only?**
- **Inlining:** Kokkos lambdas need to be inlined for GPU compilation
- **Template flexibility:** Can parameterize on memory space, execution space
- **No link issues:** Avoids complex CUDA/HIP compilation rules
- **Single source:** One file works for CPU and GPU

---

## Why This Architecture?

### Alternative 1: Rewrite Everything in CUDA

**Pros:**
- Maximum performance
- Direct GPU memory management

**Cons:**
- Years of work
- NVIDIA lock-in
- Can't use PETSc solvers
- Would lose all existing validation

**Verdict:** Not practical for a mature codebase.

### Alternative 2: OpenACC/OpenMP Offloading

**Pros:**
- Minimal code changes (pragmas)
- Works with existing C code

**Cons:**
- Compiler support is inconsistent
- Limited control over data layout
- Hard to optimize
- PETSc integration is problematic

**Verdict:** Tried this, too fragile for production.

### Alternative 3: PETSc GPU Vectors Only

**Pros:**
- PETSc handles everything
- Minimal code changes

**Cons:**
- Only works for linear algebra (SpMV, etc.)
- Custom kernels (LES, IBM, level-set) can't use this
- Limited to PETSc's supported operations

**Verdict:** Good for solvers, not enough for physics.

### Our Choice: Kokkos Dispatch Layer

**Pros:**
- Portable across GPU vendors
- Clean separation of concerns
- Incremental adoption
- Can still use PETSc for what it's good at
- Testable kernels
- CPU fallback for debugging

**Cons:**
- Data transfer overhead
- Two programming models (C + Kokkos)
- Build complexity (need Kokkos)

**Verdict:** Best balance of portability, performance, and maintainability.

---

## Data Flow

### Memory Spaces

```
┌─────────────────────────────────────────────────────────────────┐
│                         CPU MEMORY                               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    PETSc Vectors                         │   │
│   │                                                          │   │
│   │   Vec lUcat    [mz][my][mx]  (Cartesian velocity)       │   │
│   │   Vec lCsi     [mz][my][mx]  (ξ metric)                 │   │
│   │   Vec lNu_t    [mz][my][mx]  (Eddy viscosity - output)  │   │
│   │   ...                                                    │   │
│   └──────────────────────┬──────────────────────────────────┘   │
│                          │                                       │
│                          │  copyPetscToKokkos()                 │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Kokkos Host Mirror                      │   │
│   │                                                          │   │
│   │   ScalarView3D<HostSpace> ucat_h(...)                   │   │
│   └──────────────────────┬──────────────────────────────────┘   │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
                           │  Kokkos::deep_copy()
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GPU MEMORY                               │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Kokkos Device Views                     │   │
│   │                                                          │   │
│   │   ScalarView3D<DeviceSpace> ucat(...)                   │   │
│   │   ScalarView3D<DeviceSpace> nu_t(...)                   │   │
│   └──────────────────────┬──────────────────────────────────┘   │
│                          │                                       │
│                          │  Kokkos::parallel_for()              │
│                          ▼                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    GPU Kernels                           │   │
│   │                                                          │   │
│   │   Each thread computes one cell (i, j, k)               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Typical Function Flow

```
1. Application calls VFSWind_GPU_ComputeEddyViscosityLES()
2. Dispatch layer:
   a. Checks Kokkos::is_initialized()
   b. Extracts domain bounds from DMDA
   c. Allocates device Views
   d. Copies input data to device
   e. Launches kernel
   f. Copies output data back to host
   g. Returns success/failure code
3. Application continues (or falls back to CPU on failure)
```

---

## Adding New GPU Kernels

### Step 1: Create Kernel Header

Create `Source/gpu/kernels/my_kernel.hpp`:

```cpp
#ifndef VFSWIND_MY_KERNEL_HPP
#define VFSWIND_MY_KERNEL_HPP

#include "petsc_kokkos.hpp"

namespace vfswind {
namespace gpu {

struct MyKernel {
    static void execute(
        const ScalarView3D<>& input,
        ScalarView3D<>& output,
        const KernelDomainInfo& domain
    ) {
        Kokkos::parallel_for(
            "MyKernel",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {domain.lzs, domain.lys, domain.lxs},
                {domain.lze, domain.lye, domain.lxe}
            ),
            KOKKOS_LAMBDA(int k, int j, int i) {
                output(k, j, i) = /* computation */;
            }
        );
    }
};

} // namespace gpu
} // namespace vfswind

#endif
```

### Step 2: Add C Interface

In `gpu_dispatch.h`:

```c
int VFSWind_GPU_MyFunction(
    DM da,
    Vec input,
    Vec output
);
```

### Step 3: Implement Dispatch

In `gpu_dispatch.cpp`:

```cpp
int VFSWind_GPU_MyFunction(DM da, Vec input, Vec output) {
    if (!Kokkos::is_initialized()) return -1;

    KernelDomainInfo domain = extractDomainInfo(da);

    ScalarView3D<> d_input("input", domain.mz, domain.my, domain.mx);
    ScalarView3D<> d_output("output", domain.mz, domain.my, domain.mx);

    copyPetscToKokkos(da, input, d_input);

    MyKernel::execute(d_input, d_output, domain);

    copyKokkosToPetsc(da, d_output, output);

    return 0;
}
```

### Step 4: Call from C Code

In your solver file:

```c
#ifdef ENABLE_GPU
    if (VFSWind_GPU_IsAvailable()) {
        int err = VFSWind_GPU_MyFunction(da, input, output);
        if (err == 0) return;  /* GPU succeeded */
    }
#endif
    /* CPU fallback */
```

### Step 5: Add Tests

Create `tests/gpu/test_my_kernel.cpp`:

```cpp
TEST(MyKernelTest, BasicComputation) {
    // Allocate Views
    ScalarView3D<> input("input", 10, 10, 10);
    ScalarView3D<> output("output", 10, 10, 10);

    // Initialize input
    // ...

    // Run kernel
    MyKernel::execute(input, output, domain);

    // Verify output
    // ...
}
```

---

## Performance Considerations

### Data Transfer Overhead

The biggest performance concern is CPU↔GPU data transfer:

```
Problem: Each function call copies data to GPU and back

Solution approaches:
1. Batch operations (copy once, run multiple kernels)
2. Keep data on GPU across timesteps
3. Use GPU-resident PETSc vectors (-vec_type cuda)
```

### Current Status

Currently, each dispatch function does its own data transfer. This is simple but not optimal. Future optimizations could:

1. **Persistent GPU buffers:** Allocate once, reuse across timesteps
2. **Async transfers:** Overlap computation and communication
3. **Fused kernels:** Combine multiple operations into one kernel

### When to Use GPU

GPU acceleration is beneficial when:
- Grid size is large (> 100³)
- Operation is compute-bound (not memory-bound)
- Data is already on GPU (or can stay there)

GPU may be slower when:
- Grid is small (kernel launch overhead dominates)
- Operation is simple (bandwidth-limited)
- Heavy data transfer required

### Profiling

Use these tools to analyze performance:

```bash
# NVIDIA
nsys profile mpirun -np 1 ./vwis -xml control.xml
nvprof --print-gpu-trace ./vwis ...

# AMD
rocprof --stats ./vwis ...

# General
# Add to control.xml: <gpu_timing>1</gpu_timing>
```

---

## Summary

The VFS-Wind GPU architecture balances several competing concerns:

| Concern | Solution |
|---------|----------|
| Legacy C code | Dispatch layer with C interface |
| GPU portability | Kokkos abstraction |
| PETSc integration | Data copy helpers |
| Incremental adoption | Per-function GPU dispatch |
| Debugging | CPU fallback |
| Testing | Header-only kernels, unit tests |

This architecture allows us to:
1. Keep the validated C codebase
2. Add GPU acceleration incrementally
3. Support multiple GPU vendors
4. Maintain testability and debuggability
5. Optimize performance over time

The tradeoff is some data transfer overhead, which can be mitigated in future optimizations by keeping data resident on GPU and batching operations.
