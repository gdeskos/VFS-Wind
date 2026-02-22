# VFS-Wind GPU Porting Plan

## Executive Summary

This document outlines a comprehensive plan to refactor VFS-Wind for GPU portability. The goal is to enable the code to run efficiently on GPU-accelerated systems while maintaining:
- Full MPI parallelism for multi-node execution
- CPU fallback capability for systems without GPUs
- Code maintainability through portable abstractions
- Minimal disruption to existing functionality

**Target GPU platforms**: NVIDIA (CUDA), AMD (HIP/ROCm), Intel (SYCL/oneAPI)

**Estimated effort**: 12-18 months for full implementation
**Expected speedup**: 5-20x for compute-intensive kernels (problem-size dependent)

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [GPU Porting Strategy Options](#2-gpu-porting-strategy-options)
3. [Recommended Approach: Kokkos](#3-recommended-approach-kokkos)
4. [Phased Implementation Plan](#4-phased-implementation-plan)
5. [Code Refactoring Requirements](#5-code-refactoring-requirements)
6. [Data Structure Modifications](#6-data-structure-modifications)
7. [Memory Management Strategy](#7-memory-management-strategy)
8. [PETSc GPU Integration](#8-petsc-gpu-integration)
9. [Testing and Validation](#9-testing-and-validation)
10. [Performance Considerations](#10-performance-considerations)
11. [Risk Assessment](#11-risk-assessment)
12. [Timeline and Milestones](#12-timeline-and-milestones)

---

## 1. Current Architecture Analysis

### 1.1 Code Structure Overview

```
VFS-Wind/
├── Source/
│   ├── main.c              # Main driver, time-stepping loop
│   ├── rhs.c               # RHS computation (Convection, Viscous) [HIGH PRIORITY]
│   ├── momentum.c          # Pressure gradient, CFL calculation
│   ├── poisson.c           # Poisson solver (multigrid)
│   ├── poisson_hypre.c     # HYPRE-based Poisson solver
│   ├── solvers.c           # Solver orchestration, FSI coupling
│   ├── implicitsolver.c    # Implicit momentum solver (SNES)
│   ├── ibm.c               # Immersed Boundary Method
│   ├── fsi.c               # Fluid-Structure Interaction
│   ├── les.c               # Large Eddy Simulation models
│   ├── k-omega.c           # RANS turbulence model
│   ├── level.c             # Level-set method (two-phase)
│   ├── variables.h         # Core data structures
│   └── ...
```

### 1.2 Computational Hotspots (Profiling Priority)

| Kernel | File | Est. Runtime % | GPU Priority |
|--------|------|----------------|--------------|
| Convection (QUICK scheme) | rhs.c | 25-35% | HIGH |
| Viscous fluxes | rhs.c | 20-30% | HIGH |
| Poisson solver | poisson.c | 15-25% | HIGH |
| Pressure gradient | momentum.c | 5-10% | MEDIUM |
| IBM interpolation | ibm.c | 5-15% | MEDIUM |
| Level-set advection | level.c | 5-10% | MEDIUM |
| LES/RANS turbulence | les.c, k-omega.c | 3-8% | LOW |

### 1.3 Current Parallelization

- **MPI**: Domain decomposition via PETSc DMDA
- **Data distribution**: Each rank owns local subdomain with ghost layers
- **Communication**: `DMGlobalToLocal`, `DMLocalToGlobal` for halo exchange
- **Linear solvers**: PETSc KSP/SNES with HYPRE backend option

### 1.4 Key Data Structures

```c
// Core vector component (variables.h)
typedef struct {
    PetscScalar x, y, z;
} Cmpnts;

// Main simulation context (simplified)
typedef struct {
    DM da, fda;                    // PETSc distributed arrays
    Vec Ucont, Ucat, P;            // Global vectors
    Vec lUcont, lUcat, lP;         // Local vectors (with ghosts)
    Vec Csi, Eta, Zet, Aj;         // Grid metrics
    PetscInt IM, JM, KM;           // Global dimensions
    IBMNodes *ibm;                 // Immersed boundary data
} UserCtx;
```

---

## 2. GPU Porting Strategy Options

### 2.1 Option A: Native CUDA/HIP

**Pros:**
- Maximum performance potential
- Direct hardware control
- Mature ecosystem

**Cons:**
- Vendor lock-in (separate CUDA and HIP codepaths)
- Significant code duplication
- Higher maintenance burden

**Recommendation**: Not recommended as primary approach

### 2.2 Option B: OpenACC/OpenMP Target

**Pros:**
- Directive-based (minimal code changes)
- Portable across vendors (in theory)
- Gradual adoption possible

**Cons:**
- Compiler-dependent performance
- Limited control over memory management
- Complex data structures may not port well

**Recommendation**: Suitable for quick prototyping, not production

### 2.3 Option C: Kokkos (Recommended)

**Pros:**
- Performance portability (CUDA, HIP, SYCL, OpenMP)
- C++ abstraction with minimal overhead
- Active development (Sandia National Labs)
- PETSc integration available
- Proven in large-scale CFD codes (Trilinos, LAMMPS)

**Cons:**
- Requires C++ (current code is C)
- Learning curve for team
- Some refactoring needed

**Recommendation**: PRIMARY CHOICE for VFS-Wind

### 2.4 Option D: SYCL/DPC++

**Pros:**
- ISO C++ standard-based
- Intel-backed with growing AMD/NVIDIA support
- Modern C++ features

**Cons:**
- Less mature than CUDA
- Compiler support varies
- Smaller community

**Recommendation**: Consider as secondary option

---

## 3. Recommended Approach: Kokkos

### 3.1 Why Kokkos?

1. **Performance Portability**: Single source compiles to CUDA, HIP, SYCL, OpenMP
2. **Memory Abstraction**: Unified memory model with explicit control
3. **PETSc Integration**: `PetscKokkos` backend available since PETSc 3.16
4. **Community**: Used by DOE labs, wind energy codes (Nalu-Wind)
5. **Incremental Adoption**: Can coexist with existing C code

### 3.2 Kokkos Execution Model

```cpp
// Parallel loop example
Kokkos::parallel_for("Convection",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
    KOKKOS_LAMBDA(int k, int j, int i) {
        // Convection kernel
        fp1(k, j, i, 0) = compute_flux(ucont, k, j, i);
    }
);
```

### 3.3 Memory Model

```cpp
// Device view (GPU memory)
Kokkos::View<Cmpnts***, Kokkos::CudaSpace> d_ucont("ucont", nz, ny, nx);

// Host view (CPU memory)
Kokkos::View<Cmpnts***, Kokkos::HostSpace> h_ucont("ucont_host", nz, ny, nx);

// Mirror view (automatic sync)
auto h_ucont_mirror = Kokkos::create_mirror_view(d_ucont);

// Copy: host -> device
Kokkos::deep_copy(d_ucont, h_ucont_mirror);
```

---

## 4. Phased Implementation Plan

### Phase 0: Infrastructure Setup (Months 1-2)

**Goals:**
- Set up Kokkos build infrastructure
- Create GPU-aware CMake configuration
- Establish testing framework for GPU builds

**Tasks:**
- [ ] Add Kokkos as CMake dependency (FetchContent or find_package)
- [ ] Create `ENABLE_GPU` CMake option
- [ ] Set up CI with GPU runners (optional)
- [ ] Create Kokkos initialization/finalization in main.c
- [ ] Verify PETSc GPU backend compatibility

**Deliverables:**
- CMakeLists.txt with Kokkos support
- GPU build documentation
- Basic smoke test on GPU

### Phase 1: Core Kernels (Months 3-6)

**Goals:**
- Port compute-intensive kernels to Kokkos
- Maintain CPU fallback
- Validate correctness

**Priority kernels:**
1. **Convection kernel** (rhs.c: ~320 lines)
2. **Viscous kernel** (rhs.c: ~500 lines)
3. **Pressure gradient** (momentum.c: ~200 lines)
4. **RungeKutta time integration** (rhs.c: ~120 lines)

**Tasks:**
- [ ] Create `kernels/` directory for GPU code
- [ ] Implement `convection_kernel.cpp`
- [ ] Implement `viscous_kernel.cpp`
- [ ] Implement `pressure_gradient_kernel.cpp`
- [ ] Create Kokkos Views wrapper for PETSc vectors
- [ ] Add memory transfer utilities (host <-> device)
- [ ] Validate against CPU results (regression tests)

**Deliverables:**
- GPU-accelerated RHS computation
- Performance benchmarks vs CPU
- Unit tests for each kernel

### Phase 2: Linear Solvers (Months 7-9)

**Goals:**
- Enable GPU-accelerated Poisson solver
- Leverage PETSc/HYPRE GPU backends

**Tasks:**
- [ ] Enable PETSc CUDA/Kokkos backend via runtime options
- [ ] Configure HYPRE with GPU support (optional)
- [ ] Profile solver performance on GPU
- [ ] Implement custom preconditioner if needed
- [ ] Optimize matrix assembly for GPU

**Deliverables:**
- GPU-accelerated Poisson solver
- Solver benchmarks (iterations, time)
- Configuration documentation

### Phase 3: IBM and FSI (Months 10-12)

**Goals:**
- Port Immersed Boundary Method to GPU
- Handle irregular memory access patterns

**Tasks:**
- [ ] Analyze IBM interpolation access patterns
- [ ] Implement GPU-friendly data structures for IBM
- [ ] Port `ibm_interpolation` kernel
- [ ] Port force calculation kernels
- [ ] Optimize FSI coupling for GPU
- [ ] Handle dynamic body motion updates

**Deliverables:**
- GPU-accelerated IBM
- FSI validation tests
- Performance report

### Phase 4: Level-Set and Turbulence (Months 13-15)

**Goals:**
- Port remaining physics modules
- Achieve full GPU execution

**Tasks:**
- [ ] Port level-set advection (WENO schemes)
- [ ] Port level-set reinitialization
- [ ] Port LES models (Smagorinsky, dynamic)
- [ ] Port RANS models (k-omega SST)
- [ ] Implement GPU-aware I/O (optional)

**Deliverables:**
- Complete GPU physics modules
- Full application benchmarks
- Performance optimization report

### Phase 5: Optimization and Production (Months 16-18)

**Goals:**
- Performance optimization
- Production hardening
- Documentation

**Tasks:**
- [ ] Profile full application on target GPUs
- [ ] Optimize memory transfers (overlap computation/communication)
- [ ] Implement GPU-aware MPI (if available)
- [ ] Tune kernel launch configurations
- [ ] Write user documentation
- [ ] Create example GPU run scripts

**Deliverables:**
- Optimized GPU build
- User documentation
- Performance tuning guide
- Example cases with GPU benchmarks

---

## 5. Code Refactoring Requirements

### 5.1 Language Migration (C to C++)

**Current state**: Pure C with PETSc
**Target state**: C++ with Kokkos, C-linkage for PETSc

**Approach:**
1. Rename `.c` files to `.cpp` incrementally
2. Add `extern "C"` wrappers for PETSc callbacks
3. Use C++ features sparingly (templates, lambdas for kernels)

**Example wrapper:**
```cpp
// rhs.cpp
extern "C" {
    #include "petsc.h"
    #include "variables.h"
}

#include <Kokkos_Core.hpp>

// Kokkos kernel
void ConvectionGPU(UserCtx* user) {
    // ... Kokkos implementation
}

// C-callable wrapper
extern "C" PetscErrorCode Convection(UserCtx* user) {
    #ifdef ENABLE_GPU
        ConvectionGPU(user);
    #else
        ConvectionCPU(user);  // Original code
    #endif
    return 0;
}
```

### 5.2 Loop Refactoring Pattern

**Original (rhs.c):**
```c
for (k = lzs; k < lze; k++) {
    for (j = lys; j < lye; j++) {
        for (i = lxs; i < lxe; i++) {
            fp1[k][j][i].x = /* QUICK scheme */;
        }
    }
}
```

**Kokkos version:**
```cpp
auto d_fp1 = getDeviceView(fp1_vec);
auto d_ucont = getDeviceView(ucont_vec);

Kokkos::parallel_for("Convection_I",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
    KOKKOS_LAMBDA(int k, int j, int i) {
        d_fp1(k, j, i, 0) = /* QUICK scheme with d_ucont */;
    }
);
```

### 5.3 Conditional/Branching Handling

**Original (with IBM checks):**
```c
if ((int)(nvert[k][j][i] + 0.5) < 1) {
    // Interior point
} else {
    // IBM point - skip or special handling
}
```

**GPU-friendly version:**
```cpp
KOKKOS_LAMBDA(int k, int j, int i) {
    // Use mask or predicated execution
    const bool is_interior = (static_cast<int>(d_nvert(k,j,i) + 0.5) < 1);
    if (is_interior) {
        d_fp1(k, j, i, 0) = compute_flux(...);
    }
    // Warp divergence acceptable for small branch ratio
}
```

---

## 6. Data Structure Modifications

### 6.1 Extended UserCtx for GPU

```cpp
// New GPU-aware context (variables_gpu.hpp)
struct UserCtxGPU {
    // Original CPU context
    UserCtx* cpu;

    // Kokkos Views (device memory)
    Kokkos::View<Cmpnts***, DeviceSpace> d_ucont;
    Kokkos::View<Cmpnts***, DeviceSpace> d_ucat;
    Kokkos::View<PetscReal***, DeviceSpace> d_p;
    Kokkos::View<PetscReal***, DeviceSpace> d_nvert;

    // Grid metrics on device
    Kokkos::View<Cmpnts***, DeviceSpace> d_csi, d_eta, d_zet;
    Kokkos::View<PetscReal***, DeviceSpace> d_aj;

    // RHS temporary arrays
    Kokkos::View<Cmpnts***, DeviceSpace> d_fp1, d_fp2;

    // Domain info
    int xs, xe, ys, ye, zs, ze;      // Local range
    int gxs, gxe, gys, gye, gzs, gze; // Ghost range

    // Synchronization
    bool device_data_valid;
    bool host_data_valid;

    // Methods
    void syncToDevice();
    void syncToHost();
    void allocateDeviceMemory();
    void freeDeviceMemory();
};
```

### 6.2 IBM Data Structures for GPU

```cpp
// GPU-friendly IBM storage
struct IBMNodesGPU {
    // Surface points (Structure of Arrays for coalesced access)
    Kokkos::View<PetscReal*, DeviceSpace> x_bp, y_bp, z_bp;

    // Element connectivity
    Kokkos::View<int*, DeviceSpace> nv1, nv2, nv3;

    // Interpolation data (sorted by cell ownership)
    Kokkos::View<int*, DeviceSpace> cell_i, cell_j, cell_k;
    Kokkos::View<PetscReal*, DeviceSpace> interp_weights;

    // Force accumulators
    Kokkos::View<PetscReal*, DeviceSpace> F_lagr_x, F_lagr_y, F_lagr_z;

    int n_elmt;      // Number of elements
    int n_vert;      // Number of vertices
};
```

---

## 7. Memory Management Strategy

### 7.1 Memory Pools

```cpp
// Pre-allocate memory pools to avoid runtime allocation
class GPUMemoryPool {
public:
    void initialize(int nx, int ny, int nz, int num_fields);

    Kokkos::View<Cmpnts***> getVectorField(int id);
    Kokkos::View<PetscReal***> getScalarField(int id);

private:
    std::vector<Kokkos::View<Cmpnts***>> vector_fields;
    std::vector<Kokkos::View<PetscReal***>> scalar_fields;
};
```

### 7.2 Host-Device Transfer Optimization

```cpp
// Minimize transfers with dirty flags
class FieldManager {
    enum Location { HOST, DEVICE, BOTH };

    void markDirty(Location loc);
    void ensureOnDevice();
    void ensureOnHost();

    // Overlap transfer with computation
    void asyncTransferToDevice(cudaStream_t stream);
    void asyncTransferToHost(cudaStream_t stream);
};
```

### 7.3 PETSc Vector Integration

```cpp
// Wrapper to get Kokkos View from PETSc Vec
template<typename Space>
Kokkos::View<Cmpnts***, Space>
getPetscVecAsView(Vec v, DM da, int with_ghosts) {
    #ifdef PETSC_HAVE_KOKKOS
        // Use PETSc's native Kokkos support
        return PetscVecGetKokkosView<Space>(v);
    #else
        // Manual copy
        Cmpnts*** array;
        DMDAVecGetArray(da, v, &array);
        // Create View and copy
        ...
    #endif
}
```

---

## 8. PETSc GPU Integration

### 8.1 Enabling PETSc GPU Backend

**Configuration options:**
```bash
# Build PETSc with Kokkos
./configure --with-kokkos-dir=/path/to/kokkos \
            --with-cuda \
            --with-cuda-arch=80 \  # A100
            --download-hypre \
            --with-hypre-gpu
```

**Runtime options:**
```bash
# Run with GPU vectors/matrices
mpirun -np 4 ./vwis -vec_type kokkos \
                    -mat_type aijkokkos \
                    -dm_vec_type kokkos \
                    -dm_mat_type aijkokkos
```

### 8.2 HYPRE GPU Solver

```bash
# Enable HYPRE GPU in runtime
-pc_hypre_boomeramg_coarsen_type PMIS \
-pc_hypre_boomeramg_interp_type ext+i \
-pc_hypre_boomeramg_relax_type_all l1scaled-SOR/Jacobi \
-pc_hypre_boomeramg_device_level 1
```

---

## 9. Testing and Validation

### 9.1 Unit Tests (per kernel)

```cpp
// tests/gpu/test_convection_kernel.cpp
TEST(ConvectionKernel, UniformFlow) {
    // Setup uniform flow field
    auto ucont = createUniformField(1.0, 0.0, 0.0);

    // Run kernel
    ConvectionGPU(ucont, fp1);

    // Convection of uniform flow should be zero
    EXPECT_NEAR(maxAbs(fp1), 0.0, 1e-12);
}

TEST(ConvectionKernel, MatchesCPU) {
    // Random field
    auto ucont = createRandomField();

    // Run both versions
    ConvectionCPU(ucont, fp1_cpu);
    ConvectionGPU(ucont, fp1_gpu);

    // Compare
    EXPECT_NEAR(relativeError(fp1_cpu, fp1_gpu), 0.0, 1e-10);
}
```

### 9.2 Integration Tests

```cpp
// tests/gpu/test_timestep.cpp
TEST(Integration, SingleTimestep) {
    // Initialize from restart file
    UserCtx user = loadRestart("test_restart.dat");

    // Run one timestep on CPU
    runTimestep(&user, /*gpu=*/false);
    auto result_cpu = copyFields(&user);

    // Reset and run on GPU
    user = loadRestart("test_restart.dat");
    runTimestep(&user, /*gpu=*/true);
    auto result_gpu = copyFields(&user);

    // Compare
    EXPECT_NEAR(fieldError(result_cpu, result_gpu), 0.0, 1e-8);
}
```

### 9.3 Regression Tests

- Run existing test cases with GPU enabled
- Compare `Kinetic_Energy.dat`, `Converge_dU` outputs
- Tolerance: 0.1% relative difference (allows for FP ordering differences)

---

## 10. Performance Considerations

### 10.1 Expected Speedups

| Component | CPU (1 core) | GPU (A100) | Speedup |
|-----------|--------------|------------|---------|
| Convection | 1.0x | 15-25x | 15-25x |
| Viscous | 1.0x | 10-20x | 10-20x |
| Poisson | 1.0x | 5-10x | 5-10x |
| IBM | 1.0x | 3-8x | 3-8x |
| **Overall** | 1.0x | **8-15x** | 8-15x |

*Note: Speedups are highly problem-size dependent. Small problems may see less benefit due to kernel launch overhead.*

### 10.2 Memory Bandwidth Analysis

**Key metric**: Arithmetic Intensity (AI) = FLOPs / Bytes transferred

| Kernel | AI (FLOPs/Byte) | Classification |
|--------|-----------------|----------------|
| Convection | 0.5-0.8 | Memory-bound |
| Viscous | 0.3-0.5 | Memory-bound |
| Poisson SpMV | 0.1-0.2 | Memory-bound |
| Vector ops | 0.01-0.1 | Memory-bound |

**Implication**: Performance limited by GPU memory bandwidth (A100: 2 TB/s, H100: 3.35 TB/s)

### 10.3 Optimization Strategies

1. **Kernel fusion**: Combine Convection + Viscous + Pressure gradient
2. **Shared memory**: Cache stencil data in shared memory
3. **Coalesced access**: Ensure contiguous memory reads
4. **Occupancy tuning**: Optimize thread block sizes
5. **Overlap**: Compute/communication overlap with streams

---

## 11. Risk Assessment

### 11.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PETSc GPU backend issues | Medium | High | Fallback to manual transfers |
| Kokkos learning curve | Low | Medium | Training, example codes |
| IBM irregular patterns | High | Medium | Accept lower speedup for IBM |
| MPI+GPU scaling | Medium | High | GPU-aware MPI, careful profiling |
| Numerical differences | Low | Medium | Validation tests, tolerances |

### 11.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Underestimated effort | Medium | High | Phase-based approach, buffer time |
| Hardware availability | Low | High | Cloud GPU instances |
| Compiler issues | Low | Medium | Multiple compiler testing |

---

## 12. Timeline and Milestones

```
Month 1-2:   [=====] Phase 0: Infrastructure
Month 3-6:   [===========] Phase 1: Core Kernels
Month 7-9:   [=======] Phase 2: Linear Solvers
Month 10-12: [=======] Phase 3: IBM/FSI
Month 13-15: [=======] Phase 4: Level-Set/Turbulence
Month 16-18: [=======] Phase 5: Optimization

Key Milestones:
  M1 (Month 2):  GPU build working, smoke test passing
  M2 (Month 6):  RHS kernels on GPU, 10x speedup target
  M3 (Month 9):  Full solver on GPU
  M4 (Month 12): IBM on GPU
  M5 (Month 15): All physics on GPU
  M6 (Month 18): Production-ready release
```

---

## Appendix A: Build System Changes

### CMakeLists.txt additions

```cmake
# GPU configuration
option(ENABLE_GPU "Enable GPU acceleration via Kokkos" OFF)
option(KOKKOS_BACKEND "Kokkos backend: CUDA, HIP, SYCL, OpenMP" "CUDA")

if(ENABLE_GPU)
    find_package(Kokkos REQUIRED)

    # Compile GPU sources as C++
    set_source_files_properties(
        Source/rhs.cpp
        Source/momentum.cpp
        Source/convection_kernel.cpp
        PROPERTIES LANGUAGE CXX
    )

    target_link_libraries(vwis PRIVATE Kokkos::kokkos)
    target_compile_definitions(vwis PRIVATE ENABLE_GPU)
endif()
```

---

## Appendix B: Example Kernel Implementation

### convection_kernel.cpp

```cpp
#include <Kokkos_Core.hpp>
#include "vfswind_gpu.hpp"

void ConvectionKernel_I(
    Kokkos::View<Cmpnts***, DeviceSpace> ucont,
    Kokkos::View<Cmpnts***, DeviceSpace> ucat,
    Kokkos::View<Cmpnts***, DeviceSpace> fp1,
    Kokkos::View<PetscReal***, DeviceSpace> nvert,
    const DomainInfo& domain)
{
    const int lxs = domain.lxs, lxe = domain.lxe;
    const int lys = domain.lys, lye = domain.lye;
    const int lzs = domain.lzs, lze = domain.lze;

    Kokkos::parallel_for("Convection_I",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs-1}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Skip solid points
            if (static_cast<int>(nvert(k,j,i) + 0.5) >= 1 ||
                static_cast<int>(nvert(k,j,i+1) + 0.5) >= 1) {
                return;
            }

            // QUICK scheme for i-direction flux
            PetscReal uc = 0.5 * (ucont(k,j,i).x + ucont(k,j,i+1).x);
            PetscReal um = PetscMin(uc, 0.0);
            PetscReal up = PetscMax(uc, 0.0);

            // Upwind interpolation
            fp1(k,j,i).x = um * quick_interp(ucat, k, j, i, 0, +1) +
                           up * quick_interp(ucat, k, j, i, 0, -1);
            fp1(k,j,i).y = um * quick_interp(ucat, k, j, i, 1, +1) +
                           up * quick_interp(ucat, k, j, i, 1, -1);
            fp1(k,j,i).z = um * quick_interp(ucat, k, j, i, 2, +1) +
                           up * quick_interp(ucat, k, j, i, 2, -1);
        }
    );
}
```

---

## Appendix C: AI-Assisted Implementation Timeline

This section provides a realistic assessment of implementing the GPU port with AI coding assistance (e.g., Claude) combined with human expertise.

### C.1 What AI Can Do Efficiently

| Phase | Task | AI Coding Time | Notes |
|-------|------|----------------|-------|
| Phase 0 | CMake/Kokkos infrastructure | 2-4 hours | Requires human build verification |
| Phase 1 | Convert kernels to Kokkos | 1-2 days | ~10-15 kernels, mechanical transformation |
| Phase 2 | PETSc GPU configuration | 1-2 hours | Mostly configuration and options |
| Phase 3 | IBM/FSI conversion | 1-2 days | Complex, needs iterative debugging |
| Phase 4 | Level-set/turbulence | 1 day | Similar pattern to Phase 1 |

**Total AI coding time: ~1-2 weeks of interactive sessions**

### C.2 What Requires Human Expertise + Hardware

The following tasks cannot be performed by AI and require actual GPU hardware:

- **Compile and run on actual GPU** - AI cannot verify the code executes correctly
- **Debug runtime GPU errors** - Memory issues, race conditions, kernel launch failures
- **Performance profiling** - Identifying bottlenecks requires actual execution and profiling tools (nsys, nvprof, rocprof)
- **Kernel tuning** - Block sizes, occupancy optimization, shared memory usage needs real hardware experimentation
- **Integration testing** - Running full simulations and validating physical results
- **Multi-GPU scaling** - MPI+GPU communication optimization

### C.3 Realistic Combined Timeline

| Phase | AI Coding | Human Testing & Debug | Total Elapsed |
|-------|-----------|----------------------|---------------|
| Phase 0: Infrastructure | 1 day | 1-2 days | ~3 days |
| Phase 1: Core kernels | 2 days | 1-2 weeks | ~2 weeks |
| Phase 2: Linear solvers | 1 day | 1 week | ~1.5 weeks |
| Phase 3: IBM/FSI | 2 days | 2-3 weeks | ~3 weeks |
| Phase 4: Level-set/Turbulence | 1 day | 1-2 weeks | ~2 weeks |
| Phase 5: Optimization | - | 2-4 weeks | ~4 weeks |
| **Total** | **~1-2 weeks** | **~2-3 months** | **~3-4 months** |

### C.4 Recommended Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI-Assisted Development Cycle                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │   AI     │────▶│  Human   │────▶│  Human   │               │
│   │  Writes  │     │  Builds  │     │  Tests   │               │
│   │   Code   │     │  & Links │     │  on GPU  │               │
│   └──────────┘     └──────────┘     └──────────┘               │
│        │                                  │                     │
│        │           ┌──────────┐           │                     │
│        └───────────│   AI     │◀──────────┘                     │
│                    │  Fixes   │                                 │
│                    │  Issues  │                                 │
│                    └──────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Optimal approach:**
1. AI generates initial GPU kernel code
2. Human compiles and runs on GPU hardware
3. Human reports errors/performance issues back to AI
4. AI proposes fixes or optimizations
5. Iterate until kernel is correct and performant

### C.5 Accelerating the Timeline

To minimize the 3-4 month timeline:

1. **Parallel workstreams**: While Phase 1 kernels are being debugged, AI can start Phase 2 code
2. **Continuous integration**: Set up GPU CI runners for automated testing
3. **Reference implementations**: Use Nalu-Wind or similar codes as working examples
4. **Incremental validation**: Test each kernel individually before integration
5. **Performance baselines**: Establish CPU baselines early for comparison

### C.6 Cost-Benefit Summary

| Approach | Timeline | Human Effort | Risk |
|----------|----------|--------------|------|
| Traditional (no AI) | 12-18 months | Very High | Medium |
| AI-assisted | 3-4 months | Medium | Low-Medium |
| AI-assisted + experienced GPU dev | 2-3 months | Medium | Low |

**Conclusion**: AI assistance can reduce the implementation timeline by **~70-80%** compared to traditional development, but human expertise with actual GPU hardware remains essential for testing, debugging, and optimization.

---

## Appendix D: References

1. Kokkos Documentation: https://kokkos.github.io/kokkos-core-wiki/
2. PETSc GPU Support: https://petsc.org/release/manual/gpu/
3. HYPRE GPU: https://github.com/hypre-space/hypre
4. Nalu-Wind (GPU CFD example): https://github.com/Exawind/nalu-wind
5. Performance Portability Best Practices: https://doi.org/10.1177/10943420211045422

---

*Document Version: 1.1*
*Date: February 2026*
*Author: VFS-Wind Development Team*
