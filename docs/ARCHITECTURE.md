# VFS-Wind Architecture Overview

This document provides a high-level overview of the VFS-Wind codebase structure,
data flow, and module interactions to help developers understand how the code works.

---

## 1. What is VFS-Wind?

VFS-Wind (Virtual Flow Simulator - Wind) is a **parallel CFD solver** for incompressible
flows on structured curvilinear grids. It is designed for wind energy research and
supports:

- Large Eddy Simulation (LES) and RANS turbulence modeling
- Two-phase flows (level-set method)
- Immersed Boundary Methods (IBM) for complex geometries
- Fluid-Structure Interaction (FSI)
- Wind turbine rotor modeling (actuator disk/line)
- Wall functions for wall-bounded flows

---

## 2. Directory Structure

```
VFS-Wind/
├── Source/                    # All source code (~65K lines)
│   ├── main.c                 # Entry point, time-stepping loop
│   ├── variables.h            # All data structure definitions
│   ├── [physics modules].c    # Momentum, pressure, turbulence, etc.
│   ├── [I/O modules].c        # Binary output, VTK, XML parsing
│   └── tinyxml2/              # XML parsing library
├── examples/                  # Test cases (sloshing, turbines, channels)
├── Documentation/             # User manual (PDF)
├── CMakeLists.txt             # Build configuration
└── README.md                  # Quick start guide
```

---

## 3. High-Level Program Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ PETSc Init   │ → │ Read Config  │ → │ Create Grid  │            │
│  │ MPI Setup    │   │ control.xml  │   │ DMDA Setup   │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│         │                  │                  │                     │
│         ▼                  ▼                  ▼                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │ Allocate     │ ← │ Read IBM     │ ← │ Setup        │            │
│  │ Vectors      │   │ Surfaces     │   │ Solvers      │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MAIN TIME LOOP                                  │
│   for (ti = tistart; ti < totalsteps; ti++)                         │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ 1. STRUCTURE SOLVER (if FSI enabled)                        │   │
│   │    - Compute forces on immersed bodies                      │   │
│   │    - Update body position/velocity (rigid body dynamics)    │   │
│   │    - Move IBM surface mesh                                  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ 2. FLOW SOLVER (strong coupling iterations)                 │   │
│   │    ┌──────────────────────────────────────────────────────┐ │   │
│   │    │ a) Compute RHS (convection + viscous + sources)      │ │   │
│   │    │ b) Time advance momentum (RK, implicit, or AM2)      │ │   │
│   │    │ c) Solve pressure Poisson equation                   │ │   │
│   │    │ d) Project velocity (correct with pressure gradient) │ │   │
│   │    │ e) Apply IBM forcing (if immersed bodies present)    │ │   │
│   │    │ f) Update turbulence model (LES or RANS)             │ │   │
│   │    └──────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ 3. OUTPUT & DIAGNOSTICS                                     │   │
│   │    - Write velocity/pressure fields (binary or VTK)         │   │
│   │    - Compute statistics (time-averaging)                    │   │
│   │    - Calculate CFL, kinetic energy, convergence metrics     │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FINALIZATION                                 │
│  - Write final output files                                         │
│  - Deallocate memory                                                │
│  - PetscFinalize()                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Source File Organization

### Core Algorithm Files

| File | Lines | Purpose |
|------|-------|---------|
| `main.c` | 2,683 | Entry point, command-line parsing, time loop orchestration |
| `momentum.c` | 2,337 | Momentum equation discretization |
| `poisson.c` | 4,736 | Pressure Poisson equation solver (PETSc) |
| `rhs.c` | 1,785 | Right-hand side computation (convection, viscous) |
| `bcs.c` | 5,812 | All boundary condition implementations |
| `timeadvancing.c` | 1,102 | Explicit time integration (Runge-Kutta) |
| `implicitsolver.c` | 5,812 | Implicit momentum solver |

### Physics Modules

| File | Purpose |
|------|---------|
| `les.c` | Large Eddy Simulation (Smagorinsky, dynamic model) |
| `k-omega.c` | RANS k-omega SST turbulence model |
| `wallfunction.c` | Wall function models for near-wall treatment |
| `level.c` | Level-set method for two-phase flows |
| `ibm.c` | Immersed boundary method (search, interpolation) |
| `ibm_io.c` | IBM surface mesh I/O |
| `fsi.c` | Fluid-structure interaction coupling |
| `fsi_move.c` | FSI body motion and mesh updates |
| `rotor_model.c` | Wind turbine actuator disk/line models |
| `wave.c` | Wave momentum source terms |

### Utilities and I/O

| File | Purpose |
|------|---------|
| `variables.h` | Central header with all data structure definitions |
| `variables.c` | Variable initialization |
| `init.c` | Problem setup and initialization |
| `metrics.c` | Grid metrics and Jacobian computation |
| `compgeom.c` | Computational geometry utilities |
| `solvers.c` | Multigrid restriction/prolongation |
| `data.c` | Post-processing (binary → VTK/Tecplot) |
| `vtk_output.cpp` | Direct VTK output during simulation |
| `xml_input.cpp` | XML configuration parsing |

---

## 5. Module Dependencies

```
                    ┌─────────────┐
                    │   main.c    │
                    │ (orchestr.) │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Flow Solver  │  │   Structure   │  │    Output     │
│               │  │    Solver     │  │               │
│  momentum.c   │  │    fsi.c      │  │   data.c      │
│  poisson.c    │  │  fsi_move.c   │  │ vtk_output.c  │
│  rhs.c        │  │               │  │               │
└───────┬───────┘  └───────┬───────┘  └───────────────┘
        │                  │
        ├──────────┬───────┴───────┬──────────┐
        │          │               │          │
        ▼          ▼               ▼          ▼
┌───────────┐ ┌─────────┐ ┌───────────┐ ┌─────────────┐
│Turbulence │ │   IBM   │ │ Two-Phase │ │    BCs      │
│           │ │         │ │           │ │             │
│  les.c    │ │  ibm.c  │ │  level.c  │ │   bcs.c     │
│ k-omega.c │ │ibm_io.c │ │  wave.c   │ │wallfunction │
└───────────┘ └─────────┘ └───────────┘ └─────────────┘
        │          │               │          │
        └──────────┴───────────────┴──────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Infrastructure │
                  │                 │
                  │   variables.h   │
                  │   metrics.c     │
                  │   compgeom.c    │
                  │   solvers.c     │
                  └─────────────────┘
```

---

## 6. External Dependencies

| Library | Purpose | Used In |
|---------|---------|---------|
| **PETSc** | Distributed arrays (DMDA), vectors, matrices, KSP solvers | All files |
| **MPI** | Parallel communication | Via PETSc |
| **HYPRE** | Algebraic multigrid preconditioner | poisson_hypre.c |
| **BLAS/LAPACK** | Linear algebra | Matrix operations |
| **TinyXML2** | XML parsing | xml_input.cpp |

---

## 7. Build Configuration

The project uses CMake with the following key options:

```cmake
ENABLE_XML_INPUT    # Enable XML configuration file support
ENABLE_VTK_OUTPUT   # Enable direct VTK output during simulation
ENABLE_HYPRE        # Enable HYPRE algebraic multigrid
```

Build:
```bash
mkdir build && cd build
cmake .. -DPETSC_DIR=$PETSC_DIR
make
```

---

## 8. Configuration Files

### control.dat / control.xml
Main simulation parameters:
- Grid dimensions, domain size
- Reynolds number, time step
- Turbulence model selection
- Output frequency
- Physics flags (IBM, FSI, level-set, etc.)

### grid.dat / xyz.dat
Grid coordinates (PLOT3D or ASCII format)

### bcs.dat
Boundary condition types for 6 domain faces

---

## 9. Key Concepts

### Contravariant vs Cartesian Velocities
- **Ucont**: Contravariant velocities (fluxes through cell faces)
- **Ucat**: Cartesian velocities (physical velocity components)
- Conversion via `Contra2Cart()` using grid metrics

### Curvilinear Grid Metrics
- **Csi, Eta, Zet**: Contravariant basis vectors
- **Aj**: Jacobian of transformation
- Computed in `metrics.c`

### Ghost Points
- PETSc DMDA provides automatic ghost point exchange
- Local vectors (prefix `l`) include ghost points: `lUcat`, `lP`, etc.
- Global vectors are distributed without overlap

### Multigrid
- Geometric multigrid with coarsened grids
- V-cycle for pressure Poisson solve
- User contexts linked: `user->user_c` (coarse), `user->user_f` (fine)
