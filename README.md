# VFS-Wind Modernization Guide

This document describes the modernization of VFS-Wind to support modern dependencies (PETSc 3.20+, HYPRE 3.x) and build systems (CMake), with support for both Linux and macOS.

## Summary of Changes

### Build System: Makefile → CMake

A new CMake build system was created to replace the legacy Makefile:

- **`/CMakeLists.txt`** - Root CMake configuration
- **`/Source/CMakeLists.txt`** - Source directory build configuration

The original Makefile is preserved as `Source/makefile.legacy`.

### PETSc API Migration (3.1 → 3.20+)

The following API changes were made to support PETSc 3.20+:

#### Type Renames
| Old | New |
|-----|-----|
| `DA` | `DM` |
| `DALocalInfo` | `DMDALocalInfo` |
| `PetscTruth` | `PetscBool` |
| `PassiveScalar` | `PetscScalar` |
| `PETSC_NULL` | `NULL` |

#### Function Renames
| Old | New |
|-----|-----|
| `DACreate3d` | `DMDACreate3d` |
| `DADestroy` | `DMDestroy` |
| `DAVecGetArray` | `DMDAVecGetArray` |
| `DAVecRestoreArray` | `DMDAVecRestoreArray` |
| `DAGetInfo` | `DMDAGetInfo` |
| `DALocalInfo` | `DMDALocalInfo` |
| `DAGetLocalInfo` | `DMDAGetLocalInfo` |
| `DAGlobalToLocalBegin/End` | `DMGlobalToLocalBegin/End` |
| `DALocalToGlobalBegin/End` | `DMLocalToGlobalBegin/End` |
| `DALocalToLocalBegin/End` | `DMLocalToLocalBegin/End` |
| `DAGetCorners` | `DMDAGetCorners` |
| `DAGetGhostCorners` | `DMDAGetGhostCorners` |
| `DACreateGlobalVector` | `DMCreateGlobalVector` |
| `DACreateLocalVector` | `DMCreateLocalVector` |
| `DACreateNaturalVector` | `DMDACreateNaturalVector` |
| `DAGlobalToNaturalBegin/End` | `DMDAGlobalToNaturalBegin/End` |
| `DASetUniformCoordinates` | `DMDASetUniformCoordinates` |
| `DAGetLocalVector` | `DMGetLocalVector` |
| `DARestoreLocalVector` | `DMRestoreLocalVector` |
| `DAGetGlobalIndices` | `ISLocalToGlobalMappingGetIndices` |
| `PetscGetTime` | `PetscTime` |
| `MatCreateMPIAIJ` | `MatCreateAIJ` |

#### Signature Changes

**DMDACreate3d**: The `wrap` parameter was split into three boundary type parameters:
```c
// Old (PETSc 3.1):
DACreate3d(comm, wrap, stencil, M, N, P, m, n, p, dof, s, lx, ly, lz, &da)

// New (PETSc 3.20+):
DMDACreate3d(comm, bx, by, bz, stencil, M, N, P, m, n, p, dof, s, lx, ly, lz, &dm)
```

**DMDAGetInfo**: Added boundary type output parameters (14 arguments instead of 12).

**DMSetUp**: Must be called after `DMDACreate3d` before using the DM:
```c
DMDACreate3d(..., &dm);
DMSetUp(dm);  // Required in modern PETSc
DMDASetUniformCoordinates(dm, ...);
```

**Destroy functions**: Now require pointer arguments:
```c
// Old:
VecDestroy(vec);
MatDestroy(mat);
DMDestroy(dm);

// New:
VecDestroy(&vec);
MatDestroy(&mat);
DMDestroy(&dm);
```

**KSPSetOperators / PCSetOperators**: Removed 4th argument (MatStructure flag).

**PetscOptionsGet***: Added prefix parameter:
```c
// Old:
PetscOptionsGetString(NULL, "-option", str, len, &flg);

// New:
PetscOptionsGetString(NULL, NULL, "-option", str, len, &flg);
```

**PetscOptionsInsertFile**: Added second parameter.

**PetscOptionsInsertString**: Added NULL first parameter.

**SNES Solver Types**:
- `SNESTR` → `SNESNEWTONTR`
- Changed to `SNESNEWTONLS` for matrix-free compatibility

**Null Space**:
- `KSPSetNullSpace` → `MatSetNullSpace`

**Removed Functions** (replaced with compatibility macros in `variables.h`):
- `PetscGlobalSum` → `MPI_Allreduce` with `MPIU_SUM`
- `PetscGlobalMax` → `MPI_Allreduce` with `MPIU_MAX`
- `PetscGlobalMin` → `MPI_Allreduce` with `MPIU_MIN`

#### Header Changes
- `#include "petscda.h"` → `#include "petscdmda.h"`
- `#include "petscmg.h"` → Removed (MG is now in `petscksp.h`)

#### HYPRE Conditional Compilation
PETSc's HYPRE interface (`PCHYPRESetType`, etc.) is now guarded with `#ifdef PETSC_HAVE_HYPRE` and falls back to GAMG when unavailable.

### HYPRE API Migration (2.x → 3.x)

HYPRE 3.x uses 64-bit integers for indices. The following changes were made in `poisson_hypre.c`:

| Old Type | New Type |
|----------|----------|
| `int` (indices) | `HYPRE_BigInt` |
| `int` (counts) | `HYPRE_Int` |
| `double` (values) | `HYPRE_Complex` |

### Input File Format Changes

PETSc options files now use `#` for comments instead of `!`:
```
# This is a comment (new format)
-dt 0.001
#-rstart 100  # commented out option
```

### BLAS/LAPACK

Replaced discontinued ACML with system BLAS:
- macOS: Uses Accelerate framework automatically
- Linux: Uses OpenBLAS, MKL, or system BLAS

---

## Compilation Instructions

### Prerequisites

#### macOS (Homebrew)

```bash
# Install dependencies
brew install cmake open-mpi petsc hypre openblas

# Verify PETSc installation
pkg-config --modversion PETSc
```

#### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install cmake build-essential libopenmpi-dev
sudo apt install petsc-dev libhypre-dev libopenblas-dev

# Or install PETSc from source for more control
```

#### Linux (CentOS/RHEL)

```bash
sudo yum install cmake gcc gcc-c++ openmpi openmpi-devel
sudo yum install petsc petsc-devel hypre hypre-devel openblas openblas-devel

# Load MPI module if needed
module load mpi/openmpi-x86_64
```

### Building VFS-Wind

```bash
# Clone or navigate to the repository
cd /path/to/VFS-Wind

# Configure with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# The executable will be at: build/Source/vwis
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Release, Debug, RelWithDebInfo) |
| `ENABLE_TECPLOT` | OFF | Enable Tecplot output support |
| `CMAKE_PREFIX_PATH` | - | Custom paths for PETSc/HYPRE |

Example with custom PETSc location:
```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/petsc;/opt/hypre"
```

### Verifying the Build

```bash
# Check the executable
file build/Source/vwis
# Should show: Mach-O 64-bit executable (macOS) or ELF 64-bit (Linux)

# Check linked libraries
otool -L build/Source/vwis  # macOS
ldd build/Source/vwis       # Linux
```

---

## Running Simulations

### Quick Start

```bash
# Navigate to a test case
cd Instructional_Cases/Test_01_3D_Sloshing

# Create grid.dat symlink if needed
ln -sf xyz.dat grid.dat

# Run with MPI (adjust -np for your system)
mpirun -np 4 ../../build/Source/vwis
```

### Input Files

Each simulation case requires:

| File | Description |
|------|-------------|
| `control.dat` | Simulation parameters (time step, physics, etc.) |
| `grid.dat` or `xyz.dat` | Grid coordinates |
| `bcs.dat` | Boundary conditions |

### Control File Parameters

Key parameters in `control.dat`:

```bash
# Time stepping
-dt 0.001           # Time step size
-totalsteps 10000   # Total number of steps
-tio 100            # Output interval

# Physics
-ren 1000           # Reynolds number
-levelset 1         # Enable two-phase flow (0=off, 1=on)
-les 0              # LES turbulence model (0=off, 1=on)
-rans 0             # RANS turbulence model

# Two-phase parameters (when levelset=1)
-rho0 1000          # Density of fluid 0 (water)
-rho1 1             # Density of fluid 1 (air)
-mu0 1.e-3          # Dynamic viscosity of fluid 0
-mu1 1.8e-5         # Dynamic viscosity of fluid 1
-gy -9.8            # Gravity in y-direction

# Solver settings
-poisson 1          # Poisson solver type
-poisson_it 15      # Poisson solver iterations
-imp 4              # Implicit solver type

# Output
-binary 0           # Binary output (0=ASCII, 1=binary)

# Restart
#-rstart 100        # Restart from timestep 100 (commented = fresh start)
```

### Boundary Conditions

The `bcs.dat` file specifies boundary conditions for 6 faces:
```
BC_west BC_east BC_south BC_north BC_bottom BC_top
```

Common BC types:
- `1` = Dirichlet (specified value)
- `2` = Neumann (specified gradient)
- `10` = Wall (no-slip)
- `-1` = Periodic

### Output Files

The simulation produces:
- `ufield*.dat` - Velocity field (Cartesian)
- `vfield*.dat` - Velocity field (contravariant)
- `pfield*.dat` - Pressure field
- `lfield*.dat` - Level set field (two-phase)
- `nvfield*.dat` - Blanking field

These are in PETSc binary format. Use post-processing tools to convert to visualization formats.

### Running on HPC Clusters

Example SLURM script:
```bash
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --job-name=vfswind

module load openmpi petsc hypre

cd $SLURM_SUBMIT_DIR
mpirun -np 128 ./vwis
```

---

## Test Cases

| Case | Description |
|------|-------------|
| `Test_01_3D_Sloshing` | 3D sloshing in a tank (two-phase) |
| `Test_02_VIV_Mounted_Cyll` | Vortex-induced vibration of cylinder |
| `Test_03_2D_Fall_Cyll` | 2D falling cylinder |
| `Test_04_3D_Heave_Decay_Cyll_vertY` | 3D heave decay of cylinder |
| `Test_05_2D_Monochromatic_waves` | 2D monochromatic wave propagation |
| `Test_06_3D_Direct_Wave_XPeriodic` | 3D waves with periodic BC |
| `Test_10_ChannelFlow_Retau3000` | Turbulent channel flow |

---

## Troubleshooting

### Common Issues

**"Unknown first token in options file"**
- PETSc now uses `#` for comments instead of `!`
- Fix: `sed -i 's/^!/#/g' control.dat`

**"Cannot set coordinates until after DMDA has been setup"**
- Add `DMSetUp(dm)` after `DMDACreate3d()`

**"Matrix type mffd does not have a multiply transpose"**
- Change `SNESNEWTONTR` to `SNESNEWTONLS`

**"PCHYPRESetType undefined"**
- PETSc wasn't compiled with HYPRE support
- The code falls back to GAMG automatically

**Linker errors for PETSc functions**
- Ensure PETSc is found: `pkg-config --libs petsc`
- Check `CMAKE_PREFIX_PATH` if using custom installation

### Getting Help

- Check PETSc documentation: https://petsc.org/release/docs/
- HYPRE documentation: https://hypre.readthedocs.io/

---

## Files Modified

### New Files
- `/CMakeLists.txt`
- `/Source/CMakeLists.txt`
- `/README_MODERNIZATION.md`

### Modified Source Files
All `.c` files in `/Source/` were updated for PETSc API compatibility:
- `bcs.c`, `bmv.c`, `compgeom.c`, `distance.c`, `fsi.c`, `fsi_move.c`
- `ibm.c`, `ibm_io.c`, `implicitsolver.c`, `init.c`, `k-omega.c`
- `les.c`, `level.c`, `main.c`, `metrics.c`, `momentum.c`
- `poisson.c`, `poisson_hypre.c`, `rhs.c`, `rhs2.c`, `rotor_model.c`
- `solvers.c`, `timeadvancing.c`, `timeadvancing1.c`, `variables.c`
- `wallfunction.c`, `wave.c`

### Modified Header Files
- `/Source/variables.h` - Updated includes, types, and compatibility macros
- `/Source/list.h` - Updated PETSc types

### Test Case Files
All `control.dat` files updated to use `#` comments.

---

## XML Input System

VFS-Wind now supports structured XML configuration files as an alternative to the legacy `control.dat` format.

### Features

- **Structured, validated configuration** - XML provides a clear, hierarchical structure
- **Auto-detection** - If `control.xml` exists, it is automatically used
- **Backward compatible** - Falls back to `control.dat` if no XML file is found
- **Command-line override** - Use `-xml filename.xml` to specify a custom XML file

### XML Schema

```xml
<?xml version="1.0" encoding="UTF-8"?>
<vfswind version="1.0">
  <simulation>
    <timestep dt="0.001" totalsteps="10000" output_interval="100"/>
    <restart enabled="0" timestep="0"/>
  </simulation>

  <physics>
    <reynolds_number>1000</reynolds_number>
    <gravity x="0.0" y="-9.8" z="0.0"/>
  </physics>

  <turbulence>
    <les enabled="0" model="smagorinsky" cs="0.1"/>
    <rans enabled="0" model="k-omega-sst"/>
    <wall_function enabled="0"/>
  </turbulence>

  <levelset enabled="1">
    <fluid0 density="1000" viscosity="1.0e-3"/>
    <fluid1 density="1" viscosity="1.8e-5"/>
    <iterations>10</iterations>
  </levelset>

  <solvers>
    <poisson type="1" iterations="15" tolerance="5e-9"/>
    <momentum implicit="0" max_iterations="50"/>
  </solvers>

  <parallel>
    <periodic i="0" j="0" k="0"/>
  </parallel>

  <output>
    <path>./</path>
    <vtk enabled="1"/>
    <binary enabled="1"/>
  </output>

  <grid file="grid.dat" format="xyz"/>
  <boundaries file="bcs.dat"/>
</vfswind>
```

### Usage

```bash
# Auto-detect (uses control.xml if present, otherwise control.dat)
mpirun -np 4 ./vwis

# Explicit XML file
mpirun -np 4 ./vwis -xml myconfig.xml
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_XML_INPUT` | ON | Enable XML input file parsing (uses bundled TinyXML-2) |

To disable XML support:
```bash
cmake -B build -DENABLE_XML_INPUT=OFF
```

---

## VTK Output System

VFS-Wind now supports VTK XML output for direct visualization in ParaView and other VTK-compatible tools.

### Features

- **ParaView-compatible** - Native VTK XML Structured Grid format (.vts/.pvts)
- **Parallel output** - Each MPI process writes its own piece; a collection file combines them
- **Binary encoding** - Base64-encoded binary data for compact files
- **Multiple fields** - Outputs coordinates, velocity, pressure, level set (if enabled), and blanking

### Output Fields

| Field | Components | Description |
|-------|------------|-------------|
| Coordinates | X, Y, Z | Grid point locations |
| Velocity | U, V, W | Cartesian velocity components |
| Pressure | scalar | Pressure field |
| Levelset | scalar | Level set function (two-phase only) |
| Nvert | scalar | Blanking/immersed boundary marker |

### File Structure

**Parallel output (multiple MPI processes):**
```
output_000100.pvts        # Collection file (rank 0 only)
output_000100_p0.vts      # Process 0 data
output_000100_p1.vts      # Process 1 data
...
```

### Usage

**Enable via XML configuration:**
```xml
<output>
  <vtk enabled="1"/>
</output>
```

**Enable via command line:**
```bash
mpirun -np 4 ./vwis -vtk_output 1
```

**Options:**
```bash
-vtk_output 1    # Enable VTK output (default: 0)
-vtk_binary 1    # Use binary encoding (default: 1, set to 0 for ASCII)
```

### Viewing in ParaView

```bash
# Open the parallel collection file
paraview output_000100.pvts

# Or for single-process runs
paraview output_000100_p0.vts
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_VTK_OUTPUT` | ON | Enable VTK XML output for ParaView |

To disable VTK support:
```bash
cmake -B build -DENABLE_VTK_OUTPUT=OFF
```

---

## New Files Added (Modernization)

### XML Input System
- `/Source/tinyxml2/tinyxml2.h` - TinyXML-2 library header (bundled)
- `/Source/tinyxml2/tinyxml2.cpp` - TinyXML-2 library implementation (bundled)
- `/Source/xml_input.h` - XML input parsing declarations
- `/Source/xml_input.cpp` - XML input parsing implementation

### VTK Output System
- `/Source/vtk_output.h` - VTK output declarations
- `/Source/vtk_output.cpp` - VTK output implementation

### Example Files
- `/Instructional_Cases/Test_01_3D_Sloshing/control.xml` - Example XML configuration
