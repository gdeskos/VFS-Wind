# VFS-Wind

VFS-Wind is a parallel CFD solver for incompressible flows using the Virtual Flow Simulator (VFS) methodology. It supports:

- Large Eddy Simulation (LES) and RANS turbulence modeling
- Two-phase flows via level set method
- Immersed boundary methods for complex geometries
- Fluid-structure interaction
- Wind turbine rotor modeling

Built on PETSc for parallel scalability, VFS-Wind runs on laptops to HPC clusters.

## Quick Start

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run a test case
cd examples/Test_01_3D_Sloshing
mpirun -np 4 ../../build/Source/vwis
```

## Prerequisites

**macOS (Homebrew):**
```bash
brew install cmake open-mpi petsc hypre openblas
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install cmake build-essential libopenmpi-dev petsc-dev libhypre-dev libopenblas-dev
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install cmake gcc gcc-c++ openmpi openmpi-devel petsc petsc-devel hypre hypre-devel openblas openblas-devel
```

## Building

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Build targets:**
| Target | Description |
|--------|-------------|
| `vwis` | Main solver |
| `data` | Post-processing tool (converts output to VTK/Tecplot) |

**CMake options:**
| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Release, Debug, RelWithDebInfo) |
| `ENABLE_TECPLOT` | OFF | Tecplot output support |
| `ENABLE_XML_INPUT` | ON | XML configuration file support |
| `ENABLE_VTK_OUTPUT` | ON | VTK output for ParaView |
| `CMAKE_PREFIX_PATH` | - | Custom paths for PETSc/HYPRE |

**Custom PETSc location:**
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/opt/petsc;/opt/hypre"
```

## Running Simulations

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

## End-to-End Example: Turbulent Channel Flow

This section demonstrates how to run VFS-Wind from start to finish using the **Test_10_ChannelFlow_Retau3000** example case—a turbulent channel flow simulation at friction Reynolds number Re_τ ≈ 3000.

### Step 1: Build VFS-Wind

```bash
# Navigate to the VFS-Wind directory
cd /path/to/VFS-Wind

# Create build directory and configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build main solver and post-processing tool
cmake --build build -j$(nproc)

# Verify executables exist
ls -la build/Source/vwis build/Source/data
```

This builds:
- `vwis` - The main VFS-Wind solver
- `data` - Post-processing tool for converting binary output to VTK/Tecplot formats

### Step 2: Examine the Test Case

```bash
cd examples/Test_10_ChannelFlow_Retau3000
ls -la
```

The directory contains:

| File | Description |
|------|-------------|
| `control.dat` | Simulation parameters |
| `bcs.dat` | Boundary conditions (6 integers for 6 faces) |
| `xyz.dat` | Grid coordinates (121 × 41 × 61 points) |

**control.dat** key parameters:
```bash
-dt 0.001              # Time step
-totalsteps 10000      # Total simulation steps
-tio 200               # Output every 200 steps
-ren 62500             # Reynolds number
-les 2                 # LES turbulence model (Dynamic Smagorinsky)
-max_cs 0.2            # Maximum Smagorinsky coefficient
-viscosity_wallmodel 1 # Wall function model
-channel_height 0.8    # Channel height
-ii_periodic 1         # Periodic in X (streamwise)
-kk_periodic 1         # Periodic in Z (spanwise)
-xyz 1                 # Use xyz.dat grid format
-flux 1.3240512        # Mass flux (drives the flow)
```

**bcs.dat** boundary conditions:
```
100 100 1 10 100 100
```
- X-direction (West/East): Periodic (codes 100)
- Y-direction (South/North): Walls (codes 1 and 10)
- Z-direction (Bottom/Top): Periodic (codes 100)

### Step 3: Run the Simulation

```bash
# Create a working directory (recommended)
mkdir -p run_channel
cd run_channel

# Copy input files
cp ../control.dat .
cp ../bcs.dat .
cp ../xyz.dat .

# Run with MPI (adjust -np based on your system)
mpirun -np 4 ../../build/Source/vwis

# For a quick test, reduce totalsteps first:
# Edit control.dat: -totalsteps 1000
```

**Expected output during simulation:**
```
Reading control.dat...
Grid dimensions: 121 x 41 x 61
Time step: 0.001
Reynolds number: 62500
LES model enabled (Dynamic Smagorinsky)
...
Timestep 200, Time = 0.200, CFL = 0.45
Timestep 400, Time = 0.400, CFL = 0.46
...
```

**Output files generated:**
```
ufield000200_0.dat    # Velocity field at timestep 200
pfield000200_0.dat    # Pressure field at timestep 200
nvfield000200_0.dat   # Blanking field at timestep 200
su0_000200_0.dat      # Velocity sum (for averaging, if enabled)
su1_000200_0.dat      # Cross-product sum (for Reynolds stresses)
su2_000200_0.dat      # Velocity squared sum (for variances)
...
```

### Step 4: Post-Process Results

The `data` executable converts binary output files to VTK format for ParaView.

#### Basic VTK Output (Instantaneous Fields)

```bash
# Convert timesteps 200 to 1000, every 200 steps, to VTK
mpirun -np 1 ../../build/Source/data -tis 200 -tie 1000 -ts 200 -vtk 1
```

**Command-line options:**
| Option | Description |
|--------|-------------|
| `-tis N` | Starting timestep index |
| `-tie N` | Ending timestep index |
| `-ts N` | Timestep stride (default: 5) |
| `-vtk 1` | Enable VTK output (for ParaView) |
| `-binary 0` | Input files are ASCII (match simulation setting) |
| `-xyz 1` | Grid is in xyz.dat format |

**Output files:**
```
Result000200_00.vts   # VTK structured grid for timestep 200
Result000200.vtm      # Multi-block container (if multiple blocks)
Result000400_00.vts
...
```

#### Averaged Results (Reynolds Stresses, TKE)

If the simulation was run with averaging enabled (`-averaging 1`), you can extract mean velocities and turbulence statistics:

```bash
# Extract averaged results with full Reynolds stress tensor
mpirun -np 1 ../../build/Source/data -tis 1000 -tie 1000 -ts 200 -vtk 1 -avg 1
```

**Averaging options:**
| Option | Output Fields |
|--------|---------------|
| `-avg 1` | U, V, W (mean), uu, vv, ww, uv, vw, uw (Reynolds stresses), TKE |
| `-avg 2` | U, V, W (mean), TKE only |

**Output files:**
```
Result001000-avg_00.vts   # Averaged VTK file
Result001000-avg.vtm      # Multi-block container
```

**Fields in averaged VTK output:**
- `Umean` - Mean velocity vector (U, V, W)
- `uu`, `vv`, `ww` - Normal Reynolds stresses
- `uv`, `vw`, `uw` - Shear Reynolds stresses
- `TKE` - Turbulent kinetic energy (0.5 × (uu + vv + ww))
- `Nvert` - Blanking field

### Step 5: Visualize in ParaView

```bash
# Open ParaView
paraview

# Load the VTK files:
# File → Open → Select Result000200_00.vts (or .vtm for multi-block)
```

**Recommended visualizations for channel flow:**

1. **Velocity magnitude**
   - Apply: Filters → Common → Calculator
   - Expression: `mag(Ucat)` or `sqrt(Ucat_X^2 + Ucat_Y^2 + Ucat_Z^2)`

2. **Mean velocity profile**
   - Apply: Filters → Data Analysis → Plot Over Line
   - Set line from bottom wall (y=0) to top wall (y=0.4)
   - Plot U velocity vs Y

3. **Reynolds stresses** (from averaged output)
   - Color by `uu`, `vv`, `ww` to see turbulence intensity distribution
   - Plot profiles of Reynolds stresses vs wall distance

4. **Q-criterion** (vortex visualization)
   - Run post-processing with: `-qcr 1`
   - Isosurfaces of Q-criterion show vortical structures

### Step 6: Running on HPC Clusters

Example SLURM batch script for production runs:

```bash
#!/bin/bash
#SBATCH --job-name=channel_flow
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --output=channel_%j.out
#SBATCH --error=channel_%j.err

# Load modules (adjust for your system)
module load openmpi petsc hypre

# Navigate to run directory
cd $SLURM_SUBMIT_DIR

# Run simulation
mpirun -np 128 ./vwis

# Post-process final results
mpirun -np 1 ./data -tis 10000 -tie 10000 -ts 200 -vtk 1 -avg 1
```

### Tips for Channel Flow Simulations

1. **Grid resolution**: The 121×41×61 grid is suitable for wall-modeled LES. For wall-resolved LES, increase the Y-resolution near walls.

2. **Time averaging**: Enable averaging by adding `-averaging 1` to control.dat. Statistics accumulate from the start; discard initial transient data.

3. **Convergence check**: Monitor the mass flux and mean velocity profile to ensure statistical convergence.

4. **Restart capability**: After a run completes, add `-rstart N` (where N is the last timestep) to continue from that point.

5. **Output frequency**: Balance between disk space and temporal resolution. For statistics, `-tio 200` is reasonable; for animations, use smaller values.

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

**Usage:**
```bash
# Auto-detect (uses control.xml if present, otherwise control.dat)
mpirun -np 4 ./vwis

# Explicit XML file
mpirun -np 4 ./vwis -xml myconfig.xml
```

---

## VTK Output

VFS-Wind supports VTK XML output (.vts/.pvts) for direct visualization in ParaView.

**Enable via command line:**
```bash
mpirun -np 4 ./vwis -vtk_output 1
```

**Or via XML configuration:**
```xml
<output>
  <vtk enabled="1"/>
</output>
```

**Output fields:** Coordinates, Velocity (U,V,W), Pressure, Levelset (two-phase), Nvert (blanking)

---

## Post-Processing Tool

The `data` executable converts simulation output to VTK/Tecplot formats.

```bash
# Convert timesteps 0-50000 (every 100 steps) to VTK format
mpirun -np 1 ./data -tis 0 -tie 50000 -ts 100 -vtk 1
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-tis N` | Starting timestep index | required |
| `-tie N` | Ending timestep index | same as tis |
| `-ts N` | Timestep stride | 5 |
| `-vtk 1` | Enable VTK output | 0 |
| `-avg N` | Include averaged results (1=full, 2=TKE only) | 0 |
| `-binary N` | Binary input format | 0 |
| `-xyz N` | Grid format (1=xyz.dat) | 0 |
| `-qcr N` | Compute Q-criterion | 0 |
