# VFS-Wind Examples

This directory contains example test cases demonstrating VFS-Wind's capabilities.

## Available Examples

| # | Case | Description | Physics |
|---|------|-------------|---------|
| 01 | [Sloshing Tank](01_Sloshing_Tank/) | 3D sloshing in a tank | Two-phase (Level Set), laminar |
| 02 | [Channel Flow](02_ChannelFlow/) | Turbulent channel flow Re_tau~3000 | LES (Dynamic Smagorinsky) |
| 03 | [VIV Mounted Cylinder](03_VIV_Mounted_Cylinder/) | Vortex-induced vibration | IBM + FSI, laminar |
| 04 | [2D Falling Cylinder](04_2D_Fall_Cylinder/) | Cylinder falling through interface | Two-phase + IBM + FSI |
| 05 | [3D Heave Decay Cylinder](05_3D_Heave_Decay_Cylinder/) | Cylinder heave decay at interface | Two-phase + IBM + FSI, LES |

## Quick Start

Each example has a `run.sh` script with a consistent interface:

```bash
cd examples/01_Sloshing_Tank

# Run full workflow (build, preprocess, simulate, post-process)
./run.sh

# Or run individual steps
./run.sh build        # Build VFS-Wind
./run.sh preprocess   # Check input files
./run.sh simulate     # Run simulation
./run.sh postprocess  # Convert output to VTK
./run.sh clean        # Clean output files
./run.sh help         # Show help
```

### Environment Variables

All examples support these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NP_SIM` | 4 | Number of MPI processes for simulation |
| `NP_POST` | 1 | Number of MPI processes for post-processing |
| `USE_XML` | 1 | Use XML config (1) or control.dat (0) |
| `ENABLE_GPU` | 0/1 | Enable GPU acceleration via Kokkos |
| `OMP_THREADS` | auto | Number of OpenMP threads for GPU backend |
| `TIS` | 0 | Starting timestep for post-processing |
| `TIE` | varies | Ending timestep for post-processing |
| `TS` | varies | Timestep stride for post-processing |
| `AVG` | 0 | Averaging mode: 0=off, 1=full, 2=TKE only |

### Examples

```bash
# Run with 8 MPI processes
NP_SIM=8 ./run.sh simulate

# Run with GPU enabled
ENABLE_GPU=1 ./run.sh simulate

# Post-process specific timestep range
TIS=0 TIE=1000 TS=100 ./run.sh postprocess

# Use legacy control.dat instead of XML
USE_XML=0 ./run.sh simulate
```

---

## End-to-End Example: Turbulent Channel Flow

This section demonstrates a complete VFS-Wind workflow using the **02_ChannelFlow** example—a turbulent channel flow simulation at Re_tau ~ 3000.

### Step 1: Build VFS-Wind

```bash
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
cd examples/02_ChannelFlow
ls -la
```

The directory contains:

| File | Description |
|------|-------------|
| `control.xml` | Simulation parameters (XML format) |
| `control.dat` | Legacy simulation parameters |
| `bcs.dat` | Boundary conditions (6 integers for 6 faces) |
| `mesh.xml` | Mesh generation configuration |
| `xyz.dat` | Grid coordinates (121 x 41 x 61 points) |

**Key parameters in control.xml/control.dat:**
```
dt = 0.001              # Time step
totalsteps = 10000      # Total simulation steps
output_interval = 200   # Output every 200 steps
reynolds_number = 62500 # Reynolds number
les = 2                 # LES model (Dynamic Smagorinsky)
max_cs = 0.2            # Maximum Smagorinsky coefficient
channel_height = 0.8    # Channel height
ii_periodic = 1         # Periodic in X (streamwise)
kk_periodic = 1         # Periodic in Z (spanwise)
flux = 1.3240512        # Mass flux (drives the flow)
```

**bcs.dat boundary conditions:**
```
100 100 1 10 100 100
```
- X-direction (West/East): Periodic (codes 100)
- Y-direction (South/North): Walls (codes 1 and 10)
- Z-direction (Bottom/Top): Periodic (codes 100)

### Step 3: Run the Simulation

```bash
# Using run.sh (recommended)
./run.sh simulate

# Or manually
mpirun -np 4 ../../build/Source/vwis -xml control.xml

# For a quick test, reduce totalsteps in control.xml
```

**Expected output during simulation:**
```
Reading control.xml...
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
su0_000200_0.dat      # Velocity sum (for averaging)
su1_000200_0.dat      # Cross-product sum (for Reynolds stresses)
su2_000200_0.dat      # Velocity squared sum (for variances)
```

### Step 4: Post-Process Results

The `data` executable converts binary output files to VTK format for ParaView.

#### Basic VTK Output (Instantaneous Fields)

```bash
# Using run.sh
./run.sh postprocess

# Or manually: Convert timesteps 200 to 1000, every 200 steps
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
Result000200.vtm      # Multi-block container
Result000400_00.vts
...
```

#### Averaged Results (Reynolds Stresses, TKE)

If the simulation was run with averaging enabled (`-averaging 1`), extract statistics:

```bash
# Extract averaged results with full Reynolds stress tensor
mpirun -np 1 ../../build/Source/data -tis 1000 -tie 1000 -ts 200 -vtk 1 -avg 1
```

**Averaging options:**
| Option | Output Fields |
|--------|---------------|
| `-avg 1` | U, V, W (mean), uu, vv, ww, uv, vw, uw (Reynolds stresses), TKE |
| `-avg 2` | U, V, W (mean), TKE only |

**Fields in averaged VTK output:**
- `Umean` - Mean velocity vector (U, V, W)
- `uu`, `vv`, `ww` - Normal Reynolds stresses
- `uv`, `vw`, `uw` - Shear Reynolds stresses
- `TKE` - Turbulent kinetic energy (0.5 x (uu + vv + ww))

### Step 5: Visualize in ParaView

```bash
paraview
# File -> Open -> Select Result000200_00.vts (or .vtm for multi-block)
```

**Recommended visualizations for channel flow:**

1. **Velocity magnitude**
   - Apply: Filters -> Common -> Calculator
   - Expression: `mag(Ucat)` or `sqrt(Ucat_X^2 + Ucat_Y^2 + Ucat_Z^2)`

2. **Mean velocity profile**
   - Apply: Filters -> Data Analysis -> Plot Over Line
   - Set line from bottom wall (y=0) to top wall (y=0.4)
   - Plot U velocity vs Y

3. **Reynolds stresses** (from averaged output)
   - Color by `uu`, `vv`, `ww` to see turbulence intensity distribution
   - Plot profiles of Reynolds stresses vs wall distance

4. **Q-criterion** (vortex visualization)
   - Run post-processing with: `-qcr 1`
   - Isosurfaces of Q-criterion show vortical structures

### Step 6: Running on HPC Clusters

Example SLURM batch script:

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
mpirun -np 128 ./vwis -xml control.xml

# Post-process final results
mpirun -np 1 ./data -tis 10000 -tie 10000 -ts 200 -vtk 1 -avg 1
```

### Tips for Channel Flow Simulations

1. **Grid resolution**: The 121x41x61 grid is suitable for wall-modeled LES. For wall-resolved LES, increase the Y-resolution near walls.

2. **Time averaging**: Enable averaging by adding `-averaging 1` to control.dat or setting `<options averaging="1"/>` in control.xml. Statistics accumulate from the start; discard initial transient data.

3. **Convergence check**: Monitor the mass flux and mean velocity profile to ensure statistical convergence.

4. **Restart capability**: After a run completes, add `-rstart N` (where N is the last timestep) to continue from that point.

5. **Output frequency**: Balance between disk space and temporal resolution. For statistics, output every 200 steps is reasonable; for animations, use smaller values.

---

## Input Files

Each simulation case requires:

| File | Description |
|------|-------------|
| `control.xml` or `control.dat` | Simulation parameters (time step, physics, etc.) |
| `grid.dat` or `xyz.dat` | Grid coordinates |
| `bcs.dat` | Boundary conditions |
| `ibmdata00` (optional) | Immersed boundary surface mesh |

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
- `100` = Periodic (alternative)

### Output Files

The simulation produces:
- `ufield*.dat` - Velocity field (Cartesian)
- `vfield*.dat` - Velocity field (contravariant)
- `pfield*.dat` - Pressure field
- `lfield*.dat` - Level set field (two-phase cases)
- `nvfield*.dat` - Blanking field
- `FSI_position*` - FSI displacement history (FSI cases)
- `Force_Coeff_*` - Force coefficients (IBM cases)

These are in PETSc binary format. Use the `data` post-processing tool to convert to VTK for visualization.

---

## Legacy Examples

The `legacy/` directory contains older test cases that have not been updated to the new format:

| Case | Description |
|------|-------------|
| `Test_03_2D_Fall_Cyll` | Original 2D falling cylinder (now 04_2D_Fall_Cylinder) |
| `Test_04_3D_Heave_Decay_Cyll_vertY` | Original 3D heave decay (now 05_3D_Heave_Decay_Cylinder) |
| `Test_05_2D_Monochromatic_waves` | 2D monochromatic wave propagation |
| `Test_06_3D_Direct_Wave_XPeriodic` | 3D waves with periodic BC |
| `Test_08_ClipperTurbine` | Wind turbine simulation |
| `Test_09_ModelWindTurbine` | Model wind turbine simulation |
| `Test_10_ChannelFlow_Retau3000` | Original channel flow (now 02_ChannelFlow) |

These cases may require updates to work with the current version of VFS-Wind.

---

## GPU Acceleration

VFS-Wind supports GPU acceleration via Kokkos for portable performance across different hardware:

```bash
# Build with GPU support
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GPU=ON
cmake --build build -j$(nproc)

# Run with GPU enabled
ENABLE_GPU=1 ./run.sh simulate
```

See [GPU Build Guide](../docs/GPU_BUILD_GUIDE.md) for detailed instructions.
