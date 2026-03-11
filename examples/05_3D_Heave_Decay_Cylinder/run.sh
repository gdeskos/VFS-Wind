#!/bin/bash
#=============================================================================
# VFS-Wind 3D Heave Decay Cylinder Example Runner
# Test Case: Two-Phase Flow with FSI (Cylinder Heave Decay at Interface)
#
# NOTE: This test case requires a grid file (grid.dat) that is not included.
#       You must generate or provide this file before running.
#
# Usage:
#   ./run.sh              # Run full workflow (build, simulate, post-process)
#   ./run.sh build        # Build VFS-Wind only
#   ./run.sh preprocess   # Check/prepare input files
#   ./run.sh simulate     # Run simulation only
#   ./run.sh postprocess  # Post-process results only
#   ./run.sh clean        # Clean output files
#   ./run.sh help         # Show this help message
#
# Configuration:
#   Edit the variables below to customize the run
#=============================================================================

# === CONFIGURATION ===
# Number of MPI processes for simulation
NP_SIM=${NP_SIM:-8}

# Number of MPI processes for post-processing (usually 1)
NP_POST=${NP_POST:-1}

# Use XML config (1) or legacy control.dat (0)
USE_XML=${USE_XML:-1}

# Enable GPU acceleration (1=on, 0=off) - uses Kokkos for portable GPU kernels
ENABLE_GPU=${ENABLE_GPU:-0}

# Number of OpenMP threads for GPU backend (0=auto-detect)
OMP_THREADS=${OMP_THREADS:-0}

# Timestep range for post-processing
TIS=${TIS:-0}          # Starting timestep
TIE=${TIE:-10000}      # Ending timestep
TS=${TS:-1000}         # Timestep stride

# Enable averaging output (0=off, 1=Reynolds stresses, 2=TKE only)
AVG=${AVG:-0}

# Path to VFS-Wind root directory (auto-detected)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VFSWIND_ROOT="${SCRIPT_DIR}/../.."
BUILD_DIR="${VFSWIND_ROOT}/build"

# === HELPER FUNCTIONS ===
print_header() {
    echo "=============================================="
    echo " $1"
    echo "=============================================="
}

print_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

check_executable() {
    if [ ! -x "$1" ]; then
        echo "ERROR: Executable not found: $1"
        echo "       Please run './run.sh build' first."
        exit 1
    fi
}

# === BUILD FUNCTION ===
do_build() {
    print_header "Building VFS-Wind"

    cd "${VFSWIND_ROOT}"

    print_step "Configuring with CMake..."
    CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    if [ "${ENABLE_GPU}" -eq 1 ]; then
        CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GPU=ON"
        echo "  GPU acceleration: ENABLED (Kokkos)"
    else
        echo "  GPU acceleration: DISABLED"
    fi
    cmake -B build ${CMAKE_ARGS}

    if [ $? -ne 0 ]; then
        echo "ERROR: CMake configuration failed!"
        exit 1
    fi

    print_step "Building executables..."
    cmake --build build -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed!"
        exit 1
    fi

    print_step "Build successful!"
    echo "  Solver:         ${BUILD_DIR}/Source/vwis"
    echo "  Post-processor: ${BUILD_DIR}/Source/data"

    cd "${SCRIPT_DIR}"
}

# === PREPROCESS FUNCTION ===
do_preprocess() {
    print_header "Preprocessing: Checking Input Files"

    cd "${SCRIPT_DIR}"

    local errors=0

    print_step "Checking required files..."

    # Check grid file
    if [ -f "grid.dat" ]; then
        echo "  [OK] grid.dat found"
        local size=$(ls -lh grid.dat | awk '{print $5}')
        echo "       Size: ${size}"
    else
        echo "  [ERROR] grid.dat NOT found!"
        echo "         This test case requires a grid file that is not included."
        echo "         You must generate or provide grid.dat before running."
        errors=$((errors + 1))
    fi

    # Check boundary conditions
    if [ -f "bcs.dat" ]; then
        echo "  [OK] bcs.dat found"
    else
        echo "  [ERROR] bcs.dat NOT found!"
        errors=$((errors + 1))
    fi

    # Check IBM data (cylinder geometry)
    if [ -f "ibmdata00" ]; then
        echo "  [OK] ibmdata00 (cylinder geometry) found"
        local size=$(ls -lh ibmdata00 | awk '{print $5}')
        echo "       Size: ${size}"
    else
        echo "  [ERROR] ibmdata00 NOT found!"
        errors=$((errors + 1))
    fi

    # Check configuration file
    if [ "${USE_XML}" -eq 1 ]; then
        if [ -f "control.xml" ]; then
            echo "  [OK] control.xml found"
        else
            echo "  [WARNING] control.xml NOT found, will use control.dat"
        fi
    fi
    if [ -f "control.dat" ]; then
        echo "  [OK] control.dat found"
    fi

    # Check reference data
    if [ -f "_FSI_position00_good_run" ]; then
        echo "  [OK] Reference FSI data available"
    fi

    print_step "Validating setup..."

    # Check for required parameters in XML
    if [ -f "control.xml" ]; then
        if grep -q "levelset.*enabled=\"1\"" control.xml; then
            echo "  [OK] Level Set (two-phase) enabled"
        fi
        if grep -q "immersed_boundary.*enabled=\"1\"" control.xml; then
            echo "  [OK] Immersed Boundary Method enabled"
        fi
        if grep -q "fsi.*enabled=\"1\"" control.xml; then
            echo "  [OK] Fluid-Structure Interaction enabled"
        fi
    fi

    print_step "Summary"
    if [ ${errors} -eq 0 ]; then
        echo "  All required files present. Ready to simulate!"
        echo ""
        echo "  Test case: 3D Heave Decay Cylinder"
        echo "  Physics:   Re=6666.67, two-phase (level set), LES"
        echo "  FSI:       1-DOF (Y-direction heave), mass ratio=0.25"
        echo "  Gravity:   -9.81 m/s^2 (Y-direction)"
        echo "  Expected:  Cylinder oscillates and decays at interface"
    else
        echo "  Found ${errors} error(s). Please fix before running simulation."
        exit 1
    fi
}

# === SIMULATION FUNCTION ===
do_simulate() {
    print_header "Running 3D Heave Decay Cylinder Simulation"

    check_executable "${BUILD_DIR}/Source/vwis"

    cd "${SCRIPT_DIR}"

    # Check required files
    if [ ! -f "grid.dat" ]; then
        echo "ERROR: grid.dat not found!"
        echo ""
        echo "This test case requires a grid file that is not included."
        echo "You must generate or provide grid.dat before running."
        echo ""
        echo "Grid requirements:"
        echo "  - 3D structured mesh"
        echo "  - Suitable for two-phase flow with level set"
        echo "  - Sufficient resolution near the free surface (dthick=0.006)"
        exit 1
    fi
    if [ ! -f "ibmdata00" ]; then
        echo "ERROR: ibmdata00 (cylinder geometry) not found!"
        exit 1
    fi

    # Determine config file
    if [ "${USE_XML}" -eq 1 ] && [ -f "control.xml" ]; then
        CONFIG_ARGS="-xml control.xml"
        echo "Using XML configuration: control.xml"
    else
        CONFIG_ARGS=""
        echo "Using legacy configuration: control.dat"
    fi

    # Set OpenMP threads for GPU backend
    if [ "${OMP_THREADS}" -gt 0 ]; then
        export OMP_NUM_THREADS=${OMP_THREADS}
    elif [ -z "${OMP_NUM_THREADS}" ]; then
        export OMP_NUM_THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    fi

    print_step "Starting simulation with ${NP_SIM} MPI processes..."
    echo "  Physics:    Two-phase flow with Level Set"
    echo "  Re:         6666.67"
    echo "  Gravity:    -9.81 m/s^2 (Y-direction)"
    echo "  LES:        Dynamic model"
    echo "  FSI:        Heave decay with mass ratio 0.25"
    echo "  Timesteps:  10000 (dt=0.0005)"
    if [ "${ENABLE_GPU}" -eq 1 ]; then
        echo "  GPU:        ENABLED (OMP_NUM_THREADS=${OMP_NUM_THREADS})"
    else
        echo "  GPU:        DISABLED"
    fi
    echo ""

    mpirun -np ${NP_SIM} "${BUILD_DIR}/Source/vwis" ${CONFIG_ARGS}

    if [ $? -ne 0 ]; then
        echo "ERROR: Simulation failed!"
        exit 1
    fi

    print_step "Simulation completed successfully!"

    # Check for FSI output
    if [ -f "FSI_position00" ]; then
        echo "  FSI displacement data: FSI_position00"
        local lines=$(wc -l < FSI_position00)
        echo "  Recorded ${lines} timesteps of cylinder motion"

        # Compare with reference if available
        if [ -f "_FSI_position00_good_run" ]; then
            echo ""
            echo "  Reference data available: _FSI_position00_good_run"
            echo "  Compare your results to validate the simulation."
        fi
    fi
}

# === POST-PROCESSING FUNCTION ===
do_postprocess() {
    print_header "Post-Processing Results"

    check_executable "${BUILD_DIR}/Source/data"

    cd "${SCRIPT_DIR}"

    # Check if output files exist
    if ! ls ufield*.dat 1> /dev/null 2>&1; then
        echo "ERROR: No output files found (ufield*.dat)"
        echo "       Please run the simulation first."
        exit 1
    fi

    print_step "Converting binary output to VTK format..."
    echo "  Timestep range: ${TIS} to ${TIE} (stride: ${TS})"
    echo "  Averaging mode: ${AVG}"
    echo ""

    # Build post-processing command
    # Note: levelset=1, binary grid format
    POST_CMD="${BUILD_DIR}/Source/data -tis ${TIS} -tie ${TIE} -ts ${TS} -vtk 1 -xyz 0 -binary 1 -levelset 1"

    if [ "${AVG}" -gt 0 ]; then
        POST_CMD="${POST_CMD} -avg ${AVG}"
    fi

    # Add XML config if using XML
    if [ "${USE_XML}" -eq 1 ] && [ -f "control.xml" ]; then
        POST_CMD="${POST_CMD} -xml control.xml"
    fi

    mpirun -np ${NP_POST} ${POST_CMD}

    if [ $? -ne 0 ]; then
        echo "WARNING: Post-processing encountered errors (some timesteps may be missing)"
    fi

    print_step "Post-processing completed!"

    # List generated files
    echo "Generated VTK files:"
    ls -la Result*.vts Result*.vtm 2>/dev/null || echo "  (no VTK files found)"

    # FSI analysis
    if [ -f "FSI_position00" ]; then
        print_step "FSI Analysis"
        echo "  Cylinder heave history: FSI_position00"
        echo ""
        echo "  To visualize:"
        echo "    - Plot column 2 (Y displacement) vs column 1 (time)"
        echo "    - Expected: exponential decay of oscillation amplitude"
        echo "    - Natural frequency should match analytical prediction"
    fi
}

# === CLEAN FUNCTION ===
do_clean() {
    print_header "Cleaning Output Files"

    cd "${SCRIPT_DIR}"

    print_step "Removing binary output files..."
    rm -f ufield*.dat vfield*.dat pfield*.dat nvfield*.dat lfield*.dat
    rm -f su0_*.dat su1_*.dat su2_*.dat sp_*.dat
    rm -f kfield*.dat qfield*.dat cs*.dat
    rm -f *.info

    print_step "Removing VTK files..."
    rm -f Result*.vts Result*.vtm Result*.plt

    print_step "Removing FSI output files..."
    rm -f FSI_position0[0-9] FSI_Angle* DATA_FSI*
    rm -f Force_Coeff_* Momt_Coeff_* Power_*
    rm -f surface*.dat

    print_step "Removing convergence/diagnostic files..."
    rm -f Converge_* Kinetic_Energy.dat mass.dat shear_velocity.dat

    print_step "Removing log files..."
    rm -f err* output*.log

    echo "Clean completed!"
    echo "(Note: Reference file _FSI_position00_good_run is preserved)"
}

# === HELP FUNCTION ===
do_help() {
    cat << EOF
VFS-Wind 3D Heave Decay Cylinder Example Runner
================================================

Usage: ./run.sh [command]

Commands:
  (none)       Run full workflow: build, preprocess, simulate, post-process
  build        Build VFS-Wind executables only
  preprocess   Check input files and validate setup
  simulate     Run simulation only (requires previous build)
  postprocess  Post-process results only (requires simulation output)
  clean        Remove all output files
  help         Show this help message

Environment Variables:
  NP_SIM       Number of MPI processes for simulation (default: 8)
  NP_POST      Number of MPI processes for post-processing (default: 1)
  USE_XML      Use XML config file (1) or control.dat (0) (default: 1)
  ENABLE_GPU   Enable GPU acceleration (1=on, 0=off) (default: 0)
  OMP_THREADS  Number of OpenMP threads for GPU backend (default: auto)
  TIS          Starting timestep for post-processing (default: 0)
  TIE          Ending timestep for post-processing (default: 10000)
  TS           Timestep stride for post-processing (default: 1000)
  AVG          Averaging mode: 0=off, 1=Reynolds stresses, 2=TKE (default: 0)

IMPORTANT: Missing Grid File
  This test case requires a grid file (grid.dat) that is not included
  in the repository. You must generate or provide this file before running.

  Grid requirements:
    - 3D structured mesh
    - Two-phase flow capability with level set
    - Sufficient resolution near free surface (dthick = 0.006)

Examples:
  # Check input files (will report missing grid.dat)
  ./run.sh preprocess

  # Run with 16 MPI processes
  NP_SIM=16 ./run.sh simulate

  # Run with GPU enabled
  ENABLE_GPU=1 ./run.sh simulate

  # Post-process specific timestep range
  TIS=0 TIE=10000 TS=500 ./run.sh postprocess

Test Case Description:
  This test case simulates a 3D cylinder undergoing free heave decay
  at a water/air interface. The cylinder is released from an initial
  displacement and oscillates due to the restoring buoyancy force.

  Physical parameters:
    - Reynolds number: 6666.67
    - Gravity: -9.81 m/s^2 (Y-direction)
    - Density ratio: 1000:1 (water:air)
    - Mass ratio: m* = 0.25
    - Initial Y position: 0.0254 m
    - Damping: 0.0 (undamped, decay is due to hydrodynamic damping)

  Expected behavior:
    - Initial displacement from equilibrium
    - Oscillatory motion with decreasing amplitude
    - Decay rate determined by added mass and radiation damping
    - Natural frequency matches analytical prediction

Output Files:
  Simulation:
    - ufield*.dat, pfield*.dat, etc. (PETSc binary format)
    - lfield*.dat (level set field)
    - FSI_position00 (cylinder heave history)

  Reference:
    - _FSI_position00_good_run (reference FSI output for validation)

  Post-processing:
    - Result*.vts (VTK structured grid - open in ParaView)
    - Result*.vtm (VTK multi-block container)

EOF
}

# === MAIN ===
case "${1:-all}" in
    build)
        do_build
        ;;
    preprocess|pre|check)
        do_preprocess
        ;;
    simulate|sim|run)
        do_simulate
        ;;
    postprocess|post|visualize|viz)
        do_postprocess
        ;;
    clean)
        do_clean
        ;;
    help|--help|-h)
        do_help
        ;;
    all|"")
        do_build
        do_preprocess
        do_simulate
        do_postprocess
        print_header "Workflow Complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Open ParaView"
        echo "  2. File -> Open -> Select Result*.vts or Result*.vtm"
        echo "  3. Use 'Contour' filter on Level set field to view interface"
        echo "  4. Animate to see cylinder heave decay"
        echo ""
        echo "  5. Analyze heave motion:"
        echo "     Plot FSI_position00 (column 2 = Y displacement vs time)"
        echo "     Compare with reference: _FSI_position00_good_run"
        echo ""
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './run.sh help' for usage information."
        exit 1
        ;;
esac
