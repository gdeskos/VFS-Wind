#!/bin/bash
#=============================================================================
# VFS-Wind Vortex-Induced Vibration (VIV) Example Runner
# Test Case: Mounted Cylinder with Fluid-Structure Interaction
#
# Usage:
#   ./run.sh              # Run full workflow (build, simulate, post-process)
#   ./run.sh build        # Build VFS-Wind only
#   ./run.sh simulate     # Run simulation only
#   ./run.sh postprocess  # Post-process results only
#   ./run.sh visualize    # Alias for postprocess
#   ./run.sh clean        # Clean output files
#   ./run.sh help         # Show this help message
#
# Configuration:
#   Edit the variables below to customize the run
#=============================================================================

# === CONFIGURATION ===
# Number of MPI processes for simulation
NP_SIM=${NP_SIM:-4}

# Number of MPI processes for post-processing (usually 1)
NP_POST=${NP_POST:-1}

# Use XML config (1) or legacy control.dat (0)
USE_XML=${USE_XML:-0}

# Timestep range for post-processing
TIS=${TIS:-0}         # Starting timestep
TIE=${TIE:-6000}      # Ending timestep
TS=${TS:-500}         # Timestep stride

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
    cmake -B build -DCMAKE_BUILD_TYPE=Release

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

# === SIMULATION FUNCTION ===
do_simulate() {
    print_header "Running VIV Mounted Cylinder Simulation"

    check_executable "${BUILD_DIR}/Source/vwis"

    cd "${SCRIPT_DIR}"

    # Check required files
    if [ ! -f "grid.dat" ]; then
        echo "ERROR: grid.dat not found!"
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

    print_step "Starting simulation with ${NP_SIM} MPI processes..."
    echo "  Physics:    Vortex-Induced Vibration (VIV)"
    echo "  Re:         150"
    echo "  U_red:      6 (reduced velocity)"
    echo "  Mass ratio: 0.25"
    echo "  DOF:        Y-direction only (cross-flow)"
    echo "  Timesteps:  6000 (dt=0.01)"
    echo ""

    mpirun -np ${NP_SIM} "${BUILD_DIR}/Source/vwis" ${CONFIG_ARGS}

    if [ $? -ne 0 ]; then
        echo "ERROR: Simulation failed!"
        exit 1
    fi

    print_step "Simulation completed successfully!"

    # Check for FSI output
    if [ -f "FSI_DATA00" ] || [ -f "fsi_data00.dat" ]; then
        echo "FSI displacement data saved."
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
    # Note: IBM case, no levelset
    POST_CMD="${BUILD_DIR}/Source/data -tis ${TIS} -tie ${TIE} -ts ${TS} -vtk 1 -xyz 1 -binary 0 -levelset 0"

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
}

# === CLEAN FUNCTION ===
do_clean() {
    print_header "Cleaning Output Files"

    cd "${SCRIPT_DIR}"

    print_step "Removing binary output files..."
    rm -f ufield*.dat vfield*.dat wfield*.dat pfield*.dat nvfield*.dat
    rm -f su0_*.dat su1_*.dat su2_*.dat sp_*.dat
    rm -f kfield*.dat ofield*.dat qfield*.dat
    rm -f cs_*.dat
    rm -f *.info

    print_step "Removing VTK files..."
    rm -f Result*.vts Result*.vtm Result*.plt

    print_step "Removing FSI output files..."
    rm -f FSI_DATA* fsi_data*.dat
    rm -f Converge_* Kinetic_Energy.dat

    print_step "Removing log files..."
    rm -f err* output*.log *.log

    echo "Clean completed!"
}

# === HELP FUNCTION ===
do_help() {
    cat << EOF
VFS-Wind VIV Mounted Cylinder Example Runner
===============================================

Usage: ./run.sh [command]

Commands:
  (none)       Run full workflow: build, simulate, post-process
  build        Build VFS-Wind executables only
  simulate     Run simulation only (requires previous build)
  postprocess  Post-process results only (requires simulation output)
  visualize    Alias for postprocess
  clean        Remove all output files
  help         Show this help message

Environment Variables:
  NP_SIM       Number of MPI processes for simulation (default: 4)
  NP_POST      Number of MPI processes for post-processing (default: 1)
  USE_XML      Use XML config file (1) or control.dat (0) (default: 0)
  TIS          Starting timestep for post-processing (default: 0)
  TIE          Ending timestep for post-processing (default: 6000)
  TS           Timestep stride for post-processing (default: 500)
  AVG          Averaging mode: 0=off, 1=Reynolds stresses, 2=TKE (default: 0)

Examples:
  # Run with 8 MPI processes
  NP_SIM=8 ./run.sh simulate

  # Use XML configuration
  USE_XML=1 ./run.sh simulate

  # Post-process specific timestep range
  TIS=1000 TIE=6000 TS=100 ./run.sh postprocess

  # Quick test run
  NP_SIM=2 ./run.sh

Test Case Description:
  This test case simulates Vortex-Induced Vibration (VIV) of a cylinder
  mounted on a spring in cross-flow. The cylinder is free to oscillate
  in the Y-direction (cross-flow) due to vortex shedding.

  Physical parameters:
    - Reynolds number: 150 (laminar vortex shedding)
    - Reduced velocity: U_red = 6 (resonance region)
    - Mass ratio: m* = 0.25
    - Damping ratio: zeta = 0.0 (undamped)
    - Cylinder center: (0, 8, 8)

  Expected behavior:
    - Lock-in phenomenon (synchronization of vortex shedding with cylinder motion)
    - Large amplitude oscillations in Y-direction
    - Figure-8 or elliptical cylinder trajectory

Output Files:
  Simulation:
    - ufield*.dat, pfield*.dat, etc. (PETSc binary format)
    - FSI_DATA00 (cylinder displacement history)

  Post-processing:
    - Result*.vts (VTK structured grid - open in ParaView)
    - Result*.vtm (VTK multi-block container)

  FSI Analysis:
    - Plot FSI_DATA00 to visualize cylinder displacement vs time
    - Compute oscillation amplitude and frequency from displacement data

EOF
}

# === MAIN ===
case "${1:-all}" in
    build)
        do_build
        ;;
    simulate)
        do_simulate
        ;;
    postprocess|post|visualize)
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
        do_simulate
        do_postprocess
        print_header "Workflow Complete!"
        echo ""
        echo "Next steps:"
        echo "  1. Open ParaView"
        echo "  2. File -> Open -> Select Result*.vts or Result*.vtm"
        echo "  3. Apply 'Glyph' filter to visualize velocity vectors"
        echo "  4. Use 'Calculator' to compute vorticity"
        echo "  5. Animate to see vortex shedding and cylinder motion"
        echo ""
        echo "  6. Plot FSI_DATA00 to analyze cylinder displacement"
        echo ""
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './run.sh help' for usage information."
        exit 1
        ;;
esac
