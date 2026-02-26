#!/bin/bash
#=============================================================================
# VFS-Wind 3D Sloshing Example Runner
# Test Case: Two-Phase Sloshing Flow with Level Set Method
#
# Usage:
#   ./run.sh              # Run full workflow (build, simulate, post-process)
#   ./run.sh build        # Build VFS-Wind only
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
NP_SIM=${NP_SIM:-4}

# Number of MPI processes for post-processing (usually 1)
NP_POST=${NP_POST:-1}

# Use XML config (1) or legacy control.dat (0)
USE_XML=${USE_XML:-1}

# Enable GPU acceleration (1=on, 0=off) - uses Kokkos for portable GPU kernels
ENABLE_GPU=${ENABLE_GPU:-1}

# Number of OpenMP threads for GPU backend (0=auto-detect)
OMP_THREADS=${OMP_THREADS:-0}

# Timestep range for post-processing
TIS=${TIS:-0}        # Starting timestep
TIE=${TIE:-100}      # Ending timestep
TS=${TS:-10}         # Timestep stride

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

# === SIMULATION FUNCTION ===
do_simulate() {
    print_header "Running 3D Sloshing Simulation"

    check_executable "${BUILD_DIR}/Source/vwis"

    cd "${SCRIPT_DIR}"

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
        # Auto-detect: use number of performance cores
        export OMP_NUM_THREADS=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)
    fi

    print_step "Starting simulation with ${NP_SIM} MPI processes..."
    echo "  Grid:       xyz.dat (200 x 41 x 200)"
    echo "  Re:         6666.67"
    echo "  Physics:    Two-phase flow with Level Set"
    echo "  Sloshing:   Mode 2"
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
    # Note: levelset=1 enables level set field output
    POST_CMD="${BUILD_DIR}/Source/data -tis ${TIS} -tie ${TIE} -ts ${TS} -vtk 1 -xyz 1 -binary 0 -levelset 1"

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
    rm -f ufield*.dat vfield*.dat pfield*.dat nvfield*.dat
    rm -f su0_*.dat su1_*.dat su2_*.dat sp_*.dat
    rm -f kfield*.dat lfield*.dat qfield*.dat
    rm -f levelset*.dat
    rm -f *.info

    print_step "Removing VTK files..."
    rm -f Result*.vts Result*.vtm Result*.plt

    print_step "Removing log files..."
    rm -f err* output*.log

    echo "Clean completed!"
}

# === HELP FUNCTION ===
do_help() {
    cat << EOF
VFS-Wind 3D Sloshing Example Runner
=====================================

Usage: ./run.sh [command]

Commands:
  (none)       Run full workflow: build, simulate, post-process
  build        Build VFS-Wind executables only
  simulate     Run simulation only (requires previous build)
  postprocess  Post-process results only (requires simulation output)
  clean        Remove all output files
  help         Show this help message

Environment Variables:
  NP_SIM       Number of MPI processes for simulation (default: 4)
  NP_POST      Number of MPI processes for post-processing (default: 1)
  USE_XML      Use XML config file (1) or control.dat (0) (default: 1)
  ENABLE_GPU   Enable GPU acceleration (1=on, 0=off) (default: 1)
  OMP_THREADS  Number of OpenMP threads for GPU backend (default: auto)
  TIS          Starting timestep for post-processing (default: 0)
  TIE          Ending timestep for post-processing (default: 100)
  TS           Timestep stride for post-processing (default: 10)
  AVG          Averaging mode: 0=off, 1=Reynolds stresses, 2=TKE (default: 0)

Examples:
  # Run with 8 MPI processes (GPU enabled by default)
  NP_SIM=8 ./run.sh simulate

  # Run with GPU disabled (CPU only)
  ENABLE_GPU=0 ./run.sh simulate

  # Run with specific number of OpenMP threads
  OMP_THREADS=8 ./run.sh simulate

  # Use legacy control.dat instead of XML
  USE_XML=0 ./run.sh simulate

  # Post-process specific timestep range
  TIS=50 TIE=100 TS=5 ./run.sh postprocess

  # Quick test run
  NP_SIM=2 ./run.sh

Test Case Description:
  This test case simulates 3D sloshing in a rectangular tank using the
  Level Set method for two-phase flow. The simulation captures the
  free surface motion of water sloshing in a container.

  Physical parameters:
    - Water density: 1000 kg/m^3
    - Air density: 1 kg/m^3
    - Water viscosity: 1.0e-3 Pa.s
    - Air viscosity: 1.8e-5 Pa.s
    - Gravity: -9.8 m/s^2 (y-direction)

Output Files:
  Simulation:
    - ufield*.dat, pfield*.dat, etc. (PETSc binary format)
    - lfield*.dat (level set field)

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
    simulate)
        do_simulate
        ;;
    postprocess|post)
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
        echo "  3. Apply filters to visualize velocity, pressure, level set"
        echo "  4. Use 'Contour' filter on Level set field to view free surface"
        echo ""
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run './run.sh help' for usage information."
        exit 1
        ;;
esac
