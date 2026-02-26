#!/bin/bash
#
# VFS-Wind GPU Benchmark Suite for macOS (Apple Silicon)
#
# This script runs a series of benchmarks to measure GPU kernel performance.
# Designed for Mac M4 with OpenMP backend - uses small grid sizes.
#
# Usage:
#   ./run_benchmarks.sh [options]
#
# Options:
#   --quick     Run minimal benchmarks (fastest)
#   --full      Run all benchmarks including scaling study
#   --clean     Remove previous benchmark results
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# OpenMP settings for Mac
export OMP_PROC_BIND=false
export OMP_PLACES=cores

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
RUN_MODE="standard"
for arg in "$@"; do
    case $arg in
        --quick)
            RUN_MODE="quick"
            shift
            ;;
        --full)
            RUN_MODE="full"
            shift
            ;;
        --clean)
            print_status "Cleaning previous results..."
            rm -rf "$RESULTS_DIR"
            print_status "Done."
            exit 0
            ;;
        --help|-h)
            echo "VFS-Wind GPU Benchmark Suite"
            echo ""
            echo "Usage: ./run_benchmarks.sh [options]"
            echo ""
            echo "Options:"
            echo "  --quick     Run minimal benchmarks (32³, 50 iterations)"
            echo "  --full      Run full scaling study (multiple grids and threads)"
            echo "  --clean     Remove previous benchmark results"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            ;;
    esac
done

# Check build exists
if [ ! -f "$BUILD_DIR/tests/test_gpu_benchmark" ]; then
    print_error "Benchmark executable not found. Building..."
    cd "$PROJECT_ROOT"
    cmake -B build -DENABLE_GPU=ON -DKOKKOS_BACKEND=OPENMP -DENABLE_TESTING=ON \
        -DMPI_C_COMPILER=/opt/homebrew/bin/mpicc \
        -DMPI_CXX_COMPILER=/opt/homebrew/bin/mpicxx 2>&1 | tail -5
    cmake --build build --target test_gpu_benchmark -j$(sysctl -n hw.ncpu) 2>&1 | tail -5
fi

if [ ! -f "$BUILD_DIR/tests/test_gpu_benchmark" ]; then
    print_error "Failed to build benchmark executable"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"
RESULT_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.txt"
CSV_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"

print_header "VFS-Wind GPU Benchmark Suite"
echo "Mode: $RUN_MODE"
echo "Results: $RESULT_FILE"
echo ""

# System info
print_status "Collecting system information..."
{
    echo "========================================"
    echo "SYSTEM INFORMATION"
    echo "========================================"
    echo "Date: $(date)"
    echo "Host: $(hostname)"
    echo "OS: $(uname -s) $(uname -r)"
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')"
    echo "Cores: $(sysctl -n hw.ncpu)"
    echo "Memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
    echo ""
} | tee "$RESULT_FILE"

# Initialize CSV
echo "mode,grid_size,cells,threads,kernel,avg_time_us,throughput_mcells_per_sec" > "$CSV_FILE"

# Run benchmarks based on mode
print_header "Running Benchmarks"

cd "$BUILD_DIR/tests"

case $RUN_MODE in
    quick)
        # Quick mode: single grid size, fewer iterations
        GRID_SIZES="32"
        ITERATIONS=50
        THREAD_COUNTS="4"
        ;;
    full)
        # Full mode: multiple grid sizes and thread counts
        GRID_SIZES="32 48 64 80 96"
        ITERATIONS=100
        THREAD_COUNTS="1 2 4 8"
        ;;
    *)
        # Standard mode
        GRID_SIZES="32 48 64"
        ITERATIONS=100
        THREAD_COUNTS="4"
        ;;
esac

{
    echo ""
    echo "========================================"
    echo "BENCHMARK RESULTS"
    echo "========================================"
    echo "Mode: $RUN_MODE"
    echo "Grid sizes: $GRID_SIZES"
    echo "Thread counts: $THREAD_COUNTS"
    echo "Iterations per test: $ITERATIONS"
    echo ""
} >> "$RESULT_FILE"

for threads in $THREAD_COUNTS; do
    export OMP_NUM_THREADS=$threads

    for grid in $GRID_SIZES; do
        print_status "Grid: ${grid}³, Threads: $threads, Iterations: $ITERATIONS"

        VFSWIND_TIMING=1 ./test_gpu_benchmark \
            --grid_size=$grid \
            --iterations=$ITERATIONS 2>&1 | tee -a "$RESULT_FILE"

        echo "" >> "$RESULT_FILE"
    done
done

# Thread scaling study (for full mode)
if [ "$RUN_MODE" == "full" ]; then
    print_header "Thread Scaling Study (64³ grid)"

    SCALING_FILE="$RESULTS_DIR/scaling_${TIMESTAMP}.csv"
    echo "threads,kernel,avg_time_us" > "$SCALING_FILE"

    {
        echo ""
        echo "========================================"
        echo "THREAD SCALING STUDY (64³ grid)"
        echo "========================================"
        echo ""
    } >> "$RESULT_FILE"

    for threads in 1 2 4 6 8; do
        export OMP_NUM_THREADS=$threads
        print_status "Testing with $threads threads..."

        output=$(VFSWIND_TIMING=1 ./test_gpu_benchmark \
            --grid_size=64 \
            --iterations=100 2>&1)

        echo "$output" >> "$RESULT_FILE"
        echo "" >> "$RESULT_FILE"

        # Extract convection time for scaling analysis
        conv_time=$(echo "$output" | grep "Convection " | awk '{print $4}')
        if [ -n "$conv_time" ]; then
            echo "$threads,Convection,$conv_time" >> "$SCALING_FILE"
        fi
    done

    print_status "Scaling results saved to: $SCALING_FILE"
fi

# Summary
print_header "Benchmark Complete"

echo ""
echo "Results saved to:"
echo "  Full output: $RESULT_FILE"
echo "  CSV data:    $CSV_FILE"
if [ "$RUN_MODE" == "full" ]; then
    echo "  Scaling:     $RESULTS_DIR/scaling_${TIMESTAMP}.csv"
fi
echo ""

# Generate quick summary
{
    echo ""
    echo "========================================"
    echo "SUMMARY"
    echo "========================================"
    echo ""
    echo "Benchmark completed at: $(date)"
    echo "Mode: $RUN_MODE"
    echo "Grid sizes tested: $GRID_SIZES"
    echo "Thread counts: $THREAD_COUNTS"
    echo ""
} >> "$RESULT_FILE"

print_status "Done!"
