/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Kokkos Initialization/Finalization for VFS-Wind                *
 ******************************************************************/

#include "gpu_config.hpp"

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#endif

#include <cstdio>

// Make functions callable from C code
extern "C" {

/**
 * @brief Initialize Kokkos runtime
 *
 * Should be called after MPI_Init and before PetscInitialize.
 * Can be called with argc/argv to pass Kokkos command-line options.
 *
 * @param argc Pointer to argument count (can be NULL)
 * @param argv Pointer to argument vector (can be NULL)
 * @return 0 on success, non-zero on failure
 */
int VFSWind_GPU_Initialize(int* argc, char*** argv) {
#ifdef ENABLE_GPU
    try {
        if (argc != nullptr && argv != nullptr) {
            Kokkos::initialize(*argc, *argv);
        } else {
            Kokkos::initialize();
        }

        // Print GPU information
        printf("\n");
        printf("==============================================================\n");
        printf("VFS-Wind GPU Acceleration Enabled\n");
        printf("==============================================================\n");
        printf("  Kokkos Version:     %d.%d.%d\n",
               KOKKOS_VERSION / 10000,
               (KOKKOS_VERSION % 10000) / 100,
               KOKKOS_VERSION % 100);
        printf("  Backend:            %s\n", getBackendName());
        printf("  Execution Space:    %s\n",
               typeid(Kokkos::DefaultExecutionSpace).name());
        printf("  Memory Space:       %s\n",
               typeid(Kokkos::DefaultExecutionSpace::memory_space).name());

#if defined(KOKKOS_ENABLE_CUDA)
        printf("  CUDA Device:        %d\n", Kokkos::Cuda().cuda_device());
#elif defined(KOKKOS_ENABLE_HIP)
        printf("  HIP Device:         %d\n", Kokkos::HIP().hip_device());
#endif

        printf("==============================================================\n");
        printf("\n");

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Kokkos initialization failed: %s\n", e.what());
        return 1;
    }
#else
    // GPU not enabled, just print info message
    printf("\n");
    printf("VFS-Wind: GPU acceleration not enabled (compiled without -DENABLE_GPU)\n");
    printf("\n");
    return 0;
#endif
}

/**
 * @brief Finalize Kokkos runtime
 *
 * Should be called before MPI_Finalize and after PetscFinalize.
 *
 * @return 0 on success, non-zero on failure
 */
int VFSWind_GPU_Finalize(void) {
#ifdef ENABLE_GPU
    try {
        Kokkos::finalize();
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Kokkos finalization failed: %s\n", e.what());
        return 1;
    }
#else
    return 0;
#endif
}

/**
 * @brief Check if GPU is available and enabled
 *
 * @return 1 if GPU is enabled, 0 otherwise
 */
int VFSWind_GPU_IsEnabled(void) {
#ifdef ENABLE_GPU
    return isGPUEnabled() ? 1 : 0;
#else
    return 0;
#endif
}

/**
 * @brief Get GPU backend name
 *
 * @return String describing the GPU backend (e.g., "CUDA", "HIP", "None")
 */
const char* VFSWind_GPU_GetBackendName(void) {
    return getBackendName();
}

/**
 * @brief Synchronize GPU (wait for all kernels to complete)
 *
 * Useful for timing and debugging. In production, this is typically
 * not needed as Kokkos handles synchronization automatically.
 */
void VFSWind_GPU_Fence(void) {
#ifdef ENABLE_GPU
    Kokkos::fence();
#endif
}

/**
 * @brief Print GPU memory usage information
 *
 * Useful for debugging memory issues.
 */
void VFSWind_GPU_PrintMemoryInfo(void) {
#ifdef ENABLE_GPU
#if defined(KOKKOS_ENABLE_CUDA)
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.2f GB free / %.2f GB total\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));
#elif defined(KOKKOS_ENABLE_HIP)
    size_t free_mem, total_mem;
    hipMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory: %.2f GB free / %.2f GB total\n",
           free_mem / (1024.0 * 1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0 * 1024.0));
#else
    printf("GPU Memory info not available for this backend\n");
#endif
#else
    printf("GPU not enabled\n");
#endif
}

} // extern "C"

// ============================================================================
// Smoke Test Function (for validation)
// ============================================================================

#ifdef ENABLE_GPU

/**
 * @brief Simple smoke test to verify GPU functionality
 *
 * Creates a small array, fills it on GPU, and verifies the result.
 *
 * @return 0 on success, non-zero on failure
 */
extern "C" int VFSWind_GPU_SmokeTest(void) {
    const int N = 1000;
    int errors = 0;

    try {
        printf("Running GPU smoke test...\n");

        // Create device view
        Kokkos::View<double*, DeviceMemorySpace> d_array("test_array", N);

        // Fill on device
        Kokkos::parallel_for("smoke_test_fill",
            Kokkos::RangePolicy<DefaultExecutionSpace>(0, N),
            KOKKOS_LAMBDA(int i) {
                d_array(i) = static_cast<double>(i * i);
            }
        );

        // Copy to host for verification
        auto h_array = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), d_array);

        // Verify results
        for (int i = 0; i < N; ++i) {
            double expected = static_cast<double>(i * i);
            if (h_array(i) != expected) {
                errors++;
                if (errors <= 5) {
                    printf("  ERROR at i=%d: expected %.1f, got %.1f\n",
                           i, expected, h_array(i));
                }
            }
        }

        if (errors == 0) {
            printf("GPU smoke test PASSED (%d elements verified)\n", N);
        } else {
            printf("GPU smoke test FAILED (%d errors)\n", errors);
        }

        return errors;

    } catch (const std::exception& e) {
        fprintf(stderr, "GPU smoke test exception: %s\n", e.what());
        return 1;
    }
}

#else

extern "C" int VFSWind_GPU_SmokeTest(void) {
    printf("GPU smoke test skipped (GPU not enabled)\n");
    return 0;
}

#endif // ENABLE_GPU
