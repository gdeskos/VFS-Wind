/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Kokkos Initialization C Interface for VFS-Wind                 *
 ******************************************************************/

#ifndef VFSWIND_KOKKOS_INIT_H
#define VFSWIND_KOKKOS_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize GPU/Kokkos runtime
 *
 * Call this after MPI_Init but before PetscInitialize.
 *
 * @param argc Pointer to argc (can be NULL)
 * @param argv Pointer to argv (can be NULL)
 * @return 0 on success
 */
int VFSWind_GPU_Initialize(int* argc, char*** argv);

/**
 * @brief Finalize GPU/Kokkos runtime
 *
 * Call this after PetscFinalize but before MPI_Finalize.
 *
 * @return 0 on success
 */
int VFSWind_GPU_Finalize(void);

/**
 * @brief Check if GPU is enabled
 *
 * @return 1 if GPU is enabled, 0 otherwise
 */
int VFSWind_GPU_IsEnabled(void);

/**
 * @brief Get GPU backend name
 *
 * @return String like "CUDA", "HIP", "SYCL", "OpenMP", or "None"
 */
const char* VFSWind_GPU_GetBackendName(void);

/**
 * @brief Synchronize GPU execution
 *
 * Waits for all GPU kernels to complete.
 */
void VFSWind_GPU_Fence(void);

/**
 * @brief Print GPU memory usage
 */
void VFSWind_GPU_PrintMemoryInfo(void);

/**
 * @brief Run GPU smoke test
 *
 * Simple test to verify GPU is working correctly.
 *
 * @return 0 on success, non-zero on failure
 */
int VFSWind_GPU_SmokeTest(void);

#ifdef __cplusplus
}
#endif

#endif /* VFSWIND_KOKKOS_INIT_H */
