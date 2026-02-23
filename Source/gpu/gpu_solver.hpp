/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Solver Configuration for VFS-Wind                          *
 * Phase 2: Linear Solver GPU Acceleration                        *
 ******************************************************************/

#ifndef VFSWIND_GPU_SOLVER_HPP
#define VFSWIND_GPU_SOLVER_HPP

#include "petscksp.h"
#include "petscpc.h"
#include "petscdm.h"

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#endif

namespace vfswind {
namespace gpu {

// ============================================================================
// GPU Backend Detection
// ============================================================================

/**
 * @brief Enumeration of available GPU backends
 */
enum class GPUBackend {
    NONE,       // No GPU support
    CUDA,       // NVIDIA CUDA
    HIP,        // AMD ROCm/HIP
    KOKKOS,     // Kokkos abstraction layer
    SYCL        // Intel SYCL/oneAPI
};

/**
 * @brief Get the active GPU backend based on PETSc configuration
 */
inline GPUBackend getActiveGPUBackend() {
#ifdef ENABLE_GPU
    #if defined(KOKKOS_ENABLE_CUDA)
        return GPUBackend::CUDA;
    #elif defined(KOKKOS_ENABLE_HIP)
        return GPUBackend::HIP;
    #elif defined(KOKKOS_ENABLE_SYCL)
        return GPUBackend::SYCL;
    #else
        return GPUBackend::KOKKOS;  // OpenMP or Serial backend
    #endif
#else
    return GPUBackend::NONE;
#endif
}

/**
 * @brief Get string name of the GPU backend
 */
inline const char* getGPUBackendName(GPUBackend backend) {
    switch (backend) {
        case GPUBackend::CUDA:   return "CUDA";
        case GPUBackend::HIP:    return "HIP/ROCm";
        case GPUBackend::KOKKOS: return "Kokkos (CPU)";
        case GPUBackend::SYCL:   return "SYCL/oneAPI";
        default:                 return "None";
    }
}

/**
 * @brief Check if PETSc was built with GPU support
 */
inline PetscBool hasPetscGPUSupport() {
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP) || defined(PETSC_HAVE_KOKKOS)
    return PETSC_TRUE;
#else
    return PETSC_FALSE;
#endif
}

/**
 * @brief Check if HYPRE was built with GPU support
 */
inline PetscBool hasHypreGPUSupport() {
#if defined(PETSC_HAVE_HYPRE) && (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP))
    return PETSC_TRUE;
#else
    return PETSC_FALSE;
#endif
}

// ============================================================================
// GPU Solver Configuration
// ============================================================================

/**
 * @brief Configuration options for GPU solver
 */
struct GPUSolverConfig {
    // Vector type for GPU
    const char* vec_type;

    // Matrix type for GPU
    const char* mat_type;

    // Preconditioner options
    PetscBool use_hypre_gpu;
    PetscInt hypre_device_level;

    // Memory options
    PetscBool use_unified_memory;

    // Default constructor
    GPUSolverConfig() {
#ifdef ENABLE_GPU
    #if defined(KOKKOS_ENABLE_CUDA)
        vec_type = VECCUDA;
        mat_type = MATAIJCUSPARSE;
        use_hypre_gpu = PETSC_TRUE;
        hypre_device_level = 1;
    #elif defined(KOKKOS_ENABLE_HIP)
        vec_type = VECHIP;
        mat_type = MATAIJHIPSPARSE;
        use_hypre_gpu = PETSC_TRUE;
        hypre_device_level = 1;
    #elif defined(PETSC_HAVE_KOKKOS)
        vec_type = VECKOKKOS;
        mat_type = MATAIJKOKKOS;
        use_hypre_gpu = PETSC_FALSE;
        hypre_device_level = 0;
    #else
        vec_type = VECSTANDARD;
        mat_type = MATAIJ;
        use_hypre_gpu = PETSC_FALSE;
        hypre_device_level = 0;
    #endif
#else
        vec_type = VECSTANDARD;
        mat_type = MATAIJ;
        use_hypre_gpu = PETSC_FALSE;
        hypre_device_level = 0;
#endif
        use_unified_memory = PETSC_FALSE;
    }
};

/**
 * @brief Apply GPU configuration to PETSc runtime options
 *
 * This function inserts PETSc options for GPU-accelerated solving.
 * Call this before KSPCreate/KSPSetFromOptions.
 *
 * @param config GPU solver configuration
 * @return PetscErrorCode
 */
inline PetscErrorCode applyGPUSolverOptions(const GPUSolverConfig& config) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    GPUBackend backend = getActiveGPUBackend();

    if (backend == GPUBackend::CUDA) {
        // CUDA-specific options
        PetscOptionsInsertString(NULL, "-vec_type cuda");
        PetscOptionsInsertString(NULL, "-mat_type aijcusparse");
        PetscOptionsInsertString(NULL, "-dm_vec_type cuda");
        PetscOptionsInsertString(NULL, "-dm_mat_type aijcusparse");

#ifdef PETSC_HAVE_HYPRE
        if (config.use_hypre_gpu) {
            // HYPRE GPU options for CUDA
            char opt[256];
            snprintf(opt, sizeof(opt),
                     "-pc_hypre_boomeramg_device_level %d",
                     (int)config.hypre_device_level);
            PetscOptionsInsertString(NULL, opt);

            // GPU-optimized coarsening and interpolation
            PetscOptionsInsertString(NULL, "-pc_hypre_boomeramg_coarsen_type PMIS");
            PetscOptionsInsertString(NULL, "-pc_hypre_boomeramg_interp_type ext+i");
            PetscOptionsInsertString(NULL, "-pc_hypre_boomeramg_relax_type_all l1scaled-SOR/Jacobi");
        }
#endif
    }
    else if (backend == GPUBackend::HIP) {
        // HIP/ROCm-specific options
        PetscOptionsInsertString(NULL, "-vec_type hip");
        PetscOptionsInsertString(NULL, "-mat_type aijhipsparse");
        PetscOptionsInsertString(NULL, "-dm_vec_type hip");
        PetscOptionsInsertString(NULL, "-dm_mat_type aijhipsparse");

#ifdef PETSC_HAVE_HYPRE
        if (config.use_hypre_gpu) {
            char opt[256];
            snprintf(opt, sizeof(opt),
                     "-pc_hypre_boomeramg_device_level %d",
                     (int)config.hypre_device_level);
            PetscOptionsInsertString(NULL, opt);

            PetscOptionsInsertString(NULL, "-pc_hypre_boomeramg_coarsen_type PMIS");
            PetscOptionsInsertString(NULL, "-pc_hypre_boomeramg_interp_type ext+i");
        }
#endif
    }
    else if (backend == GPUBackend::KOKKOS) {
        // Kokkos backend (OpenMP or Serial)
        // No GPU-specific options needed
        PetscPrintf(PETSC_COMM_WORLD,
                    "VFS-Wind: Using Kokkos CPU backend (no GPU vectors)\n");
    }
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Configure a KSP solver for GPU execution
 *
 * @param ksp KSP solver context
 * @param config GPU solver configuration
 * @return PetscErrorCode
 */
inline PetscErrorCode configureKSPForGPU(KSP ksp, const GPUSolverConfig& config) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    GPUBackend backend = getActiveGPUBackend();

    if (backend == GPUBackend::CUDA || backend == GPUBackend::HIP) {
        PC pc;
        PetscCall(KSPGetPC(ksp, &pc));

#ifdef PETSC_HAVE_HYPRE
        // Get preconditioner type
        PCType pctype;
        PetscCall(PCGetType(pc, &pctype));

        if (pctype && strcmp(pctype, PCHYPRE) == 0) {
            // Configure HYPRE for GPU
            if (config.use_hypre_gpu) {
                // Set device level (1 = fine grid on GPU)
                char opt[256];
                snprintf(opt, sizeof(opt),
                         "-pc_hypre_boomeramg_device_level %d",
                         (int)config.hypre_device_level);
                PetscOptionsInsertString(NULL, opt);
            }
        }
#endif
    }
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Configure a DM (distributed mesh) for GPU vectors/matrices
 *
 * @param dm PETSc DM object
 * @param config GPU solver configuration
 * @return PetscErrorCode
 */
inline PetscErrorCode configureDMForGPU(DM dm, const GPUSolverConfig& config) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    GPUBackend backend = getActiveGPUBackend();

    if (backend == GPUBackend::CUDA) {
#ifdef PETSC_HAVE_CUDA
        PetscCall(DMSetVecType(dm, VECCUDA));
        PetscCall(DMSetMatType(dm, MATAIJCUSPARSE));
#endif
    }
    else if (backend == GPUBackend::HIP) {
#ifdef PETSC_HAVE_HIP
        PetscCall(DMSetVecType(dm, VECHIP));
        PetscCall(DMSetMatType(dm, MATAIJHIPSPARSE));
#endif
    }
#ifdef PETSC_HAVE_KOKKOS
    else if (backend == GPUBackend::KOKKOS) {
        PetscCall(DMSetVecType(dm, VECKOKKOS));
        PetscCall(DMSetMatType(dm, MATAIJKOKKOS));
    }
#endif
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ============================================================================
// GPU Solver Diagnostics
// ============================================================================

/**
 * @brief Print GPU solver configuration to stdout
 */
inline PetscErrorCode printGPUSolverInfo() {
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_WORLD,
                "==============================================================\n"
                "VFS-Wind GPU Solver Configuration\n"
                "==============================================================\n");

#ifdef ENABLE_GPU
    GPUBackend backend = getActiveGPUBackend();
    PetscPrintf(PETSC_COMM_WORLD, "  GPU Backend:        %s\n",
                getGPUBackendName(backend));
#else
    PetscPrintf(PETSC_COMM_WORLD, "  GPU Backend:        Disabled (CPU only)\n");
#endif

    PetscPrintf(PETSC_COMM_WORLD, "  PETSc GPU Support:  %s\n",
                hasPetscGPUSupport() ? "Yes" : "No");

#ifdef PETSC_HAVE_CUDA
    PetscPrintf(PETSC_COMM_WORLD, "  PETSc CUDA:         Yes\n");
#else
    PetscPrintf(PETSC_COMM_WORLD, "  PETSc CUDA:         No\n");
#endif

#ifdef PETSC_HAVE_HIP
    PetscPrintf(PETSC_COMM_WORLD, "  PETSc HIP:          Yes\n");
#else
    PetscPrintf(PETSC_COMM_WORLD, "  PETSc HIP:          No\n");
#endif

#ifdef PETSC_HAVE_KOKKOS
    PetscPrintf(PETSC_COMM_WORLD, "  PETSc Kokkos:       Yes\n");
#else
    PetscPrintf(PETSC_COMM_WORLD, "  PETSc Kokkos:       No\n");
#endif

#ifdef PETSC_HAVE_HYPRE
    PetscPrintf(PETSC_COMM_WORLD, "  HYPRE:              Yes\n");
    PetscPrintf(PETSC_COMM_WORLD, "  HYPRE GPU Support:  %s\n",
                hasHypreGPUSupport() ? "Yes" : "No");
#else
    PetscPrintf(PETSC_COMM_WORLD, "  HYPRE:              No\n");
#endif

    PetscPrintf(PETSC_COMM_WORLD,
                "==============================================================\n");

    PetscFunctionReturn(PETSC_SUCCESS);
}

/**
 * @brief Print recommended GPU runtime options
 */
inline PetscErrorCode printGPURuntimeHelp() {
    PetscFunctionBeginUser;

    PetscPrintf(PETSC_COMM_WORLD,
        "VFS-Wind GPU Runtime Options:\n"
        "\n"
        "NVIDIA CUDA:\n"
        "  mpirun -np 4 ./vwis -xml control.xml \\\n"
        "      -vec_type cuda \\\n"
        "      -mat_type aijcusparse \\\n"
        "      -pc_type hypre \\\n"
        "      -pc_hypre_boomeramg_device_level 1\n"
        "\n"
        "AMD HIP/ROCm:\n"
        "  mpirun -np 4 ./vwis -xml control.xml \\\n"
        "      -vec_type hip \\\n"
        "      -mat_type aijhipsparse \\\n"
        "      -pc_type hypre\n"
        "\n"
        "Kokkos (auto-detect):\n"
        "  mpirun -np 4 ./vwis -xml control.xml \\\n"
        "      -vec_type kokkos \\\n"
        "      -mat_type aijkokkos\n"
        "\n"
        "Environment Variables:\n"
        "  CUDA_VISIBLE_DEVICES=0,1  # Select NVIDIA GPUs\n"
        "  HIP_VISIBLE_DEVICES=0,1   # Select AMD GPUs\n"
        "  KOKKOS_NUM_THREADS=1      # Usually 1 for GPU\n"
        "\n"
    );

    PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace gpu
} // namespace vfswind

#endif // VFSWIND_GPU_SOLVER_HPP
