/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Solver Configuration - C Interface                         *
 * Phase 2: Linear Solver GPU Acceleration                        *
 ******************************************************************/

#ifndef VFSWIND_GPU_SOLVER_H
#define VFSWIND_GPU_SOLVER_H

#include "petscksp.h"
#include "petscpc.h"
#include "petscdm.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Apply GPU solver options to PETSc runtime
 *
 * Call this early in initialization (after PetscInitialize, before solver setup)
 * to configure PETSc for GPU execution.
 *
 * @return PetscErrorCode
 */
PetscErrorCode VFSWind_ApplyGPUSolverOptions(void);

/**
 * @brief Configure a KSP solver for GPU execution
 *
 * @param ksp KSP solver context
 * @return PetscErrorCode
 */
PetscErrorCode VFSWind_ConfigureKSPForGPU(KSP ksp);

/**
 * @brief Configure a DM for GPU vectors/matrices
 *
 * @param dm PETSc DM object
 * @return PetscErrorCode
 */
PetscErrorCode VFSWind_ConfigureDMForGPU(DM dm);

/**
 * @brief Print GPU solver configuration information
 *
 * @return PetscErrorCode
 */
PetscErrorCode VFSWind_PrintGPUSolverInfo(void);

/**
 * @brief Print GPU runtime help message
 *
 * @return PetscErrorCode
 */
PetscErrorCode VFSWind_PrintGPURuntimeHelp(void);

/**
 * @brief Check if GPU solver support is available
 *
 * @return PETSC_TRUE if GPU support available, PETSC_FALSE otherwise
 */
PetscBool VFSWind_HasGPUSolverSupport(void);

/**
 * @brief Get the name of the active GPU backend
 *
 * @return String name of the backend (e.g., "CUDA", "HIP", "None")
 */
const char* VFSWind_GetGPUBackendName(void);

#ifdef __cplusplus
}
#endif

#endif /* VFSWIND_GPU_SOLVER_H */
