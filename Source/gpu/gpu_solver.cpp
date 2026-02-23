/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Solver Configuration - Implementation                      *
 * Phase 2: Linear Solver GPU Acceleration                        *
 ******************************************************************/

#include "gpu_solver.h"

#ifdef ENABLE_GPU
#include "gpu_solver.hpp"
#endif

extern "C" {

PetscErrorCode VFSWind_ApplyGPUSolverOptions(void) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    vfswind::gpu::GPUSolverConfig config;
    PetscCall(vfswind::gpu::applyGPUSolverOptions(config));
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VFSWind_ConfigureKSPForGPU(KSP ksp) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    vfswind::gpu::GPUSolverConfig config;
    PetscCall(vfswind::gpu::configureKSPForGPU(ksp, config));
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VFSWind_ConfigureDMForGPU(DM dm) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    vfswind::gpu::GPUSolverConfig config;
    PetscCall(vfswind::gpu::configureDMForGPU(dm, config));
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VFSWind_PrintGPUSolverInfo(void) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    PetscCall(vfswind::gpu::printGPUSolverInfo());
#else
    PetscPrintf(PETSC_COMM_WORLD,
                "==============================================================\n"
                "VFS-Wind GPU Solver Configuration\n"
                "==============================================================\n"
                "  GPU Support:        Disabled (not compiled with -DENABLE_GPU)\n"
                "==============================================================\n");
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VFSWind_PrintGPURuntimeHelp(void) {
    PetscFunctionBeginUser;

#ifdef ENABLE_GPU
    PetscCall(vfswind::gpu::printGPURuntimeHelp());
#else
    PetscPrintf(PETSC_COMM_WORLD,
                "GPU runtime options not available (compile with -DENABLE_GPU=ON)\n");
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscBool VFSWind_HasGPUSolverSupport(void) {
#ifdef ENABLE_GPU
    vfswind::gpu::GPUBackend backend = vfswind::gpu::getActiveGPUBackend();
    return (backend != vfswind::gpu::GPUBackend::NONE) ? PETSC_TRUE : PETSC_FALSE;
#else
    return PETSC_FALSE;
#endif
}

const char* VFSWind_GetGPUBackendName(void) {
#ifdef ENABLE_GPU
    vfswind::gpu::GPUBackend backend = vfswind::gpu::getActiveGPUBackend();
    return vfswind::gpu::getGPUBackendName(backend);
#else
    return "None (GPU disabled)";
#endif
}

} // extern "C"
