/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Solver Configuration Tests                                 *
 * Phase 2: Linear Solver GPU Acceleration                        *
 ******************************************************************/

#include <gtest/gtest.h>
#include <petscksp.h>
#include <petscdmda.h>

#ifdef ENABLE_GPU
#include "gpu_solver.hpp"
#include "gpu_solver.h"
#endif

class GPUSolverTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize PETSc (only once for all tests)
        PetscInitializeNoArguments();
    }

    static void TearDownTestSuite() {
        // Finalize PETSc
        PetscFinalize();
    }
};

// Test: GPU backend detection
TEST_F(GPUSolverTest, BackendDetection) {
#ifdef ENABLE_GPU
    vfswind::gpu::GPUBackend backend = vfswind::gpu::getActiveGPUBackend();

    // Ensure we get a valid backend
    EXPECT_TRUE(
        backend == vfswind::gpu::GPUBackend::NONE ||
        backend == vfswind::gpu::GPUBackend::CUDA ||
        backend == vfswind::gpu::GPUBackend::HIP ||
        backend == vfswind::gpu::GPUBackend::KOKKOS ||
        backend == vfswind::gpu::GPUBackend::SYCL
    );

    // Get backend name
    const char* name = vfswind::gpu::getGPUBackendName(backend);
    EXPECT_NE(name, nullptr);
    EXPECT_GT(strlen(name), 0);

    std::cout << "Active GPU backend: " << name << std::endl;
#else
    // GPU disabled - test should pass trivially
    SUCCEED();
#endif
}

// Test: GPU solver config default construction
TEST_F(GPUSolverTest, DefaultConfig) {
#ifdef ENABLE_GPU
    vfswind::gpu::GPUSolverConfig config;

    // Config should have valid type strings
    EXPECT_NE(config.vec_type, nullptr);
    EXPECT_NE(config.mat_type, nullptr);

    std::cout << "Default vec_type: " << config.vec_type << std::endl;
    std::cout << "Default mat_type: " << config.mat_type << std::endl;
#else
    SUCCEED();
#endif
}

// Test: C interface - print info
TEST_F(GPUSolverTest, CInterfacePrintInfo) {
#ifdef ENABLE_GPU
    PetscErrorCode ierr = VFSWind_PrintGPUSolverInfo();
    EXPECT_EQ(ierr, 0);
#else
    SUCCEED();
#endif
}

// Test: C interface - has GPU support
TEST_F(GPUSolverTest, CInterfaceHasSupport) {
#ifdef ENABLE_GPU
    PetscBool has_support = VFSWind_HasGPUSolverSupport();
    // Should return TRUE or FALSE (valid result)
    EXPECT_TRUE(has_support == PETSC_TRUE || has_support == PETSC_FALSE);

    const char* backend_name = VFSWind_GetGPUBackendName();
    EXPECT_NE(backend_name, nullptr);
    std::cout << "C interface backend: " << backend_name << std::endl;
#else
    SUCCEED();
#endif
}

// Test: Apply GPU options (should not crash)
TEST_F(GPUSolverTest, ApplyOptions) {
#ifdef ENABLE_GPU
    PetscErrorCode ierr = VFSWind_ApplyGPUSolverOptions();
    EXPECT_EQ(ierr, 0);
#else
    SUCCEED();
#endif
}

// Test: Configure DM for GPU
TEST_F(GPUSolverTest, ConfigureDM) {
#ifdef ENABLE_GPU
    // Create a simple 3D DMDA
    DM dm;
    PetscErrorCode ierr;

    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        8, 8, 8,
                        PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        1, 1,
                        NULL, NULL, NULL,
                        &dm);
    ASSERT_EQ(ierr, 0);

    // Configure for GPU (must be done before DMSetUp)
    ierr = VFSWind_ConfigureDMForGPU(dm);
    EXPECT_EQ(ierr, 0);

    // Set up DM
    ierr = DMSetUp(dm);
    EXPECT_EQ(ierr, 0);

    // Clean up
    ierr = DMDestroy(&dm);
    EXPECT_EQ(ierr, 0);
#else
    SUCCEED();
#endif
}

// Test: Configure KSP for GPU
TEST_F(GPUSolverTest, ConfigureKSP) {
#ifdef ENABLE_GPU
    // Create a simple 3D DMDA
    DM dm;
    PetscErrorCode ierr;

    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        8, 8, 8,
                        PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        1, 1,
                        NULL, NULL, NULL,
                        &dm);
    ASSERT_EQ(ierr, 0);

    // Must set up DM before creating matrix
    ierr = DMSetUp(dm);
    ASSERT_EQ(ierr, 0);

    // Create a simple matrix
    Mat A;
    ierr = DMCreateMatrix(dm, &A);
    ASSERT_EQ(ierr, 0);

    // Create KSP
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    ASSERT_EQ(ierr, 0);

    ierr = KSPSetOperators(ksp, A, A);
    EXPECT_EQ(ierr, 0);

    // Configure for GPU
    ierr = VFSWind_ConfigureKSPForGPU(ksp);
    EXPECT_EQ(ierr, 0);

    // Clean up
    ierr = KSPDestroy(&ksp);
    EXPECT_EQ(ierr, 0);

    ierr = MatDestroy(&A);
    EXPECT_EQ(ierr, 0);

    ierr = DMDestroy(&dm);
    EXPECT_EQ(ierr, 0);
#else
    SUCCEED();
#endif
}

// Test: PETSc GPU support detection functions
TEST_F(GPUSolverTest, PetscGPUDetection) {
#ifdef ENABLE_GPU
    PetscBool has_petsc_gpu = vfswind::gpu::hasPetscGPUSupport();
    PetscBool has_hypre_gpu = vfswind::gpu::hasHypreGPUSupport();

    // These are informational - either result is valid
    std::cout << "PETSc GPU support: " << (has_petsc_gpu ? "Yes" : "No") << std::endl;
    std::cout << "HYPRE GPU support: " << (has_hypre_gpu ? "Yes" : "No") << std::endl;

    SUCCEED();
#else
    SUCCEED();
#endif
}

// Test: Simple Poisson solve (verifies GPU solver integration)
TEST_F(GPUSolverTest, SimplePoissonSolve) {
#ifdef ENABLE_GPU
    DM dm;
    Mat A;
    Vec x, b;
    KSP ksp;
    PetscErrorCode ierr;

    // Apply GPU options first
    ierr = VFSWind_ApplyGPUSolverOptions();
    ASSERT_EQ(ierr, 0);

    // Create 3D DMDA (16x16x16 grid)
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        16, 16, 16,
                        PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        1, 1,
                        NULL, NULL, NULL,
                        &dm);
    ASSERT_EQ(ierr, 0);

    // Configure DM for GPU
    ierr = VFSWind_ConfigureDMForGPU(dm);
    ASSERT_EQ(ierr, 0);

    ierr = DMSetUp(dm);
    ASSERT_EQ(ierr, 0);

    // Create matrix and vectors
    ierr = DMCreateMatrix(dm, &A);
    ASSERT_EQ(ierr, 0);

    ierr = DMCreateGlobalVector(dm, &x);
    ASSERT_EQ(ierr, 0);

    ierr = DMCreateGlobalVector(dm, &b);
    ASSERT_EQ(ierr, 0);

    // Fill matrix with simple Laplacian stencil
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(dm, &info);
    ASSERT_EQ(ierr, 0);

    MatStencil row, col[7];
    PetscScalar v[7];

    for (int k = info.zs; k < info.zs + info.zm; k++) {
        for (int j = info.ys; j < info.ys + info.ym; j++) {
            for (int i = info.xs; i < info.xs + info.xm; i++) {
                row.i = i; row.j = j; row.k = k;

                int n = 0;
                if (i > 0) {
                    col[n].i = i - 1; col[n].j = j; col[n].k = k;
                    v[n++] = -1.0;
                }
                if (i < info.mx - 1) {
                    col[n].i = i + 1; col[n].j = j; col[n].k = k;
                    v[n++] = -1.0;
                }
                if (j > 0) {
                    col[n].i = i; col[n].j = j - 1; col[n].k = k;
                    v[n++] = -1.0;
                }
                if (j < info.my - 1) {
                    col[n].i = i; col[n].j = j + 1; col[n].k = k;
                    v[n++] = -1.0;
                }
                if (k > 0) {
                    col[n].i = i; col[n].j = j; col[n].k = k - 1;
                    v[n++] = -1.0;
                }
                if (k < info.mz - 1) {
                    col[n].i = i; col[n].j = j; col[n].k = k + 1;
                    v[n++] = -1.0;
                }
                col[n].i = i; col[n].j = j; col[n].k = k;
                v[n++] = (PetscScalar)6.0;

                ierr = MatSetValuesStencil(A, 1, &row, n, col, v, INSERT_VALUES);
                ASSERT_EQ(ierr, 0);
            }
        }
    }

    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    ASSERT_EQ(ierr, 0);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    ASSERT_EQ(ierr, 0);

    // Set RHS to constant
    ierr = VecSet(b, 1.0);
    ASSERT_EQ(ierr, 0);

    // Create and configure KSP
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);
    ASSERT_EQ(ierr, 0);

    ierr = KSPSetOperators(ksp, A, A);
    ASSERT_EQ(ierr, 0);

    ierr = KSPSetType(ksp, KSPGMRES);
    ASSERT_EQ(ierr, 0);

    ierr = VFSWind_ConfigureKSPForGPU(ksp);
    ASSERT_EQ(ierr, 0);

    ierr = KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 100);
    ASSERT_EQ(ierr, 0);

    ierr = KSPSetFromOptions(ksp);
    ASSERT_EQ(ierr, 0);

    // Solve
    ierr = KSPSolve(ksp, b, x);
    EXPECT_EQ(ierr, 0);

    // Check convergence
    KSPConvergedReason reason;
    ierr = KSPGetConvergedReason(ksp, &reason);
    EXPECT_EQ(ierr, 0);

    PetscInt its;
    ierr = KSPGetIterationNumber(ksp, &its);
    EXPECT_EQ(ierr, 0);

    std::cout << "Poisson solve: " << its << " iterations, reason: " << reason << std::endl;

    // Should converge (positive reason)
    EXPECT_GT(reason, 0);

    // Clean up
    ierr = KSPDestroy(&ksp);
    EXPECT_EQ(ierr, 0);

    ierr = VecDestroy(&x);
    EXPECT_EQ(ierr, 0);

    ierr = VecDestroy(&b);
    EXPECT_EQ(ierr, 0);

    ierr = MatDestroy(&A);
    EXPECT_EQ(ierr, 0);

    ierr = DMDestroy(&dm);
    EXPECT_EQ(ierr, 0);
#else
    SUCCEED();
#endif
}
