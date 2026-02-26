/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Dispatch - Implementation                                  *
 *                                                                *
 * This file implements the C-callable GPU dispatch functions.    *
 * It handles:                                                    *
 * 1. Data transfer between PETSc vectors and Kokkos views        *
 * 2. Domain information extraction from DMDA                     *
 * 3. Dispatching to appropriate GPU kernels                      *
 * 4. Error handling and fallback to CPU                          *
 ******************************************************************/

#include "gpu_dispatch.h"

#ifdef ENABLE_GPU

#include <Kokkos_Core.hpp>
#include "variables.h"  /* For Cmpnts type */
#include "gpu_config.hpp"
#include "gpu_timer.hpp"  /* For performance timing */
#include "kernels/kernels.hpp"
#include "kernels/turbulence_kernels.hpp"

using namespace vfswind::gpu;

/*=========================================================================
 * HELPER FUNCTIONS
 *=========================================================================
 * These internal functions handle common operations like extracting
 * domain info from DMDA and copying data between PETSc and Kokkos.
 *========================================================================*/

namespace {

/**
 * @brief Extract domain info from DMDA
 *
 * Converts PETSc DMDA local info to KernelDomainInfo structure
 * used by GPU kernels.
 *
 * @param da PETSc DMDA object
 * @param stencil_width Offset from boundaries (1 for standard, 2 for WENO3, 3 for WENO5)
 */
KernelDomainInfo extractDomainInfo(DM da, int stencil_width = 1) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    KernelDomainInfo domain;
    domain.mx = info.mx;
    domain.my = info.my;
    domain.mz = info.mz;

    // Local indices (excluding ghost cells based on stencil width)
    domain.lxs = (info.xs == 0) ? stencil_width : info.xs;
    domain.lxe = (info.xs + info.xm == info.mx) ? info.mx - stencil_width : info.xs + info.xm;
    domain.lys = (info.ys == 0) ? stencil_width : info.ys;
    domain.lye = (info.ys + info.ym == info.my) ? info.my - stencil_width : info.ys + info.ym;
    domain.lzs = (info.zs == 0) ? stencil_width : info.zs;
    domain.lze = (info.zs + info.zm == info.mz) ? info.mz - stencil_width : info.zs + info.zm;

    // Reynolds number (set later if needed)
    domain.ren = 1.0;

    return domain;
}

/**
 * @brief Copy PETSc scalar Vec to Kokkos View
 *
 * Handles the data transfer from PETSc's distributed array format
 * to Kokkos' contiguous view format.
 */
void copyPetscToKokkos(DM da, Vec vec, ScalarView3D<DeviceMemorySpace>& view) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscReal ***arr;
    DMDAVecGetArray(da, vec, &arr);

    auto view_h = Kokkos::create_mirror_view(view);

    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                view_h(k, j, i) = arr[k][j][i];
            }
        }
    }

    Kokkos::deep_copy(view, view_h);
    DMDAVecRestoreArray(da, vec, &arr);
}

/**
 * @brief Copy Kokkos View to PETSc scalar Vec
 */
void copyKokkosToPetsc(DM da, const ScalarView3D<DeviceMemorySpace>& view, Vec vec) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    PetscReal ***arr;
    DMDAVecGetArray(da, vec, &arr);

    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                arr[k][j][i] = view_h(k, j, i);
            }
        }
    }

    DMDAVecRestoreArray(da, vec, &arr);
}

/**
 * @brief Copy PETSc vector (Cmpnts) Vec to Kokkos View
 */
void copyPetscVectorToKokkos(DM fda, Vec vec, VectorView3D<DeviceMemorySpace>& view) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(fda, &info);

    Cmpnts ***arr;
    DMDAVecGetArray(fda, vec, &arr);

    auto view_h = Kokkos::create_mirror_view(view);

    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                view_h(k, j, i).x = arr[k][j][i].x;
                view_h(k, j, i).y = arr[k][j][i].y;
                view_h(k, j, i).z = arr[k][j][i].z;
            }
        }
    }

    Kokkos::deep_copy(view, view_h);
    DMDAVecRestoreArray(fda, vec, &arr);
}

/**
 * @brief Copy Kokkos View to PETSc vector Vec
 */
void copyKokkosVectorToPetsc(DM fda, const VectorView3D<DeviceMemorySpace>& view, Vec vec) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(fda, &info);

    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    Cmpnts ***arr;
    DMDAVecGetArray(fda, vec, &arr);

    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                arr[k][j][i].x = view_h(k, j, i).x;
                arr[k][j][i].y = view_h(k, j, i).y;
                arr[k][j][i].z = view_h(k, j, i).z;
            }
        }
    }

    DMDAVecRestoreArray(fda, vec, &arr);
}

} // anonymous namespace

/*=========================================================================
 * GPU AVAILABILITY FUNCTIONS
 *========================================================================*/

extern "C" {

int VFSWind_GPU_IsAvailable(void) {
    return Kokkos::is_initialized() ? 1 : 0;
}

/* Note: VFSWind_GPU_GetBackendName is implemented in kokkos_init.cpp */

void VFSWind_GPU_PrintInfo(void) {
    printPhase4Info();
}

/*=========================================================================
 * TIMING FUNCTIONS
 *========================================================================*/

int VFSWind_GPU_IsTimingEnabled(void) {
    return isTimingEnabled() ? 1 : 0;
}

void VFSWind_GPU_PrintTimingSummary(void) {
    TimerRegistry::instance().printSummary();
}

void VFSWind_GPU_ClearTimingData(void) {
    TimerRegistry::instance().clear();
}

/*=========================================================================
 * LES TURBULENCE MODEL DISPATCH
 *========================================================================*/

int VFSWind_GPU_ComputeEddyViscosityLES(
    DM da, DM fda,
    Vec lUcat,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    Vec lCs,
    Vec lNu_t
) {
    if (!Kokkos::is_initialized()) return -1;

    GPUTimer timer_total("LES_EddyViscosity_Total");
    GPUTimer timer_h2d("LES_EddyViscosity_HostToDevice");
    GPUTimer timer_kernel("LES_EddyViscosity_Kernel");
    GPUTimer timer_d2h("LES_EddyViscosity_DeviceToHost");

    timer_total.start();

    // Get domain info
    KernelDomainInfo domain = extractDomainInfo(da);

    // Allocate Kokkos views
    VectorView3D<DeviceMemorySpace> ucat("ucat", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> csi("csi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> eta("eta", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> zet("zet", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> aj("aj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> cs("cs", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nu_t("nu_t", domain.mz, domain.my, domain.mx);

    // Copy data from PETSc to Kokkos
    timer_h2d.start();
    copyPetscVectorToKokkos(fda, lUcat, ucat);
    copyPetscVectorToKokkos(fda, lCsi, csi);
    copyPetscVectorToKokkos(fda, lEta, eta);
    copyPetscVectorToKokkos(fda, lZet, zet);
    copyPetscToKokkos(da, lAj, aj);
    copyPetscToKokkos(da, lNvert, nvert);
    copyPetscToKokkos(da, lCs, cs);
    timer_h2d.stop();

    // Execute GPU kernel
    timer_kernel.start();
    les::SmagorinskyKernel::computeEddyViscosity(
        ucat, nvert, csi, eta, zet, aj, cs, nu_t, domain
    );
    timer_kernel.stop();

    // Copy result back to PETSc
    timer_d2h.start();
    copyKokkosToPetsc(da, nu_t, lNu_t);
    timer_d2h.stop();

    timer_total.stop();

    // Record timings (only if VFSWIND_TIMING=1)
    if (isTimingEnabled()) {
        TimerRegistry::instance().record("LES_EddyViscosity_Total", timer_total.elapsed_us());
        TimerRegistry::instance().record("LES_EddyViscosity_H2D", timer_h2d.elapsed_us());
        TimerRegistry::instance().record("LES_EddyViscosity_Kernel", timer_kernel.elapsed_us());
        TimerRegistry::instance().record("LES_EddyViscosity_D2H", timer_d2h.elapsed_us());
    }

    return 0;
}

int VFSWind_GPU_ComputeEddyViscosityLES_ConstantCs(
    DM da, DM fda,
    Vec lUcat,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    double Cs,
    Vec lNu_t
) {
    if (!Kokkos::is_initialized()) return -1;

    KernelDomainInfo domain = extractDomainInfo(da);

    VectorView3D<DeviceMemorySpace> ucat("ucat", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> csi("csi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> eta("eta", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> zet("zet", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> aj("aj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nu_t("nu_t", domain.mz, domain.my, domain.mx);

    copyPetscVectorToKokkos(fda, lUcat, ucat);
    copyPetscVectorToKokkos(fda, lCsi, csi);
    copyPetscVectorToKokkos(fda, lEta, eta);
    copyPetscVectorToKokkos(fda, lZet, zet);
    copyPetscToKokkos(da, lAj, aj);
    copyPetscToKokkos(da, lNvert, nvert);

    les::SmagorinskyKernel::computeEddyViscosityConstantCs(
        ucat, nvert, csi, eta, zet, aj, Cs, nu_t, domain
    );

    copyKokkosToPetsc(da, nu_t, lNu_t);

    return 0;
}

int VFSWind_GPU_ComputeDynamicSmagorinsky(
    DM da, DM fda,
    Vec lUcat,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    Vec lCs
) {
    if (!Kokkos::is_initialized()) return -1;

    KernelDomainInfo domain = extractDomainInfo(da);

    VectorView3D<DeviceMemorySpace> ucat("ucat", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> csi("csi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> eta("eta", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> zet("zet", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> aj("aj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> cs("cs", domain.mz, domain.my, domain.mx);

    copyPetscVectorToKokkos(fda, lUcat, ucat);
    copyPetscVectorToKokkos(fda, lCsi, csi);
    copyPetscVectorToKokkos(fda, lEta, eta);
    copyPetscVectorToKokkos(fda, lZet, zet);
    copyPetscToKokkos(da, lAj, aj);
    copyPetscToKokkos(da, lNvert, nvert);

    les::DynamicSmagorinskyKernel::computeDynamicCs(
        ucat, nvert, csi, eta, zet, aj, cs, domain
    );

    copyKokkosToPetsc(da, cs, lCs);

    return 0;
}

/*=========================================================================
 * LEVEL-SET DISPATCH
 *========================================================================*/

int VFSWind_GPU_AdvectLevelset(
    DM da, DM fda,
    Vec Levelset,
    Vec lUcont,
    Vec lAj,
    Vec lNvert,
    double dt,
    VFSWind_LevelsetScheme scheme
) {
    if (!Kokkos::is_initialized()) return -1;

    // Convert scheme enum and determine stencil width
    levelset::LevelsetAdvectionKernel::Scheme kokkos_scheme;
    int stencil_width = 2;  // Default for WENO3
    switch (scheme) {
        case LEVELSET_WENO5:
            kokkos_scheme = levelset::LevelsetAdvectionKernel::Scheme::WENO5;
            stencil_width = 3;  // WENO5 needs 3-cell offset
            break;
        case LEVELSET_ENO2:
            kokkos_scheme = levelset::LevelsetAdvectionKernel::Scheme::ENO2;
            stencil_width = 2;  // ENO2 needs 2-cell offset
            break;
        case LEVELSET_WENO3:
        default:
            kokkos_scheme = levelset::LevelsetAdvectionKernel::Scheme::WENO3;
            stencil_width = 2;  // WENO3 needs 2-cell offset
            break;
    }

    KernelDomainInfo domain = extractDomainInfo(da, stencil_width);

    ScalarView3D<DeviceMemorySpace> phi("phi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> ucont("ucont", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> aj("aj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);

    copyPetscToKokkos(da, Levelset, phi);
    copyPetscVectorToKokkos(fda, lUcont, ucont);
    copyPetscToKokkos(da, lAj, aj);
    copyPetscToKokkos(da, lNvert, nvert);

    // Advect level-set
    LevelsetKernel::advect(phi, ucont, nvert, aj, dt, domain, kokkos_scheme);

    copyKokkosToPetsc(da, phi, Levelset);

    return 0;
}

int VFSWind_GPU_ReinitLevelset(
    DM da,
    Vec Levelset,
    Vec lAj,
    Vec lNvert,
    double h,
    int max_iter,
    double tol
) {
    if (!Kokkos::is_initialized()) return -1;

    // Reinitialization uses first-order derivatives, but needs offset for safety
    KernelDomainInfo domain = extractDomainInfo(da, 2);

    ScalarView3D<DeviceMemorySpace> phi("phi", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> aj("aj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);

    copyPetscToKokkos(da, Levelset, phi);
    copyPetscToKokkos(da, lAj, aj);
    copyPetscToKokkos(da, lNvert, nvert);

    int iters = LevelsetKernel::reinitialize(phi, nvert, aj, h, domain, max_iter, tol);

    copyKokkosToPetsc(da, phi, Levelset);

    return iters;
}

int VFSWind_GPU_ComputeTwoPhaseProperties(
    DM da,
    Vec lLevelset,
    Vec lDensity,
    Vec lMu,
    double rho1, double rho2,
    double mu1, double mu2,
    double epsilon
) {
    if (!Kokkos::is_initialized()) return -1;

    KernelDomainInfo domain = extractDomainInfo(da);

    ScalarView3D<DeviceMemorySpace> phi("phi", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> rho("rho", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> mu("mu", domain.mz, domain.my, domain.mx);

    copyPetscToKokkos(da, lLevelset, phi);

    LevelsetKernel::computeProperties(phi, rho, mu, rho1, rho2, mu1, mu2, epsilon, domain);

    copyKokkosToPetsc(da, rho, lDensity);
    copyKokkosToPetsc(da, mu, lMu);

    return 0;
}

/*=========================================================================
 * CONVECTION/VISCOUS DISPATCH
 *========================================================================*/

int VFSWind_GPU_ComputeConvection(
    DM da, DM fda,
    Vec lUcont, Vec lUcat,
    Vec lNvert,
    Vec Conv
) {
    if (!Kokkos::is_initialized()) return -1;

    KernelDomainInfo domain = extractDomainInfo(da);

    VectorView3D<DeviceMemorySpace> ucont("ucont", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> ucat("ucat", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> conv("conv", domain.mz, domain.my, domain.mx);

    copyPetscVectorToKokkos(fda, lUcont, ucont);
    copyPetscVectorToKokkos(fda, lUcat, ucat);
    copyPetscToKokkos(da, lNvert, nvert);

    kernels::ConvectionKernel::execute(ucont, ucat, nvert, conv, domain);

    copyKokkosVectorToPetsc(fda, conv, Conv);

    return 0;
}

int VFSWind_GPU_ComputeViscous(
    DM da, DM fda,
    Vec lUcat,
    Vec lNvert,
    Vec lICsi, Vec lIEta, Vec lIZet,
    Vec lJCsi, Vec lJEta, Vec lJZet,
    Vec lKCsi, Vec lKEta, Vec lKZet,
    Vec lIAj, Vec lJAj, Vec lKAj,
    Vec lNu_t,
    double ren,
    Vec Visc
) {
    if (!Kokkos::is_initialized()) return -1;

    KernelDomainInfo domain = extractDomainInfo(da);
    domain.ren = ren;

    VectorView3D<DeviceMemorySpace> ucat("ucat", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", domain.mz, domain.my, domain.mx);

    VectorView3D<DeviceMemorySpace> icsi("icsi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> ieta("ieta", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> izet("izet", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> jcsi("jcsi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> jeta("jeta", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> jzet("jzet", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> kcsi("kcsi", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> keta("keta", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> kzet("kzet", domain.mz, domain.my, domain.mx);

    ScalarView3D<DeviceMemorySpace> iaj("iaj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> jaj("jaj", domain.mz, domain.my, domain.mx);
    ScalarView3D<DeviceMemorySpace> kaj("kaj", domain.mz, domain.my, domain.mx);

    ScalarView3D<DeviceMemorySpace> nu_t("nu_t", domain.mz, domain.my, domain.mx);
    VectorView3D<DeviceMemorySpace> visc("visc", domain.mz, domain.my, domain.mx);

    // Copy inputs
    copyPetscVectorToKokkos(fda, lUcat, ucat);
    copyPetscToKokkos(da, lNvert, nvert);

    copyPetscVectorToKokkos(fda, lICsi, icsi);
    copyPetscVectorToKokkos(fda, lIEta, ieta);
    copyPetscVectorToKokkos(fda, lIZet, izet);
    copyPetscVectorToKokkos(fda, lJCsi, jcsi);
    copyPetscVectorToKokkos(fda, lJEta, jeta);
    copyPetscVectorToKokkos(fda, lJZet, jzet);
    copyPetscVectorToKokkos(fda, lKCsi, kcsi);
    copyPetscVectorToKokkos(fda, lKEta, keta);
    copyPetscVectorToKokkos(fda, lKZet, kzet);

    copyPetscToKokkos(da, lIAj, iaj);
    copyPetscToKokkos(da, lJAj, jaj);
    copyPetscToKokkos(da, lKAj, kaj);

    if (lNu_t) {
        copyPetscToKokkos(da, lNu_t, nu_t);
    } else {
        Kokkos::deep_copy(nu_t, 0.0);
    }

    // Execute kernel
    ScalarView3D<DeviceMemorySpace>* nu_t_ptr = lNu_t ? &nu_t : nullptr;
    kernels::ViscousKernel::execute(
        ucat, nvert,
        icsi, ieta, izet,
        jcsi, jeta, jzet,
        kcsi, keta, kzet,
        iaj, jaj, kaj,
        nu_t_ptr, visc, domain
    );

    copyKokkosVectorToPetsc(fda, visc, Visc);

    return 0;
}

/*=========================================================================
 * RANS k-omega DISPATCH (placeholder - full implementation needed)
 *========================================================================*/

int VFSWind_GPU_ComputeTurbulentViscosityRANS(
    DM da, DM fda2,
    Vec lK_Omega,
    Vec lNvert,
    Vec lDistance,
    Vec lNu_t,
    double ren,
    VFSWind_RANSModel model
) {
    // TODO: Implement full RANS dispatch
    // For now, return error to indicate not implemented
    return -1;
}

int VFSWind_GPU_ComputeKOmegaRHS(
    DM da, DM fda, DM fda2,
    Vec lUcat, Vec lUcont,
    Vec lK_Omega,
    Vec lNu_t,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    Vec lDistance,
    Vec K_Omega_RHS,
    double ren,
    VFSWind_RANSModel model
) {
    // TODO: Implement full RANS dispatch
    return -1;
}

} // extern "C"

#else // !ENABLE_GPU

/*=========================================================================
 * STUB IMPLEMENTATIONS (when GPU is disabled)
 *=========================================================================
 * These stubs return appropriate error codes when GPU is not available.
 *========================================================================*/

extern "C" {

int VFSWind_GPU_IsAvailable(void) { return 0; }

/* Note: VFSWind_GPU_GetBackendName is implemented in kokkos_init.cpp */

void VFSWind_GPU_PrintInfo(void) {
    /* No-op when GPU is disabled */
}

int VFSWind_GPU_IsTimingEnabled(void) { return 0; }
void VFSWind_GPU_PrintTimingSummary(void) { /* No-op */ }
void VFSWind_GPU_ClearTimingData(void) { /* No-op */ }

int VFSWind_GPU_ComputeEddyViscosityLES(
    DM da, DM fda, Vec lUcat, Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj, Vec lNvert, Vec lCs, Vec lNu_t
) { return -1; }

int VFSWind_GPU_ComputeEddyViscosityLES_ConstantCs(
    DM da, DM fda, Vec lUcat, Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj, Vec lNvert, double Cs, Vec lNu_t
) { return -1; }

int VFSWind_GPU_ComputeDynamicSmagorinsky(
    DM da, DM fda, Vec lUcat, Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj, Vec lNvert, Vec lCs
) { return -1; }

int VFSWind_GPU_ComputeTurbulentViscosityRANS(
    DM da, DM fda2, Vec lK_Omega, Vec lNvert, Vec lDistance,
    Vec lNu_t, double ren, VFSWind_RANSModel model
) { return -1; }

int VFSWind_GPU_ComputeKOmegaRHS(
    DM da, DM fda, DM fda2, Vec lUcat, Vec lUcont, Vec lK_Omega,
    Vec lNu_t, Vec lCsi, Vec lEta, Vec lZet, Vec lAj, Vec lNvert,
    Vec lDistance, Vec K_Omega_RHS, double ren, VFSWind_RANSModel model
) { return -1; }

int VFSWind_GPU_AdvectLevelset(
    DM da, DM fda, Vec Levelset, Vec lUcont, Vec lAj, Vec lNvert,
    double dt, VFSWind_LevelsetScheme scheme
) { return -1; }

int VFSWind_GPU_ReinitLevelset(
    DM da, Vec Levelset, Vec lAj, Vec lNvert,
    double h, int max_iter, double tol
) { return -1; }

int VFSWind_GPU_ComputeTwoPhaseProperties(
    DM da, Vec lLevelset, Vec lDensity, Vec lMu,
    double rho1, double rho2, double mu1, double mu2, double epsilon
) { return -1; }

int VFSWind_GPU_ComputeConvection(
    DM da, DM fda, Vec lUcont, Vec lUcat, Vec lNvert, Vec Conv
) { return -1; }

int VFSWind_GPU_ComputeViscous(
    DM da, DM fda, Vec lUcat, Vec lNvert,
    Vec lICsi, Vec lIEta, Vec lIZet,
    Vec lJCsi, Vec lJEta, Vec lJZet,
    Vec lKCsi, Vec lKEta, Vec lKZet,
    Vec lIAj, Vec lJAj, Vec lKAj,
    Vec lNu_t, double ren, Vec Visc
) { return -1; }

} // extern "C"

#endif // ENABLE_GPU
