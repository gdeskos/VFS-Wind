/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Dispatch - C Interface for GPU Kernels                     *
 *                                                                *
 * PURPOSE:                                                       *
 * This header provides C-callable functions that dispatch        *
 * computational work to GPU kernels. It serves as the bridge     *
 * between the existing C codebase and the C++ Kokkos GPU         *
 * kernels.                                                       *
 *                                                                *
 * USAGE:                                                         *
 * 1. Include this header in C files that need GPU acceleration   *
 * 2. Call VFSWind_GPU_* functions instead of CPU implementations *
 * 3. The dispatch functions handle data transfer automatically   *
 *                                                                *
 * ARCHITECTURE:                                                  *
 * ┌─────────────────┐     ┌──────────────────┐                  *
 * │  C Code         │ --> │  gpu_dispatch.h  │                  *
 * │  (les.c, etc.)  │     │  (C interface)   │                  *
 * └─────────────────┘     └────────┬─────────┘                  *
 *                                  │                             *
 *                         ┌────────▼─────────┐                  *
 *                         │ gpu_dispatch.cpp │                  *
 *                         │ (C++ wrapper)    │                  *
 *                         └────────┬─────────┘                  *
 *                                  │                             *
 *                         ┌────────▼─────────┐                  *
 *                         │ Kokkos Kernels   │                  *
 *                         │ (les_kernel.hpp) │                  *
 *                         └──────────────────┘                  *
 ******************************************************************/

#ifndef VFSWIND_GPU_DISPATCH_H
#define VFSWIND_GPU_DISPATCH_H

#include <petscdm.h>
#include <petscdmda.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=========================================================================
 * GPU AVAILABILITY CHECK
 *=========================================================================
 * These functions check if GPU acceleration is available and configured.
 * Always call VFSWind_GPU_IsAvailable() before using GPU functions.
 *========================================================================*/

/**
 * @brief Check if GPU acceleration is available
 *
 * Returns 1 if:
 * - Code was compiled with ENABLE_GPU
 * - Kokkos is initialized
 * - A suitable GPU backend is available
 *
 * @return 1 if GPU available, 0 otherwise
 */
int VFSWind_GPU_IsAvailable(void);

/**
 * @brief Get the name of the active GPU backend
 *
 * @return String like "CUDA", "HIP", "OpenMP", or "Serial"
 */
const char* VFSWind_GPU_GetBackendName(void);

/**
 * @brief Print GPU configuration info
 *
 * Prints details about the GPU backend, device info, and memory.
 */
void VFSWind_GPU_PrintInfo(void);

/*=========================================================================
 * GPU TIMING FUNCTIONS
 *=========================================================================
 * Enable timing by setting environment variable: VFSWIND_TIMING=1
 *========================================================================*/

/**
 * @brief Check if GPU timing is enabled
 *
 * @return 1 if VFSWIND_TIMING=1, 0 otherwise
 */
int VFSWind_GPU_IsTimingEnabled(void);

/**
 * @brief Print timing summary for all GPU operations
 *
 * Prints a table showing total time, average, min, max for each kernel.
 * Only prints if timing was enabled and data was collected.
 */
void VFSWind_GPU_PrintTimingSummary(void);

/**
 * @brief Clear all recorded timing data
 */
void VFSWind_GPU_ClearTimingData(void);

/*=========================================================================
 * LES TURBULENCE MODEL - GPU DISPATCH
 *=========================================================================
 * These functions compute LES turbulent eddy viscosity on GPU.
 *
 * ALGORITHM:
 * Static Smagorinsky:  ν_t = (Cs × Δ)² × |S|
 * Dynamic Smagorinsky: ν_t = (Cs_dynamic × Δ)² × |S|
 *
 * Where:
 *   Cs = Smagorinsky constant (static: ~0.1, dynamic: computed)
 *   Δ  = Filter width = (cell volume)^(1/3) = (1/aj)^(1/3)
 *   |S| = Strain rate magnitude = sqrt(2 × S_ij × S_ij)
 *========================================================================*/

/**
 * @brief Compute LES eddy viscosity on GPU (Static Smagorinsky)
 *
 * This function replaces Compute_eddy_viscosity_LES() on GPU.
 *
 * DATA FLOW:
 * Input:  lUcat (local Cartesian velocity)
 *         lCsi, lEta, lZet (metric tensors)
 *         lAj (Jacobian)
 *         lNvert (solid marker)
 *         lCs (Smagorinsky constant field)
 * Output: lNu_t (turbulent viscosity)
 *
 * @param da       DMDA for scalar fields
 * @param fda      DMDA for vector fields
 * @param lUcat    Local Cartesian velocity vector
 * @param lCsi     Local csi metric vector
 * @param lEta     Local eta metric vector
 * @param lZet     Local zet metric vector
 * @param lAj      Local Jacobian
 * @param lNvert   Local solid marker (0=fluid, >0=solid)
 * @param lCs      Local Smagorinsky constant
 * @param lNu_t    Output: Local turbulent viscosity
 * @return 0 on success, non-zero on error
 */
int VFSWind_GPU_ComputeEddyViscosityLES(
    DM da, DM fda,
    Vec lUcat,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    Vec lCs,
    Vec lNu_t
);

/**
 * @brief Compute LES eddy viscosity with constant Cs
 *
 * Simplified version using a single Cs value for all cells.
 *
 * @param da       DMDA for scalar fields
 * @param fda      DMDA for vector fields
 * @param lUcat    Local Cartesian velocity
 * @param lCsi, lEta, lZet  Metric vectors
 * @param lAj      Jacobian
 * @param lNvert   Solid marker
 * @param Cs       Constant Smagorinsky coefficient
 * @param lNu_t    Output: turbulent viscosity
 * @return 0 on success
 */
int VFSWind_GPU_ComputeEddyViscosityLES_ConstantCs(
    DM da, DM fda,
    Vec lUcat,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    double Cs,
    Vec lNu_t
);

/**
 * @brief Compute Dynamic Smagorinsky constant on GPU
 *
 * Uses the Germano identity with Lilly's least-squares:
 *   Cs² = <L_ij M_ij> / <M_ij M_ij>
 *
 * @param da, fda  DMDA objects
 * @param lUcat    Cartesian velocity
 * @param lCsi, lEta, lZet  Metrics
 * @param lAj      Jacobian
 * @param lNvert   Solid marker
 * @param lCs      Output: Dynamic Cs field
 * @return 0 on success
 */
int VFSWind_GPU_ComputeDynamicSmagorinsky(
    DM da, DM fda,
    Vec lUcat,
    Vec lCsi, Vec lEta, Vec lZet,
    Vec lAj,
    Vec lNvert,
    Vec lCs
);

/*=========================================================================
 * RANS k-omega TURBULENCE MODEL - GPU DISPATCH
 *=========================================================================
 * These functions compute RANS k-omega turbulence on GPU.
 *
 * SUPPORTED MODELS:
 * - RANS_WILCOX_LOW_RE (1)  : Wilcox k-omega with low-Re corrections
 * - RANS_WILCOX_HIGH_RE (2) : Wilcox k-omega standard
 * - RANS_SST_MENTER (3)     : Menter SST k-omega
 *
 * EQUATIONS:
 * k-equation:     ∂k/∂t = P_k - β*ωk + ∇·[(ν + σ_k ν_t)∇k]
 * omega-equation: ∂ω/∂t = αS² - βω² + ∇·[(ν + σ_ω ν_t)∇ω] + CD_kω
 *========================================================================*/

/**
 * @brief RANS model type enumeration (matches rans variable in code)
 */
typedef enum {
    RANS_WILCOX_LOW_RE = 1,
    RANS_WILCOX_HIGH_RE = 2,
    RANS_SST_MENTER = 3
} VFSWind_RANSModel;

/**
 * @brief Compute k-omega turbulent viscosity on GPU
 *
 * Standard k-omega: ν_t = k / ω
 * SST:              ν_t = a1 × k / max(a1 × ω, Ω × F2)
 *
 * @param da         DMDA for scalar fields
 * @param fda2       DMDA for k-omega (2-component) fields
 * @param lK_Omega   Local k-omega field (k=.x, omega=.y)
 * @param lNvert     Solid marker
 * @param lDistance  Distance to nearest wall
 * @param lNu_t      Output: turbulent viscosity
 * @param ren        Reynolds number
 * @param model      RANS model type
 * @return 0 on success
 */
int VFSWind_GPU_ComputeTurbulentViscosityRANS(
    DM da, DM fda2,
    Vec lK_Omega,
    Vec lNvert,
    Vec lDistance,
    Vec lNu_t,
    double ren,
    VFSWind_RANSModel model
);

/**
 * @brief Compute k-omega RHS terms on GPU
 *
 * Computes production, dissipation, and cross-diffusion terms.
 *
 * @param da, fda, fda2  DMDA objects
 * @param lUcat, lUcont  Velocity fields
 * @param lK_Omega       k-omega field
 * @param lNu_t          Turbulent viscosity
 * @param lCsi, lEta, lZet  Metrics
 * @param lAj            Jacobian
 * @param lNvert         Solid marker
 * @param lDistance      Wall distance
 * @param K_Omega_RHS    Output: k-omega RHS
 * @param ren            Reynolds number
 * @param model          RANS model type
 * @return 0 on success
 */
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
);

/*=========================================================================
 * LEVEL-SET METHOD - GPU DISPATCH
 *=========================================================================
 * These functions compute level-set advection and reinitialization on GPU.
 *
 * ADVECTION SCHEMES:
 * - WENO3: 3rd order Weighted Essentially Non-Oscillatory
 * - WENO5: 5th order WENO
 * - ENO2:  2nd order Essentially Non-Oscillatory
 *
 * ALGORITHM:
 * 1. Advection:        ∂φ/∂t + u·∇φ = 0
 * 2. Reinitialization: ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0
 *========================================================================*/

/**
 * @brief Level-set advection scheme enumeration
 */
typedef enum {
    LEVELSET_ENO2 = 0,
    LEVELSET_WENO3 = 1,
    LEVELSET_WENO5 = 2
} VFSWind_LevelsetScheme;

/**
 * @brief Advect level-set field on GPU
 *
 * Uses high-order WENO reconstruction for sharp interface tracking.
 *
 * DATA FLOW:
 * Input:  Levelset (global level-set field)
 *         lUcont (contravariant velocity for advection)
 *         lAj (Jacobian)
 *         lNvert (solid marker)
 * Output: Levelset (updated level-set)
 *
 * @param da         DMDA for scalar fields
 * @param fda        DMDA for vector fields
 * @param Levelset   Level-set field (input/output)
 * @param lUcont     Contravariant velocity
 * @param lAj        Jacobian
 * @param lNvert     Solid marker
 * @param dt         Time step
 * @param scheme     Advection scheme (WENO3, WENO5, ENO2)
 * @return 0 on success
 */
int VFSWind_GPU_AdvectLevelset(
    DM da, DM fda,
    Vec Levelset,
    Vec lUcont,
    Vec lAj,
    Vec lNvert,
    double dt,
    VFSWind_LevelsetScheme scheme
);

/**
 * @brief Reinitialize level-set to signed distance function on GPU
 *
 * Uses iterative approach to restore |∇φ| = 1 property.
 *
 * @param da         DMDA
 * @param Levelset   Level-set field (input/output)
 * @param lAj        Jacobian
 * @param lNvert     Solid marker
 * @param h          Grid spacing
 * @param max_iter   Maximum iterations
 * @param tol        Convergence tolerance
 * @return Number of iterations performed
 */
int VFSWind_GPU_ReinitLevelset(
    DM da,
    Vec Levelset,
    Vec lAj,
    Vec lNvert,
    double h,
    int max_iter,
    double tol
);

/**
 * @brief Compute density and viscosity from level-set on GPU
 *
 * Uses smoothed Heaviside function for two-phase flows:
 *   ρ = ρ₁ + (ρ₂ - ρ₁) × H(φ)
 *   μ = μ₁ + (μ₂ - μ₁) × H(φ)
 *
 * @param da         DMDA
 * @param lLevelset  Level-set field
 * @param lDensity   Output: density field
 * @param lMu        Output: viscosity field
 * @param rho1, rho2 Fluid densities
 * @param mu1, mu2   Fluid viscosities
 * @param epsilon    Interface thickness for smoothing
 * @return 0 on success
 */
int VFSWind_GPU_ComputeTwoPhaseProperties(
    DM da,
    Vec lLevelset,
    Vec lDensity,
    Vec lMu,
    double rho1, double rho2,
    double mu1, double mu2,
    double epsilon
);

/*=========================================================================
 * CONVECTION/VISCOUS TERMS - GPU DISPATCH
 *=========================================================================
 * These functions compute momentum equation RHS terms on GPU.
 *
 * CONVECTION:
 * Uses QUICK scheme: ∂(ρu_i u_j)/∂x_j
 *
 * VISCOUS:
 * ∂/∂x_j [(ν + ν_t)(∂u_i/∂x_j + ∂u_j/∂x_i)]
 *========================================================================*/

/**
 * @brief Compute convection term on GPU
 *
 * @param da, fda    DMDA objects
 * @param lUcont     Contravariant velocity
 * @param lUcat      Cartesian velocity
 * @param lNvert     Solid marker
 * @param Conv       Output: convection term
 * @return 0 on success
 */
int VFSWind_GPU_ComputeConvection(
    DM da, DM fda,
    Vec lUcont, Vec lUcat,
    Vec lNvert,
    Vec Conv
);

/**
 * @brief Compute viscous term on GPU
 *
 * @param da, fda    DMDA objects
 * @param lUcat      Cartesian velocity
 * @param lNvert     Solid marker
 * @param lICsi, lIEta, lIZet  I-face metrics
 * @param lJCsi, lJEta, lJZet  J-face metrics
 * @param lKCsi, lKEta, lKZet  K-face metrics
 * @param lIAj, lJAj, lKAj     Face Jacobians
 * @param lNu_t      Turbulent viscosity (can be NULL)
 * @param ren        Reynolds number
 * @param Visc       Output: viscous term
 * @return 0 on success
 */
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
);

#ifdef __cplusplus
}
#endif

#endif /* VFSWIND_GPU_DISPATCH_H */
