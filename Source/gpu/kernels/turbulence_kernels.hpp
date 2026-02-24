/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Turbulence GPU Kernels - Unified Header                        *
 * Phase 4: Level-Set and Turbulence Models                       *
 ******************************************************************/

#ifndef VFSWIND_TURBULENCE_KERNELS_HPP
#define VFSWIND_TURBULENCE_KERNELS_HPP

#ifdef ENABLE_GPU

// Level-set kernels
#include "levelset_kernel.hpp"

// Turbulence model kernels
#include "les_kernel.hpp"
#include "rans_kernel.hpp"

namespace vfswind {
namespace gpu {

/**
 * @brief Turbulence model type enumeration
 */
enum class TurbulenceModel {
    NONE = 0,
    LES_STATIC_SMAGORINSKY = 1,
    LES_DYNAMIC_SMAGORINSKY = 2,
    RANS_KOMEGA_WILCOX_LOW_RE = 3,
    RANS_KOMEGA_WILCOX_HIGH_RE = 4,
    RANS_KOMEGA_SST = 5
};

/**
 * @brief Unified turbulent viscosity computation
 *
 * Dispatches to appropriate kernel based on model type.
 */
class TurbulenceKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    /**
     * @brief Compute turbulent viscosity based on model selection
     *
     * @param ucat Cartesian velocity
     * @param nvert Solid marker
     * @param csi, eta, zet Metrics
     * @param aj Jacobian
     * @param nu_t Output: turbulent viscosity
     * @param domain Domain info
     * @param model Turbulence model type
     * @param model_data Additional model data (k, omega for RANS)
     * @param Re_nu Reynolds number
     */
    static void computeTurbulentViscosity(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& csi,
        const VectorView& eta,
        const VectorView& zet,
        const ScalarView& aj,
        ScalarView& nu_t,
        const KernelDomainInfo& domain,
        TurbulenceModel model,
        double Re_nu = 1000.0,
        double Cs_constant = 0.1
    ) {
        switch (model) {
            case TurbulenceModel::NONE:
                Kokkos::deep_copy(nu_t, 0.0);
                break;

            case TurbulenceModel::LES_STATIC_SMAGORINSKY:
                les::SmagorinskyKernel::computeEddyViscosityConstantCs(
                    ucat, nvert, csi, eta, zet, aj, Cs_constant, nu_t, domain
                );
                break;

            case TurbulenceModel::LES_DYNAMIC_SMAGORINSKY: {
                ScalarView Cs("Cs", domain.mz, domain.my, domain.mx);
                les::DynamicSmagorinskyKernel::computeDynamicCs(
                    ucat, nvert, csi, eta, zet, aj, Cs, domain
                );
                les::SmagorinskyKernel::computeEddyViscosity(
                    ucat, nvert, csi, eta, zet, aj, Cs, nu_t, domain
                );
                break;
            }

            default:
                // RANS models require k-omega data, not handled here
                Kokkos::deep_copy(nu_t, 0.0);
                break;
        }
    }

    /**
     * @brief Compute RANS turbulent viscosity (k-omega based)
     */
    static void computeRANSViscosity(
        const ScalarView& k,
        const ScalarView& omega,
        const ScalarView& nvert,
        const ScalarView& F1,
        ScalarView& nu_t,
        double Re_nu,
        rans::RANSModel model,
        const KernelDomainInfo& domain
    ) {
        rans::KOmegaKernel::computeTurbulentViscosity(
            k, omega, nvert, F1, nu_t, Re_nu, model, domain
        );
    }
};

/**
 * @brief Unified level-set operations
 */
class LevelsetKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    /**
     * @brief Advect level-set field
     */
    static void advect(
        ScalarView& levelset,
        const VectorView& ucont,
        const ScalarView& nvert,
        const ScalarView& aj,
        double dt,
        const KernelDomainInfo& domain,
        levelset::LevelsetAdvectionKernel::Scheme scheme =
            levelset::LevelsetAdvectionKernel::Scheme::WENO3
    ) {
        ScalarView rhs("levelset_rhs", domain.mz, domain.my, domain.mx);

        levelset::LevelsetAdvectionKernel::computeAdvectionRHS(
            levelset, ucont, nvert, aj, rhs, domain, scheme
        );

        levelset::LevelsetAdvectionKernel::updateExplicit(
            levelset, rhs, nvert, dt, domain
        );
    }

    /**
     * @brief Reinitialize level-set to signed distance function
     */
    static int reinitialize(
        ScalarView& levelset,
        const ScalarView& nvert,
        const ScalarView& aj,
        double h,
        const KernelDomainInfo& domain,
        int max_iter = 100,
        double tol = 1.0e-6
    ) {
        ScalarView levelset0("levelset0", domain.mz, domain.my, domain.mx);
        Kokkos::deep_copy(levelset0, levelset);

        return levelset::LevelsetReinitKernel::reinitialize(
            levelset, levelset0, nvert, aj, h, domain, max_iter, tol
        );
    }

    /**
     * @brief Compute fluid properties from level-set
     */
    static void computeProperties(
        const ScalarView& levelset,
        ScalarView& rho,
        ScalarView& mu,
        double rho1, double rho2,
        double mu1, double mu2,
        double epsilon,
        const KernelDomainInfo& domain
    ) {
        levelset::LevelsetPropertiesKernel::computeDensity(
            levelset, rho, rho1, rho2, epsilon, domain
        );

        levelset::LevelsetPropertiesKernel::computeViscosity(
            levelset, mu, mu1, mu2, epsilon, domain
        );
    }
};

/**
 * @brief Print Phase 4 GPU kernel info
 */
inline void printPhase4Info() {
    printf("==============================================================\n");
    printf("VFS-Wind Phase 4: Level-Set & Turbulence GPU Kernels\n");
    printf("==============================================================\n");
    printf("  Level-Set Kernels:\n");
    printf("    - WENO3 advection\n");
    printf("    - WENO5 advection\n");
    printf("    - ENO2 advection\n");
    printf("    - Signed distance reinitialization\n");
    printf("    - Two-phase property computation\n");
    printf("  LES Kernels:\n");
    printf("    - Static Smagorinsky model\n");
    printf("    - Dynamic Smagorinsky model\n");
    printf("    - van Driest wall damping\n");
    printf("  RANS Kernels:\n");
    printf("    - k-omega Wilcox (Low-Re)\n");
    printf("    - k-omega Wilcox (High-Re)\n");
    printf("    - k-omega SST (Menter)\n");
    printf("==============================================================\n");
}

} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_TURBULENCE_KERNELS_HPP
