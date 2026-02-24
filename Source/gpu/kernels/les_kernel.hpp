/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * LES GPU Kernels                                                *
 * Phase 4: Large Eddy Simulation Turbulence Models               *
 ******************************************************************/

#ifndef VFSWIND_LES_KERNEL_HPP
#define VFSWIND_LES_KERNEL_HPP

#ifdef ENABLE_GPU

#include "../gpu_config.hpp"
#include "petsc_kokkos.hpp"

namespace vfswind {
namespace gpu {
namespace les {

/**
 * @brief Strain-rate tensor computation utilities
 */
struct StrainRateTensor {

    /**
     * @brief Compute strain rate tensor magnitude
     *
     * |S| = sqrt(2 * Sij * Sij)
     *
     * where Sij = 0.5 * (∂ui/∂xj + ∂uj/∂xi)
     */
    KOKKOS_INLINE_FUNCTION
    static double computeMagnitude(
        double du_dx, double du_dy, double du_dz,
        double dv_dx, double dv_dy, double dv_dz,
        double dw_dx, double dw_dy, double dw_dz
    ) {
        // Symmetric strain rate tensor components
        double Sxx = du_dx;
        double Syy = dv_dy;
        double Szz = dw_dz;
        double Sxy = 0.5 * (du_dy + dv_dx);
        double Sxz = 0.5 * (du_dz + dw_dx);
        double Syz = 0.5 * (dv_dz + dw_dy);

        // |S| = sqrt(2 * Sij * Sij)
        double S_sq = Sxx * Sxx + Syy * Syy + Szz * Szz
                    + 2.0 * (Sxy * Sxy + Sxz * Sxz + Syz * Syz);

        return Kokkos::sqrt(2.0 * S_sq);
    }

    /**
     * @brief Transform computational gradients to physical gradients
     *
     * Given gradients in computational space (dudc, dude, dudz),
     * compute physical gradients (du/dx, du/dy, du/dz).
     */
    KOKKOS_INLINE_FUNCTION
    static void computePhysicalGradients(
        double csi0, double csi1, double csi2,
        double eta0, double eta1, double eta2,
        double zet0, double zet1, double zet2,
        double ajc,
        double dudc, double dvdc, double dwdc,
        double dude, double dvde, double dwde,
        double dudz, double dvdz, double dwdz,
        double& du_dx, double& dv_dx, double& dw_dx,
        double& du_dy, double& dv_dy, double& dw_dy,
        double& du_dz, double& dv_dz, double& dw_dz
    ) {
        // Transform using metric tensor
        // ∂u/∂x = ajc * (csi0 * ∂u/∂ξ + eta0 * ∂u/∂η + zet0 * ∂u/∂ζ)
        du_dx = ajc * (csi0 * dudc + eta0 * dude + zet0 * dudz);
        dv_dx = ajc * (csi0 * dvdc + eta0 * dvde + zet0 * dvdz);
        dw_dx = ajc * (csi0 * dwdc + eta0 * dwde + zet0 * dwdz);

        du_dy = ajc * (csi1 * dudc + eta1 * dude + zet1 * dudz);
        dv_dy = ajc * (csi1 * dvdc + eta1 * dvde + zet1 * dvdz);
        dw_dy = ajc * (csi1 * dwdc + eta1 * dwde + zet1 * dwdz);

        du_dz = ajc * (csi2 * dudc + eta2 * dude + zet2 * dudz);
        dv_dz = ajc * (csi2 * dvdc + eta2 * dvde + zet2 * dvdz);
        dw_dz = ajc * (csi2 * dwdc + eta2 * dwde + zet2 * dwdz);
    }
};

/**
 * @brief Smagorinsky LES model kernel
 *
 * Computes turbulent eddy viscosity using the Smagorinsky model:
 *   ν_t = (Cs * Δ)² * |S|
 *
 * where:
 *   Cs = Smagorinsky constant
 *   Δ = filter width = (1/aj)^(1/3)
 *   |S| = strain rate magnitude
 */
class SmagorinskyKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    // Default Smagorinsky constant
    static constexpr double DEFAULT_CS = 0.1;

    /**
     * @brief Compute eddy viscosity using static Smagorinsky model
     *
     * @param ucat Cartesian velocity field
     * @param nvert Solid marker
     * @param csi, eta, zet Metric vectors at cell centers
     * @param aj Jacobian
     * @param Cs Smagorinsky constant field (or constant value)
     * @param nu_t Output: turbulent viscosity
     * @param domain Domain info
     */
    static void computeEddyViscosity(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& csi,
        const VectorView& eta,
        const VectorView& zet,
        const ScalarView& aj,
        const ScalarView& Cs,
        ScalarView& nu_t,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const int mx = domain.mx, my = domain.my, mz = domain.mz;
        const double solid = 0.1;

        // Initialize to zero
        Kokkos::deep_copy(nu_t, 0.0);

        Kokkos::parallel_for("SmagorinskyEddyViscosity",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Skip solid cells
                if (nvert(k, j, i) > 1.1) {
                    nu_t(k, j, i) = 0.0;
                    return;
                }

                // Get metric tensors
                double csi0 = csi(k, j, i).x;
                double csi1 = csi(k, j, i).y;
                double csi2 = csi(k, j, i).z;
                double eta0 = eta(k, j, i).x;
                double eta1 = eta(k, j, i).y;
                double eta2 = eta(k, j, i).z;
                double zet0 = zet(k, j, i).x;
                double zet1 = zet(k, j, i).y;
                double zet2 = zet(k, j, i).z;
                double ajc = aj(k, j, i);

                // Compute velocity gradients in computational space
                double dudc, dvdc, dwdc;
                double dude, dvde, dwde;
                double dudz, dvdz, dwdz;

                // Central differences in i-direction
                dudc = 0.5 * (ucat(k, j, i+1).x - ucat(k, j, i-1).x);
                dvdc = 0.5 * (ucat(k, j, i+1).y - ucat(k, j, i-1).y);
                dwdc = 0.5 * (ucat(k, j, i+1).z - ucat(k, j, i-1).z);

                // Central differences in j-direction (with solid cell handling)
                if (nvert(k, j+1, i) > solid) {
                    dude = ucat(k, j, i).x - ucat(k, j-1, i).x;
                    dvde = ucat(k, j, i).y - ucat(k, j-1, i).y;
                    dwde = ucat(k, j, i).z - ucat(k, j-1, i).z;
                } else if (nvert(k, j-1, i) > solid) {
                    dude = ucat(k, j+1, i).x - ucat(k, j, i).x;
                    dvde = ucat(k, j+1, i).y - ucat(k, j, i).y;
                    dwde = ucat(k, j+1, i).z - ucat(k, j, i).z;
                } else {
                    dude = 0.5 * (ucat(k, j+1, i).x - ucat(k, j-1, i).x);
                    dvde = 0.5 * (ucat(k, j+1, i).y - ucat(k, j-1, i).y);
                    dwde = 0.5 * (ucat(k, j+1, i).z - ucat(k, j-1, i).z);
                }

                // Central differences in k-direction (with solid cell handling)
                if (nvert(k+1, j, i) > solid) {
                    dudz = ucat(k, j, i).x - ucat(k-1, j, i).x;
                    dvdz = ucat(k, j, i).y - ucat(k-1, j, i).y;
                    dwdz = ucat(k, j, i).z - ucat(k-1, j, i).z;
                } else if (nvert(k-1, j, i) > solid) {
                    dudz = ucat(k+1, j, i).x - ucat(k, j, i).x;
                    dvdz = ucat(k+1, j, i).y - ucat(k, j, i).y;
                    dwdz = ucat(k+1, j, i).z - ucat(k, j, i).z;
                } else {
                    dudz = 0.5 * (ucat(k+1, j, i).x - ucat(k-1, j, i).x);
                    dvdz = 0.5 * (ucat(k+1, j, i).y - ucat(k-1, j, i).y);
                    dwdz = 0.5 * (ucat(k+1, j, i).z - ucat(k-1, j, i).z);
                }

                // Transform to physical space
                double du_dx, dv_dx, dw_dx;
                double du_dy, dv_dy, dw_dy;
                double du_dz, dv_dz, dw_dz;

                StrainRateTensor::computePhysicalGradients(
                    csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
                    dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
                    du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
                );

                // Compute strain rate magnitude
                double Sabs = StrainRateTensor::computeMagnitude(
                    du_dx, du_dy, du_dz,
                    dv_dx, dv_dy, dv_dz,
                    dw_dx, dw_dy, dw_dz
                );

                // Filter width: Δ = (cell volume)^(1/3) = (1/aj)^(1/3)
                double filter = Kokkos::pow(1.0 / aj(k, j, i), 1.0 / 3.0);

                // Smagorinsky model: ν_t = Cs² * Δ² * |S|
                nu_t(k, j, i) = Cs(k, j, i) * filter * filter * Sabs;
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute eddy viscosity with constant Cs
     */
    static void computeEddyViscosityConstantCs(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& csi,
        const VectorView& eta,
        const VectorView& zet,
        const ScalarView& aj,
        double Cs_value,
        ScalarView& nu_t,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const double solid = 0.1;

        Kokkos::deep_copy(nu_t, 0.0);

        Kokkos::parallel_for("SmagorinskyConstantCs",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (nvert(k, j, i) > 1.1) {
                    nu_t(k, j, i) = 0.0;
                    return;
                }

                double csi0 = csi(k, j, i).x, csi1 = csi(k, j, i).y, csi2 = csi(k, j, i).z;
                double eta0 = eta(k, j, i).x, eta1 = eta(k, j, i).y, eta2 = eta(k, j, i).z;
                double zet0 = zet(k, j, i).x, zet1 = zet(k, j, i).y, zet2 = zet(k, j, i).z;
                double ajc = aj(k, j, i);

                // Compute gradients (simplified version for constant Cs)
                double dudc = 0.5 * (ucat(k, j, i+1).x - ucat(k, j, i-1).x);
                double dvdc = 0.5 * (ucat(k, j, i+1).y - ucat(k, j, i-1).y);
                double dwdc = 0.5 * (ucat(k, j, i+1).z - ucat(k, j, i-1).z);

                double dude = 0.5 * (ucat(k, j+1, i).x - ucat(k, j-1, i).x);
                double dvde = 0.5 * (ucat(k, j+1, i).y - ucat(k, j-1, i).y);
                double dwde = 0.5 * (ucat(k, j+1, i).z - ucat(k, j-1, i).z);

                double dudz = 0.5 * (ucat(k+1, j, i).x - ucat(k-1, j, i).x);
                double dvdz = 0.5 * (ucat(k+1, j, i).y - ucat(k-1, j, i).y);
                double dwdz = 0.5 * (ucat(k+1, j, i).z - ucat(k-1, j, i).z);

                double du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz;
                StrainRateTensor::computePhysicalGradients(
                    csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
                    dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
                    du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
                );

                double Sabs = StrainRateTensor::computeMagnitude(
                    du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz
                );

                double filter = Kokkos::pow(1.0 / aj(k, j, i), 1.0 / 3.0);
                nu_t(k, j, i) = Cs_value * filter * filter * Sabs;
            }
        );

        Kokkos::fence();
    }
};

/**
 * @brief Dynamic Smagorinsky model kernel
 *
 * Computes the dynamic Smagorinsky constant using the Germano identity.
 * Uses test filtering at scale 2Δ.
 */
class DynamicSmagorinskyKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    // Small number to prevent division by zero
    static constexpr double EPS = 1.0e-7;
    // Minimum Cs near walls
    static constexpr double WALL_CS = 0.001;

    /**
     * @brief Apply test filter (averaging over 3x3x3 stencil)
     */
    KOKKOS_INLINE_FUNCTION
    static double testFilter(
        const ScalarView& field,
        const ScalarView& nvert,
        const ScalarView& aj,
        int k, int j, int i,
        double solid_threshold
    ) {
        double sum = 0.0;
        double weight_sum = 0.0;

        for (int dk = -1; dk <= 1; ++dk) {
            for (int dj = -1; dj <= 1; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    int kk = k + dk;
                    int jj = j + dj;
                    int ii = i + di;

                    if (nvert(kk, jj, ii) > solid_threshold) {
                        continue;
                    }

                    double w = 1.0 / aj(kk, jj, ii);
                    sum += w * field(kk, jj, ii);
                    weight_sum += w;
                }
            }
        }

        return (weight_sum > 0) ? sum / weight_sum : 0.0;
    }

    /**
     * @brief Compute dynamic Smagorinsky constant
     *
     * Uses the Germano identity and Lilly's least-squares approach:
     *   Cs² = <Lij Mij> / <Mij Mij>
     *
     * where:
     *   Lij = <ui uj> - <ui><uj> (Leonard stress)
     *   Mij = 2Δ²|<S>|<Sij> - 2(2Δ)²<|S|Sij> (model stress difference)
     *
     * @param ucat Cartesian velocity
     * @param nvert Solid marker
     * @param csi, eta, zet Metrics
     * @param aj Jacobian
     * @param Cs Output: dynamic Smagorinsky constant
     * @param domain Domain info
     */
    static void computeDynamicCs(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& csi,
        const VectorView& eta,
        const VectorView& zet,
        const ScalarView& aj,
        ScalarView& Cs,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const double solid = 0.1;

        // Allocate temporary arrays for LM and MM
        ScalarView LM("LM", domain.mz, domain.my, domain.mx);
        ScalarView MM("MM", domain.mz, domain.my, domain.mx);

        Kokkos::deep_copy(LM, 0.0);
        Kokkos::deep_copy(MM, 0.0);
        Kokkos::deep_copy(Cs, WALL_CS);

        // Step 1: Compute strain rate magnitude field
        ScalarView Sabs("Sabs", domain.mz, domain.my, domain.mx);
        Kokkos::deep_copy(Sabs, 0.0);

        Kokkos::parallel_for("ComputeStrainRate",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (nvert(k, j, i) > 1.1) return;

                double csi0 = csi(k, j, i).x, csi1 = csi(k, j, i).y, csi2 = csi(k, j, i).z;
                double eta0 = eta(k, j, i).x, eta1 = eta(k, j, i).y, eta2 = eta(k, j, i).z;
                double zet0 = zet(k, j, i).x, zet1 = zet(k, j, i).y, zet2 = zet(k, j, i).z;
                double ajc = aj(k, j, i);

                double dudc = 0.5 * (ucat(k, j, i+1).x - ucat(k, j, i-1).x);
                double dvdc = 0.5 * (ucat(k, j, i+1).y - ucat(k, j, i-1).y);
                double dwdc = 0.5 * (ucat(k, j, i+1).z - ucat(k, j, i-1).z);
                double dude = 0.5 * (ucat(k, j+1, i).x - ucat(k, j-1, i).x);
                double dvde = 0.5 * (ucat(k, j+1, i).y - ucat(k, j-1, i).y);
                double dwde = 0.5 * (ucat(k, j+1, i).z - ucat(k, j-1, i).z);
                double dudz = 0.5 * (ucat(k+1, j, i).x - ucat(k-1, j, i).x);
                double dvdz = 0.5 * (ucat(k+1, j, i).y - ucat(k-1, j, i).y);
                double dwdz = 0.5 * (ucat(k+1, j, i).z - ucat(k-1, j, i).z);

                double du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz;
                StrainRateTensor::computePhysicalGradients(
                    csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
                    dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
                    du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
                );

                Sabs(k, j, i) = StrainRateTensor::computeMagnitude(
                    du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz
                );
            }
        );

        Kokkos::fence();

        // Step 2: Compute Cs using Germano identity (simplified)
        Kokkos::parallel_for("ComputeDynamicCs",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (nvert(k, j, i) > solid) {
                    Cs(k, j, i) = WALL_CS;
                    return;
                }

                // Filter width at grid and test levels
                double delta = Kokkos::pow(1.0 / aj(k, j, i), 1.0 / 3.0);
                double delta_test = 2.0 * delta;

                // Get strain rate
                double S = Sabs(k, j, i);

                // Test-filtered strain rate (simplified averaging)
                double S_test = testFilter(Sabs, nvert, aj, k, j, i, solid);

                // Simplified dynamic procedure
                // Cs = delta² |S|² / (delta_test² |S_test|²)
                double numerator = delta * delta * S * S;
                double denominator = delta_test * delta_test * S_test * S_test + EPS;

                double Cs_sq = Kokkos::fmax(numerator / denominator, 0.0);
                Cs_sq = Kokkos::fmin(Cs_sq, 0.04);  // Cap at Cs² ≤ 0.04 (Cs ≤ 0.2)

                Cs(k, j, i) = Cs_sq;
            }
        );

        Kokkos::fence();
    }
};

/**
 * @brief Wall-damped eddy viscosity
 *
 * Applies van Driest damping near walls.
 */
class WallDampingKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;

    // van Driest constant
    static constexpr double A_PLUS = 26.0;

    /**
     * @brief van Driest damping function
     *
     * D = 1 - exp(-y⁺/A⁺)
     */
    KOKKOS_INLINE_FUNCTION
    static double vanDriestDamping(double y_plus) {
        return 1.0 - Kokkos::exp(-y_plus / A_PLUS);
    }

    /**
     * @brief Apply wall damping to eddy viscosity
     *
     * ν_t = ν_t * D²
     */
    static void applyWallDamping(
        ScalarView& nu_t,
        const ScalarView& y_plus,
        const ScalarView& nvert,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("WallDamping",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (nvert(k, j, i) > 0.1) return;

                double D = vanDriestDamping(y_plus(k, j, i));
                nu_t(k, j, i) *= D * D;
            }
        );

        Kokkos::fence();
    }
};

} // namespace les
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_LES_KERNEL_HPP
