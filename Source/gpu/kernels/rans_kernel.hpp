/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * RANS GPU Kernels                                               *
 * Phase 4: k-omega and SST Turbulence Models                     *
 ******************************************************************/

#ifndef VFSWIND_RANS_KERNEL_HPP
#define VFSWIND_RANS_KERNEL_HPP

#ifdef ENABLE_GPU

#include "../gpu_config.hpp"
#include "petsc_kokkos.hpp"
#include "les_kernel.hpp"  // For StrainRateTensor

namespace vfswind {
namespace gpu {
namespace rans {

/**
 * @brief k-omega model constants
 *
 * Based on Wilcox (1988, 2006) and Menter SST (1994)
 */
struct KOmegaConstants {
    // Wilcox k-omega constants
    static constexpr double BETA_STAR = 0.09;
    static constexpr double BETA1 = 0.075;
    static constexpr double BETA2 = 0.0828;
    static constexpr double SIGMA = 0.5;
    static constexpr double SIGMA_STAR = 0.5;
    static constexpr double ALPHA1 = 5.0 / 9.0;
    static constexpr double ALPHA2 = 0.44;
    static constexpr double A1 = 0.31;

    // SST blending constants
    static constexpr double SIGMA_K1 = 0.85;
    static constexpr double SIGMA_K2 = 1.0;
    static constexpr double SIGMA_O1 = 0.5;
    static constexpr double SIGMA_O2 = 0.856;

    // Low-Re constants (Wilcox)
    static constexpr double RB = 8.0;
    static constexpr double RK = 6.0;
    static constexpr double RW = 2.7;
    static constexpr double ALPHA0_STAR = BETA1 / 3.0;
    static constexpr double ALPHA0 = 0.1;

    /**
     * @brief Wall omega boundary condition
     *
     * ω_wall = 6 / (β₁ Re_ν y²)
     */
    KOKKOS_INLINE_FUNCTION
    static double wallOmega(double Re_nu, double dist) {
        return 6.0 / (BETA1 * Re_nu * dist * dist);
    }
};

/**
 * @brief RANS model type enumeration
 */
enum class RANSModel {
    WILCOX_LOW_RE = 1,   // Wilcox low-Re k-omega
    WILCOX_HIGH_RE = 2,  // Wilcox high-Re k-omega
    SST_MENTER = 3       // Menter SST k-omega
};

/**
 * @brief k-omega two-equation data structure
 */
template<typename Space = DeviceMemorySpace>
struct KOmegaData {
    Kokkos::View<double***, Space> k;       // Turbulent kinetic energy
    Kokkos::View<double***, Space> omega;   // Specific dissipation rate
    Kokkos::View<double***, Space> k_rhs;   // k equation RHS
    Kokkos::View<double***, Space> omega_rhs; // omega equation RHS
    Kokkos::View<double***, Space> nu_t;    // Turbulent viscosity
    Kokkos::View<double***, Space> F1;      // SST blending function

    void allocate(int mz, int my, int mx) {
        k = Kokkos::View<double***, Space>("k", mz, my, mx);
        omega = Kokkos::View<double***, Space>("omega", mz, my, mx);
        k_rhs = Kokkos::View<double***, Space>("k_rhs", mz, my, mx);
        omega_rhs = Kokkos::View<double***, Space>("omega_rhs", mz, my, mx);
        nu_t = Kokkos::View<double***, Space>("nu_t", mz, my, mx);
        F1 = Kokkos::View<double***, Space>("F1", mz, my, mx);
    }
};

/**
 * @brief k-omega model kernel
 */
class KOmegaKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;
    using Constants = KOmegaConstants;

    /**
     * @brief Get model coefficients for low-Re correction
     *
     * Based on Wilcox (2006) pp.75, 108-109
     */
    KOKKOS_INLINE_FUNCTION
    static void getAlphaBetaStar(
        double Re_nu, double K, double O,
        RANSModel model,
        double& alpha, double& alpha_star, double& beta_star
    ) {
        if (model == RANSModel::WILCOX_LOW_RE) {
            double ReT = K / O * Re_nu;

            alpha_star = (Constants::ALPHA0_STAR + ReT / Constants::RK)
                       / (1.0 + ReT / Constants::RK);

            alpha = Constants::ALPHA1
                  * (Constants::ALPHA0 + ReT / Constants::RW)
                  / (1.0 + ReT / Constants::RW)
                  / alpha_star;

            double Re_ratio = Kokkos::pow(ReT / Constants::RB, 4.0);
            beta_star = 0.09 * (5.0 / 18.0 + Re_ratio) / (1.0 + Re_ratio);
        } else {
            alpha = Constants::ALPHA1;
            alpha_star = 1.0;
            beta_star = Constants::BETA_STAR;
        }
    }

    /**
     * @brief Compute SST blending function F1
     *
     * F1 blends between k-omega (near wall) and k-epsilon (free stream)
     *
     * F1 = tanh(arg1⁴)
     * arg1 = min(max(sqrt(k)/(β*ω*y), 500ν/(y²ω)), 4ρσω2*k/(CDkω*y²))
     */
    static void computeF1(
        const ScalarView& k,
        const ScalarView& omega,
        const ScalarView& distance,
        const ScalarView& nvert,
        ScalarView& F1,
        double Re_nu,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const double nu = 1.0 / Re_nu;

        Kokkos::parallel_for("ComputeF1",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int kk, int j, int i) {
                if (nvert(kk, j, i) > 0.1) {
                    F1(kk, j, i) = 1.0;
                    return;
                }

                double K = Kokkos::fmax(k(kk, j, i), 1.0e-10);
                double O = Kokkos::fmax(omega(kk, j, i), 1.0e-10);
                double y = Kokkos::fmax(distance(kk, j, i), 1.0e-10);

                // Cross-diffusion term CDkω
                double dkdx = 0.5 * (k(kk, j, i+1) - k(kk, j, i-1));
                double dkdy = 0.5 * (k(kk, j+1, i) - k(kk, j-1, i));
                double dkdz = 0.5 * (k(kk+1, j, i) - k(kk-1, j, i));
                double dodx = 0.5 * (omega(kk, j, i+1) - omega(kk, j, i-1));
                double dody = 0.5 * (omega(kk, j+1, i) - omega(kk, j-1, i));
                double dodz = 0.5 * (omega(kk+1, j, i) - omega(kk-1, j, i));

                double CDkw = Kokkos::fmax(
                    2.0 * Constants::SIGMA_O2 / O * (dkdx * dodx + dkdy * dody + dkdz * dodz),
                    1.0e-10
                );

                // arg1 components
                double term1 = Kokkos::sqrt(K) / (Constants::BETA_STAR * O * y);
                double term2 = 500.0 * nu / (y * y * O);
                double term3 = 4.0 * Constants::SIGMA_O2 * K / (CDkw * y * y);

                double arg1 = Kokkos::fmin(Kokkos::fmax(term1, term2), term3);

                F1(kk, j, i) = Kokkos::tanh(Kokkos::pow(arg1, 4.0));
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute turbulent viscosity
     *
     * For standard k-omega: ν_t = k / ω
     * For SST: ν_t = a1 * k / max(a1 * ω, |S| * F2)
     */
    static void computeTurbulentViscosity(
        const ScalarView& k,
        const ScalarView& omega,
        const ScalarView& nvert,
        const ScalarView& F1,
        ScalarView& nu_t,
        double Re_nu,
        RANSModel model,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("ComputeNuT",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int kk, int j, int i) {
                if (nvert(kk, j, i) > 1.1) {
                    nu_t(kk, j, i) = 0.0;
                    return;
                }

                double K = Kokkos::fmax(k(kk, j, i), 1.0e-10);
                double O = Kokkos::fmax(omega(kk, j, i), 1.0e-10);

                if (model == RANSModel::SST_MENTER) {
                    // SST limiter
                    nu_t(kk, j, i) = Constants::A1 * K
                                   / Kokkos::fmax(Constants::A1 * O, 1.0e-10);
                } else {
                    // Standard k-omega
                    double alpha, alpha_star, beta_star;
                    getAlphaBetaStar(Re_nu, K, O, model, alpha, alpha_star, beta_star);
                    nu_t(kk, j, i) = alpha_star * K / O;
                }
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute k-omega RHS (production, dissipation, diffusion)
     *
     * k-equation: ∂k/∂t = P_k - β*ωk + ∇·[(ν + σ_k ν_t)∇k]
     * ω-equation: ∂ω/∂t = αP_k/ν_t - βω² + ∇·[(ν + σ_ω ν_t)∇ω] + CDkω
     *
     * @param ucat Cartesian velocity
     * @param ucont Contravariant velocity
     * @param k Turbulent kinetic energy
     * @param omega Specific dissipation rate
     * @param nu_t Turbulent viscosity
     * @param nvert Solid marker
     * @param F1 SST blending function
     * @param csi, eta, zet Metrics
     * @param aj Jacobian
     * @param k_rhs Output k RHS
     * @param omega_rhs Output omega RHS
     * @param Re_nu Reynolds number
     * @param model RANS model type
     * @param domain Domain info
     */
    static void computeRHS(
        const VectorView& ucat,
        const VectorView& ucont,
        const ScalarView& k,
        const ScalarView& omega,
        const ScalarView& nu_t,
        const ScalarView& nvert,
        const ScalarView& F1,
        const VectorView& csi,
        const VectorView& eta,
        const VectorView& zet,
        const ScalarView& aj,
        ScalarView& k_rhs,
        ScalarView& omega_rhs,
        double Re_nu,
        RANSModel model,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const double nu = 1.0 / Re_nu;
        const double solid = 1.1;

        Kokkos::deep_copy(k_rhs, 0.0);
        Kokkos::deep_copy(omega_rhs, 0.0);

        Kokkos::parallel_for("KOmegaRHS",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int kk, int j, int i) {
                if (nvert(kk, j, i) > solid) {
                    k_rhs(kk, j, i) = 0.0;
                    omega_rhs(kk, j, i) = 0.0;
                    return;
                }

                // Get values
                double K = Kokkos::fmax(k(kk, j, i), 1.0e-10);
                double O = Kokkos::fmax(omega(kk, j, i), 1.0e-10);
                double nut = nu_t(kk, j, i);
                double ajc = aj(kk, j, i);

                // Get model coefficients
                double alpha, alpha_star, beta_star;
                getAlphaBetaStar(Re_nu, K, O, model, alpha, alpha_star, beta_star);

                // SST blending for beta and sigma
                double f1 = (model == RANSModel::SST_MENTER) ? F1(kk, j, i) : 1.0;
                double beta = f1 * Constants::BETA1 + (1.0 - f1) * Constants::BETA2;
                double sigma_k = f1 * Constants::SIGMA_K1 + (1.0 - f1) * Constants::SIGMA_K2;
                double sigma_o = f1 * Constants::SIGMA_O1 + (1.0 - f1) * Constants::SIGMA_O2;

                // Compute strain rate magnitude for production
                double csi0 = csi(kk, j, i).x, csi1 = csi(kk, j, i).y, csi2 = csi(kk, j, i).z;
                double eta0 = eta(kk, j, i).x, eta1 = eta(kk, j, i).y, eta2 = eta(kk, j, i).z;
                double zet0 = zet(kk, j, i).x, zet1 = zet(kk, j, i).y, zet2 = zet(kk, j, i).z;

                double dudc = 0.5 * (ucat(kk, j, i+1).x - ucat(kk, j, i-1).x);
                double dvdc = 0.5 * (ucat(kk, j, i+1).y - ucat(kk, j, i-1).y);
                double dwdc = 0.5 * (ucat(kk, j, i+1).z - ucat(kk, j, i-1).z);
                double dude = 0.5 * (ucat(kk, j+1, i).x - ucat(kk, j-1, i).x);
                double dvde = 0.5 * (ucat(kk, j+1, i).y - ucat(kk, j-1, i).y);
                double dwde = 0.5 * (ucat(kk, j+1, i).z - ucat(kk, j-1, i).z);
                double dudz = 0.5 * (ucat(kk+1, j, i).x - ucat(kk-1, j, i).x);
                double dvdz = 0.5 * (ucat(kk+1, j, i).y - ucat(kk-1, j, i).y);
                double dwdz = 0.5 * (ucat(kk+1, j, i).z - ucat(kk-1, j, i).z);

                double du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz;
                les::StrainRateTensor::computePhysicalGradients(
                    csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
                    dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
                    du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
                );

                double Sabs = les::StrainRateTensor::computeMagnitude(
                    du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz
                );

                // Production term: P_k = ν_t * S²
                double Pk = nut * Sabs * Sabs;

                // Dissipation terms
                double Dk = beta_star * O * K;        // k dissipation
                double Do = beta * O * O;              // omega dissipation

                // k-equation RHS: P_k - D_k
                k_rhs(kk, j, i) = Pk - Dk;

                // omega-equation RHS: α * P_k / ν_t - D_ω
                double Po = alpha * Sabs * Sabs;
                omega_rhs(kk, j, i) = Po - Do;

                // Cross-diffusion term for SST
                if (model == RANSModel::SST_MENTER) {
                    double dkdx = 0.5 * (k(kk, j, i+1) - k(kk, j, i-1));
                    double dkdy = 0.5 * (k(kk, j+1, i) - k(kk, j-1, i));
                    double dkdz = 0.5 * (k(kk+1, j, i) - k(kk-1, j, i));
                    double dodx = 0.5 * (omega(kk, j, i+1) - omega(kk, j, i-1));
                    double dody = 0.5 * (omega(kk, j+1, i) - omega(kk, j-1, i));
                    double dodz = 0.5 * (omega(kk+1, j, i) - omega(kk-1, j, i));

                    double CDkw = 2.0 * (1.0 - f1) * Constants::SIGMA_O2 / O
                                * (dkdx * dodx + dkdy * dody + dkdz * dodz);

                    omega_rhs(kk, j, i) += CDkw;
                }
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Apply wall boundary conditions for k and omega
     *
     * k_wall = 0 (no-slip)
     * omega_wall = 6ν / (β₁ y²) (asymptotic behavior)
     */
    static void applyWallBC(
        ScalarView& k,
        ScalarView& omega,
        const ScalarView& distance,
        const ScalarView& nvert,
        double Re_nu,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const double wall_threshold = 0.5;  // Near-wall marker

        Kokkos::parallel_for("KOmegaWallBC",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int kk, int j, int i) {
                // Check if this is a near-wall cell
                if (nvert(kk, j, i) > wall_threshold && nvert(kk, j, i) < 1.5) {
                    double y = distance(kk, j, i);
                    if (y < 1.0e-10) y = 1.0e-10;

                    k(kk, j, i) = 0.0;
                    omega(kk, j, i) = Constants::wallOmega(Re_nu, y);
                }
            }
        );

        Kokkos::fence();
    }
};

/**
 * @brief SST k-omega model kernel (Menter 1994)
 *
 * Combines k-omega near walls with k-epsilon in free stream
 */
class SSTKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;
    using Constants = KOmegaConstants;

    /**
     * @brief Compute F2 blending function for SST limiter
     *
     * F2 = tanh(arg2²)
     * arg2 = max(2*sqrt(k)/(β*ωy), 500ν/(y²ω))
     */
    static void computeF2(
        const ScalarView& k,
        const ScalarView& omega,
        const ScalarView& distance,
        const ScalarView& nvert,
        ScalarView& F2,
        double Re_nu,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;
        const double nu = 1.0 / Re_nu;

        Kokkos::parallel_for("ComputeF2",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int kk, int j, int i) {
                if (nvert(kk, j, i) > 0.1) {
                    F2(kk, j, i) = 0.0;
                    return;
                }

                double K = Kokkos::fmax(k(kk, j, i), 1.0e-10);
                double O = Kokkos::fmax(omega(kk, j, i), 1.0e-10);
                double y = Kokkos::fmax(distance(kk, j, i), 1.0e-10);

                double term1 = 2.0 * Kokkos::sqrt(K) / (Constants::BETA_STAR * O * y);
                double term2 = 500.0 * nu / (y * y * O);

                double arg2 = Kokkos::fmax(term1, term2);
                F2(kk, j, i) = Kokkos::tanh(arg2 * arg2);
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute SST turbulent viscosity with vorticity limiter
     *
     * ν_t = a1 * k / max(a1 * ω, Ω * F2)
     *
     * where Ω is the vorticity magnitude
     */
    static void computeTurbulentViscositySST(
        const ScalarView& k,
        const ScalarView& omega,
        const VectorView& ucat,
        const ScalarView& nvert,
        const ScalarView& F2,
        const VectorView& csi,
        const VectorView& eta,
        const VectorView& zet,
        const ScalarView& aj,
        ScalarView& nu_t,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("SSTNuT",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int kk, int j, int i) {
                if (nvert(kk, j, i) > 1.1) {
                    nu_t(kk, j, i) = 0.0;
                    return;
                }

                double K = Kokkos::fmax(k(kk, j, i), 1.0e-10);
                double O = Kokkos::fmax(omega(kk, j, i), 1.0e-10);
                double f2 = F2(kk, j, i);

                // Compute vorticity magnitude
                double csi0 = csi(kk, j, i).x, csi1 = csi(kk, j, i).y, csi2 = csi(kk, j, i).z;
                double eta0 = eta(kk, j, i).x, eta1 = eta(kk, j, i).y, eta2 = eta(kk, j, i).z;
                double zet0 = zet(kk, j, i).x, zet1 = zet(kk, j, i).y, zet2 = zet(kk, j, i).z;
                double ajc = aj(kk, j, i);

                double dudc = 0.5 * (ucat(kk, j, i+1).x - ucat(kk, j, i-1).x);
                double dvdc = 0.5 * (ucat(kk, j, i+1).y - ucat(kk, j, i-1).y);
                double dwdc = 0.5 * (ucat(kk, j, i+1).z - ucat(kk, j, i-1).z);
                double dude = 0.5 * (ucat(kk, j+1, i).x - ucat(kk, j-1, i).x);
                double dvde = 0.5 * (ucat(kk, j+1, i).y - ucat(kk, j-1, i).y);
                double dwde = 0.5 * (ucat(kk, j+1, i).z - ucat(kk, j-1, i).z);
                double dudz = 0.5 * (ucat(kk+1, j, i).x - ucat(kk-1, j, i).x);
                double dvdz = 0.5 * (ucat(kk+1, j, i).y - ucat(kk-1, j, i).y);
                double dwdz = 0.5 * (ucat(kk+1, j, i).z - ucat(kk-1, j, i).z);

                double du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz;
                les::StrainRateTensor::computePhysicalGradients(
                    csi0, csi1, csi2, eta0, eta1, eta2, zet0, zet1, zet2, ajc,
                    dudc, dvdc, dwdc, dude, dvde, dwde, dudz, dvdz, dwdz,
                    du_dx, dv_dx, dw_dx, du_dy, dv_dy, dw_dy, du_dz, dv_dz, dw_dz
                );

                // Vorticity components: Ω_i = ε_ijk ∂u_j/∂x_k
                double omega_x = dw_dy - dv_dz;
                double omega_y = du_dz - dw_dx;
                double omega_z = dv_dx - du_dy;

                double Omega_mag = Kokkos::sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);

                // SST limiter
                nu_t(kk, j, i) = Constants::A1 * K
                              / Kokkos::fmax(Constants::A1 * O, Omega_mag * f2);
            }
        );

        Kokkos::fence();
    }
};

} // namespace rans
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_RANS_KERNEL_HPP
