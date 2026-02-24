/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Level-Set GPU Kernels                                          *
 * Phase 4: Level-Set Advection with WENO Schemes                 *
 ******************************************************************/

#ifndef VFSWIND_LEVELSET_KERNEL_HPP
#define VFSWIND_LEVELSET_KERNEL_HPP

#ifdef ENABLE_GPU

#include "../gpu_config.hpp"
#include "petsc_kokkos.hpp"

namespace vfswind {
namespace gpu {
namespace levelset {

/**
 * @brief WENO scheme helper functions
 *
 * Implements Weighted Essentially Non-Oscillatory (WENO) reconstruction
 * for high-order level-set advection.
 */
struct WENOSchemes {

    /**
     * @brief Sign function
     */
    KOKKOS_INLINE_FUNCTION
    static double sign(double x) {
        if (x > 0.0) return 1.0;
        if (x < 0.0) return -1.0;
        return 0.0;
    }

    /**
     * @brief Minmod limiter for UNO scheme
     */
    KOKKOS_INLINE_FUNCTION
    static double minmod(double m1, double m2) {
        return 0.5 * (sign(m1) + sign(m2)) * Kokkos::fmin(Kokkos::fabs(m1), Kokkos::fabs(m2));
    }

    /**
     * @brief ENO2 reconstruction
     *
     * Second-order Essentially Non-Oscillatory scheme
     *
     * @param f0, f1, f2, f3 Stencil values
     * @param wavespeed Local wave speed (determines upwind direction)
     * @return Reconstructed interface value
     */
    KOKKOS_INLINE_FUNCTION
    static double eno2(double f0, double f1, double f2, double f3, double wavespeed) {
        if (wavespeed > 0) {
            return f1 + 0.5 * minmod(f2 - f1, f1 - f0);
        } else {
            return f2 - 0.5 * minmod(f3 - f2, f2 - f1);
        }
    }

    /**
     * @brief WENO3 reconstruction
     *
     * Third-order Weighted Essentially Non-Oscillatory scheme
     * Uses two candidate stencils with smoothness-based weighting.
     *
     * @param f0, f1, f2, f3 Stencil values (4-point stencil)
     * @param wavespeed Local wave speed (determines upwind direction)
     * @return Reconstructed interface value
     */
    KOKKOS_INLINE_FUNCTION
    static double weno3(double f0, double f1, double f2, double f3, double wavespeed) {
        double fL, fC, fR;

        if (wavespeed > 0) {
            fL = f0;
            fC = f1;
            fR = f2;
        } else {
            // Mirror for downwind
            fL = f3;
            fC = f2;
            fR = f1;
        }

        // Ideal weights
        const double d0 = 2.0 / 3.0;
        const double d1 = 1.0 / 3.0;
        const double eps = 1.0e-6;

        // Smoothness indicators
        double beta0 = (fC - fR) * (fC - fR);
        double beta1 = (fL - fC) * (fL - fC);

        // Nonlinear weights
        double alpha0 = d0 / ((eps + beta0) * (eps + beta0));
        double alpha1 = d1 / ((eps + beta1) * (eps + beta1));
        double sum_alpha = alpha0 + alpha1;

        double w0 = alpha0 / sum_alpha;
        double w1 = alpha1 / sum_alpha;

        // Candidate polynomials
        double u0 = 0.5 * fC + 0.5 * fR;
        double u1 = -0.5 * fL + 1.5 * fC;

        return w0 * u0 + w1 * u1;
    }

    /**
     * @brief WENO5 reconstruction
     *
     * Fifth-order Weighted Essentially Non-Oscillatory scheme
     * Reference: "WENO Schemes and Discontinuous Galerkin Methods for CFD"
     *            by Shu, ICASE
     *
     * @param f0-f5 Stencil values (6-point stencil)
     * @param wavespeed Local wave speed (determines upwind direction)
     * @return Reconstructed interface value
     */
    KOKKOS_INLINE_FUNCTION
    static double weno5(double f0, double f1, double f2, double f3,
                        double f4, double f5, double wavespeed) {
        double A, B, C, D, E;

        if (wavespeed > 0) {
            A = f0; B = f1; C = f2; D = f3; E = f4;
        } else {
            // Mirror for downwind
            A = f5; B = f4; C = f3; D = f2; E = f1;
        }

        const double eps = 1.0e-6;

        // Ideal weights (upwind-biased)
        const double d0 = 0.3;
        const double d1 = 0.6;
        const double d2 = 0.1;

        // Smoothness indicators (from Jiang & Shu 1996)
        double beta0 = (13.0 / 12.0) * Kokkos::pow(A - 2.0 * B + C, 2.0)
                     + (1.0 / 4.0) * Kokkos::pow(A - 4.0 * B + 3.0 * C, 2.0);

        double beta1 = (13.0 / 12.0) * Kokkos::pow(B - 2.0 * C + D, 2.0)
                     + (1.0 / 4.0) * Kokkos::pow(B - D, 2.0);

        double beta2 = (13.0 / 12.0) * Kokkos::pow(C - 2.0 * D + E, 2.0)
                     + (1.0 / 4.0) * Kokkos::pow(3.0 * C - 4.0 * D + E, 2.0);

        // Nonlinear weights
        double alpha0 = d0 / ((eps + beta0) * (eps + beta0));
        double alpha1 = d1 / ((eps + beta1) * (eps + beta1));
        double alpha2 = d2 / ((eps + beta2) * (eps + beta2));
        double sum_alpha = alpha0 + alpha1 + alpha2;

        double w0 = alpha0 / sum_alpha;
        double w1 = alpha1 / sum_alpha;
        double w2 = alpha2 / sum_alpha;

        // Candidate polynomials
        double u0 = (2.0 / 6.0) * A - (7.0 / 6.0) * B + (11.0 / 6.0) * C;
        double u1 = (-1.0 / 6.0) * B + (5.0 / 6.0) * C + (2.0 / 6.0) * D;
        double u2 = (2.0 / 6.0) * C + (5.0 / 6.0) * D - (1.0 / 6.0) * E;

        return w0 * u0 + w1 * u1 + w2 * u2;
    }
};

/**
 * @brief Level-set advection kernel
 *
 * Advects the level-set function using high-order WENO reconstruction.
 * Supports both 3rd and 5th order schemes.
 */
class LevelsetAdvectionKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    /**
     * @brief Advection scheme type
     */
    enum class Scheme {
        ENO2,
        WENO3,
        WENO5
    };

    /**
     * @brief Compute level-set advection RHS using WENO3
     *
     * Computes: RHS = -u · ∇φ using dimension-by-dimension splitting
     *
     * @param levelset Level-set field φ (input)
     * @param ucont Contravariant velocity (input)
     * @param nvert Solid cell marker (input)
     * @param aj Jacobian (input)
     * @param rhs RHS output
     * @param domain Domain info
     * @param scheme Advection scheme to use
     */
    static void computeAdvectionRHS(
        const ScalarView& levelset,
        const VectorView& ucont,
        const ScalarView& nvert,
        const ScalarView& aj,
        ScalarView& rhs,
        const KernelDomainInfo& domain,
        Scheme scheme = Scheme::WENO3
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        // Initialize RHS to zero
        Kokkos::deep_copy(rhs, 0.0);

        // Compute advection term dimension by dimension
        Kokkos::parallel_for("LevelsetAdvection_WENO",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Skip solid cells
                if (nvert(k, j, i) > 0.1) {
                    rhs(k, j, i) = 0.0;
                    return;
                }

                double flux_i = 0.0, flux_j = 0.0, flux_k = 0.0;

                // ============================================================
                // I-direction flux (using ucont.x as wave speed)
                // ============================================================
                {
                    double wavespeed_im = ucont(k, j, i-1).x;
                    double wavespeed_ip = ucont(k, j, i).x;

                    // Get stencil values for WENO3 (4-point stencil)
                    double phi_im2 = levelset(k, j, i-2);
                    double phi_im1 = levelset(k, j, i-1);
                    double phi_i   = levelset(k, j, i);
                    double phi_ip1 = levelset(k, j, i+1);
                    double phi_ip2 = levelset(k, j, i+2);

                    // Reconstruct at i-1/2 and i+1/2 faces
                    double phi_face_im = WENOSchemes::weno3(phi_im2, phi_im1, phi_i, phi_ip1, wavespeed_im);
                    double phi_face_ip = WENOSchemes::weno3(phi_im1, phi_i, phi_ip1, phi_ip2, wavespeed_ip);

                    // Flux difference
                    flux_i = wavespeed_ip * phi_face_ip - wavespeed_im * phi_face_im;
                }

                // ============================================================
                // J-direction flux (using ucont.y as wave speed)
                // ============================================================
                {
                    double wavespeed_jm = ucont(k, j-1, i).y;
                    double wavespeed_jp = ucont(k, j, i).y;

                    double phi_jm2 = levelset(k, j-2, i);
                    double phi_jm1 = levelset(k, j-1, i);
                    double phi_j   = levelset(k, j, i);
                    double phi_jp1 = levelset(k, j+1, i);
                    double phi_jp2 = levelset(k, j+2, i);

                    double phi_face_jm = WENOSchemes::weno3(phi_jm2, phi_jm1, phi_j, phi_jp1, wavespeed_jm);
                    double phi_face_jp = WENOSchemes::weno3(phi_jm1, phi_j, phi_jp1, phi_jp2, wavespeed_jp);

                    flux_j = wavespeed_jp * phi_face_jp - wavespeed_jm * phi_face_jm;
                }

                // ============================================================
                // K-direction flux (using ucont.z as wave speed)
                // ============================================================
                {
                    double wavespeed_km = ucont(k-1, j, i).z;
                    double wavespeed_kp = ucont(k, j, i).z;

                    double phi_km2 = levelset(k-2, j, i);
                    double phi_km1 = levelset(k-1, j, i);
                    double phi_k   = levelset(k, j, i);
                    double phi_kp1 = levelset(k+1, j, i);
                    double phi_kp2 = levelset(k+2, j, i);

                    double phi_face_km = WENOSchemes::weno3(phi_km2, phi_km1, phi_k, phi_kp1, wavespeed_km);
                    double phi_face_kp = WENOSchemes::weno3(phi_km1, phi_k, phi_kp1, phi_kp2, wavespeed_kp);

                    flux_k = wavespeed_kp * phi_face_kp - wavespeed_km * phi_face_km;
                }

                // Combine fluxes (negative because RHS = -∇·(u*φ))
                rhs(k, j, i) = -(flux_i + flux_j + flux_k) * aj(k, j, i);
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Update level-set field with explicit Euler step
     *
     * @param levelset Level-set field to update (in/out)
     * @param rhs RHS from advection
     * @param nvert Solid cell marker
     * @param dt Time step
     * @param domain Domain info
     */
    static void updateExplicit(
        ScalarView& levelset,
        const ScalarView& rhs,
        const ScalarView& nvert,
        double dt,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("LevelsetUpdate",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                if (nvert(k, j, i) < 0.1) {
                    levelset(k, j, i) += dt * rhs(k, j, i);
                }
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute Godunov Hamiltonian for reinitialization
     *
     * Used in the reinitialization equation:
     * ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0
     *
     * @param dxm, dxp Derivatives in x (minus/plus)
     * @param dym, dyp Derivatives in y (minus/plus)
     * @param dzm, dzp Derivatives in z (minus/plus)
     * @param sign Sign of original level-set
     * @return Godunov Hamiltonian value
     */
    KOKKOS_INLINE_FUNCTION
    static double godunovHamiltonian(
        double dxm, double dxp,
        double dym, double dyp,
        double dzm, double dzp,
        double sign_phi
    ) {
        double grad_sq = 0.0;

        if (sign_phi > 0) {
            // For positive region: use upwind scheme
            double ax = Kokkos::fmax(dxm, 0.0);
            double bx = Kokkos::fmin(dxp, 0.0);
            double ay = Kokkos::fmax(dym, 0.0);
            double by = Kokkos::fmin(dyp, 0.0);
            double az = Kokkos::fmax(dzm, 0.0);
            double bz = Kokkos::fmin(dzp, 0.0);

            grad_sq = Kokkos::fmax(ax * ax, bx * bx)
                    + Kokkos::fmax(ay * ay, by * by)
                    + Kokkos::fmax(az * az, bz * bz);
        } else if (sign_phi < 0) {
            // For negative region: use downwind scheme
            double ax = Kokkos::fmin(dxm, 0.0);
            double bx = Kokkos::fmax(dxp, 0.0);
            double ay = Kokkos::fmin(dym, 0.0);
            double by = Kokkos::fmax(dyp, 0.0);
            double az = Kokkos::fmin(dzm, 0.0);
            double bz = Kokkos::fmax(dzp, 0.0);

            grad_sq = Kokkos::fmax(ax * ax, bx * bx)
                    + Kokkos::fmax(ay * ay, by * by)
                    + Kokkos::fmax(az * az, bz * bz);
        }

        return Kokkos::sqrt(grad_sq);
    }
};

/**
 * @brief Level-set reinitialization kernel
 *
 * Reinitializes the level-set function to a signed distance function
 * using the equation: ∂φ/∂τ + S(φ₀)(|∇φ| - 1) = 0
 */
class LevelsetReinitKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;

    /**
     * @brief Smoothed sign function
     *
     * S(φ) = φ / √(φ² + |∇φ|²h²)
     */
    KOKKOS_INLINE_FUNCTION
    static double smoothedSign(double phi, double grad_phi_mag, double h) {
        const double eps = h * h;
        return phi / Kokkos::sqrt(phi * phi + grad_phi_mag * grad_phi_mag * eps);
    }

    /**
     * @brief Compute one step of explicit reinitialization
     *
     * @param phi Current level-set (input/output)
     * @param phi0 Original level-set (for sign)
     * @param nvert Solid marker
     * @param aj Jacobian
     * @param dtau Pseudo-time step
     * @param h Grid spacing
     * @param domain Domain info
     * @return Maximum change (for convergence check)
     */
    static double reinitStep(
        ScalarView& phi,
        const ScalarView& phi0,
        const ScalarView& nvert,
        const ScalarView& aj,
        double dtau,
        double h,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        double max_change = 0.0;

        Kokkos::parallel_reduce("LevelsetReinit",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i, double& lmax) {
                if (nvert(k, j, i) > 0.1) return;

                // Compute one-sided derivatives
                double dx_m = (phi(k, j, i) - phi(k, j, i-1)) / h;
                double dx_p = (phi(k, j, i+1) - phi(k, j, i)) / h;
                double dy_m = (phi(k, j, i) - phi(k, j-1, i)) / h;
                double dy_p = (phi(k, j+1, i) - phi(k, j, i)) / h;
                double dz_m = (phi(k, j, i) - phi(k-1, j, i)) / h;
                double dz_p = (phi(k+1, j, i) - phi(k, j, i)) / h;

                // Approximate gradient magnitude for sign function
                double grad_mag = Kokkos::sqrt(
                    0.25 * (dx_m + dx_p) * (dx_m + dx_p) +
                    0.25 * (dy_m + dy_p) * (dy_m + dy_p) +
                    0.25 * (dz_m + dz_p) * (dz_m + dz_p)
                );

                // Smoothed sign function
                double S = smoothedSign(phi0(k, j, i), grad_mag, h);

                // Godunov Hamiltonian
                double H = LevelsetAdvectionKernel::godunovHamiltonian(
                    dx_m, dx_p, dy_m, dy_p, dz_m, dz_p, S
                );

                // Update
                double dphi = -dtau * S * (H - 1.0);
                phi(k, j, i) += dphi;

                lmax = Kokkos::fmax(lmax, Kokkos::fabs(dphi));
            },
            Kokkos::Max<double>(max_change)
        );

        Kokkos::fence();
        return max_change;
    }

    /**
     * @brief Run full reinitialization until convergence
     *
     * @param phi Level-set to reinitialize
     * @param phi0 Original level-set
     * @param nvert Solid marker
     * @param aj Jacobian
     * @param h Grid spacing
     * @param domain Domain info
     * @param max_iter Maximum iterations
     * @param tol Convergence tolerance
     * @return Number of iterations
     */
    static int reinitialize(
        ScalarView& phi,
        const ScalarView& phi0,
        const ScalarView& nvert,
        const ScalarView& aj,
        double h,
        const KernelDomainInfo& domain,
        int max_iter = 100,
        double tol = 1.0e-6
    ) {
        // CFL condition for reinitialization
        const double dtau = 0.5 * h;

        for (int iter = 0; iter < max_iter; ++iter) {
            double change = reinitStep(phi, phi0, nvert, aj, dtau, h, domain);

            if (change < tol) {
                return iter + 1;
            }
        }

        return max_iter;
    }
};

/**
 * @brief Compute density and viscosity from level-set
 *
 * Uses Heaviside smoothing for two-phase flows.
 */
class LevelsetPropertiesKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;

    /**
     * @brief Smoothed Heaviside function
     */
    KOKKOS_INLINE_FUNCTION
    static double heaviside(double phi, double epsilon) {
        if (phi < -epsilon) {
            return 0.0;
        } else if (phi > epsilon) {
            return 1.0;
        } else {
            return 0.5 * (1.0 + phi / epsilon +
                   (1.0 / M_PI) * Kokkos::sin(M_PI * phi / epsilon));
        }
    }

    /**
     * @brief Compute density field from level-set
     *
     * ρ = ρ₁ + (ρ₂ - ρ₁) H(φ)
     *
     * @param phi Level-set field
     * @param rho Density output
     * @param rho1, rho2 Fluid densities
     * @param epsilon Interface thickness
     * @param domain Domain info
     */
    static void computeDensity(
        const ScalarView& phi,
        ScalarView& rho,
        double rho1,
        double rho2,
        double epsilon,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("ComputeDensity",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                double H = heaviside(phi(k, j, i), epsilon);
                rho(k, j, i) = rho1 + (rho2 - rho1) * H;
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute viscosity field from level-set
     *
     * μ = μ₁ + (μ₂ - μ₁) H(φ)
     */
    static void computeViscosity(
        const ScalarView& phi,
        ScalarView& mu,
        double mu1,
        double mu2,
        double epsilon,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("ComputeViscosity",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                double H = heaviside(phi(k, j, i), epsilon);
                mu(k, j, i) = mu1 + (mu2 - mu1) * H;
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Compute surface tension force
     *
     * F_st = σ κ δ(φ) ∇φ
     *
     * where κ is the curvature and δ is the Dirac delta.
     */
    KOKKOS_INLINE_FUNCTION
    static double diracDelta(double phi, double epsilon) {
        if (Kokkos::fabs(phi) > epsilon) {
            return 0.0;
        }
        return (1.0 / (2.0 * epsilon)) * (1.0 + Kokkos::cos(M_PI * phi / epsilon));
    }
};

} // namespace levelset
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_LEVELSET_KERNEL_HPP
