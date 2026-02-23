/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Convection Kernel for VFS-Wind                             *
 * Implements QUICK scheme convection on GPU via Kokkos           *
 ******************************************************************/

#ifndef VFSWIND_CONVECTION_KERNEL_HPP
#define VFSWIND_CONVECTION_KERNEL_HPP

#ifdef ENABLE_GPU

#include "petsc_kokkos.hpp"
#include <Kokkos_Core.hpp>

namespace vfswind {
namespace gpu {
namespace kernels {

/**
 * @brief GPU Convection kernel using QUICK scheme
 *
 * This kernel computes the convective terms for the Navier-Stokes equations
 * using the QUICK (Quadratic Upstream Interpolation for Convective Kinematics)
 * scheme. The convection term is: div(u * u)
 *
 * The computation is split into three phases:
 * 1. Compute i-direction fluxes -> fp1
 * 2. Compute j-direction fluxes -> fp2
 * 3. Compute k-direction fluxes -> fp3
 * 4. Combine fluxes to get final convection term
 */
class ConvectionKernel {
public:
    // View types
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;
    using HostScalarView = ScalarView3D<Kokkos::HostSpace>;
    using HostVectorView = VectorView3D<Kokkos::HostSpace>;

    /**
     * @brief Execute the convection kernel
     *
     * @param ucont Contravariant velocity components (input)
     * @param ucat Cartesian velocity components (input)
     * @param nvert Solid cell marker (input)
     * @param conv Convection term output (output)
     * @param domain Domain information
     */
    static void execute(
        const VectorView& ucont,
        const VectorView& ucat,
        const ScalarView& nvert,
        VectorView& conv,
        const KernelDomainInfo& domain
    );

private:
    // QUICK scheme coefficient
    static constexpr double QUICK_COEF = 0.125;

    /**
     * @brief Compute i-direction flux using QUICK scheme
     */
    static void computeFluxI(
        const VectorView& ucont,
        const VectorView& ucat,
        const ScalarView& nvert,
        VectorView& fp1,
        const KernelDomainInfo& domain
    );

    /**
     * @brief Compute j-direction flux using QUICK scheme
     */
    static void computeFluxJ(
        const VectorView& ucont,
        const VectorView& ucat,
        const ScalarView& nvert,
        VectorView& fp2,
        const KernelDomainInfo& domain
    );

    /**
     * @brief Compute k-direction flux using QUICK scheme
     */
    static void computeFluxK(
        const VectorView& ucont,
        const VectorView& ucat,
        const ScalarView& nvert,
        VectorView& fp3,
        const KernelDomainInfo& domain
    );

    /**
     * @brief Combine directional fluxes into final convection term
     */
    static void combineFluxes(
        const VectorView& fp1,
        const VectorView& fp2,
        const VectorView& fp3,
        VectorView& conv,
        const KernelDomainInfo& domain
    );
};

// ============================================================================
// Implementation
// ============================================================================

inline void ConvectionKernel::execute(
    const VectorView& ucont,
    const VectorView& ucat,
    const ScalarView& nvert,
    VectorView& conv,
    const KernelDomainInfo& domain
) {
    // Allocate temporary flux arrays
    VectorView fp1("fp1", domain.mz, domain.my, domain.mx);
    VectorView fp2("fp2", domain.mz, domain.my, domain.mx);
    VectorView fp3("fp3", domain.mz, domain.my, domain.mx);

    // Initialize to zero
    Kokkos::deep_copy(fp1, Cmpnts3());
    Kokkos::deep_copy(fp2, Cmpnts3());
    Kokkos::deep_copy(fp3, Cmpnts3());

    // Compute directional fluxes
    computeFluxI(ucont, ucat, nvert, fp1, domain);
    computeFluxJ(ucont, ucat, nvert, fp2, domain);
    computeFluxK(ucont, ucat, nvert, fp3, domain);

    // Combine fluxes
    combineFluxes(fp1, fp2, fp3, conv, domain);

    // Ensure all GPU work is done
    Kokkos::fence();
}

inline void ConvectionKernel::computeFluxI(
    const VectorView& ucont,
    const VectorView& ucat,
    const ScalarView& nvert,
    VectorView& fp1,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const int mx = domain.mx;
    const double coef = QUICK_COEF;

    Kokkos::parallel_for("Convection_FluxI",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs-1}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Compute face velocity
            double ucon = ucont(k, j, i).x * 0.5;
            double up = ucon + fabs(ucon);  // Max(ucon, 0)
            double um = ucon - fabs(ucon);  // Min(ucon, 0)

            Cmpnts3 flux;

            // Interior nodes - full QUICK stencil
            if (i > 0 && i < mx - 2 &&
                nvert(k, j, i+1) < 0.1 &&
                nvert(k, j, i-1) < 0.1) {
                // QUICK scheme: q(face) = q(upwind) + coef*(-q(far) - 2*q(near) + 3*q(upwind))
                flux.x =
                    um * (coef * (-ucat(k,j,i+2).x - 2.0*ucat(k,j,i+1).x + 3.0*ucat(k,j,i  ).x) + ucat(k,j,i+1).x) +
                    up * (coef * (-ucat(k,j,i-1).x - 2.0*ucat(k,j,i  ).x + 3.0*ucat(k,j,i+1).x) + ucat(k,j,i  ).x);
                flux.y =
                    um * (coef * (-ucat(k,j,i+2).y - 2.0*ucat(k,j,i+1).y + 3.0*ucat(k,j,i  ).y) + ucat(k,j,i+1).y) +
                    up * (coef * (-ucat(k,j,i-1).y - 2.0*ucat(k,j,i  ).y + 3.0*ucat(k,j,i+1).y) + ucat(k,j,i  ).y);
                flux.z =
                    um * (coef * (-ucat(k,j,i+2).z - 2.0*ucat(k,j,i+1).z + 3.0*ucat(k,j,i  ).z) + ucat(k,j,i+1).z) +
                    up * (coef * (-ucat(k,j,i-1).z - 2.0*ucat(k,j,i  ).z + 3.0*ucat(k,j,i+1).z) + ucat(k,j,i  ).z);
            }
            // Left boundary or solid neighbor on left
            else if (i == 0 || nvert(k, j, i-1) > 0.1) {
                flux.x =
                    um * (coef * (-ucat(k,j,i+2).x - 2.0*ucat(k,j,i+1).x + 3.0*ucat(k,j,i  ).x) + ucat(k,j,i+1).x) +
                    up * (coef * (-ucat(k,j,i  ).x - 2.0*ucat(k,j,i  ).x + 3.0*ucat(k,j,i+1).x) + ucat(k,j,i  ).x);
                flux.y =
                    um * (coef * (-ucat(k,j,i+2).y - 2.0*ucat(k,j,i+1).y + 3.0*ucat(k,j,i  ).y) + ucat(k,j,i+1).y) +
                    up * (coef * (-ucat(k,j,i  ).y - 2.0*ucat(k,j,i  ).y + 3.0*ucat(k,j,i+1).y) + ucat(k,j,i  ).y);
                flux.z =
                    um * (coef * (-ucat(k,j,i+2).z - 2.0*ucat(k,j,i+1).z + 3.0*ucat(k,j,i  ).z) + ucat(k,j,i+1).z) +
                    up * (coef * (-ucat(k,j,i  ).z - 2.0*ucat(k,j,i  ).z + 3.0*ucat(k,j,i+1).z) + ucat(k,j,i  ).z);
            }
            // Right boundary or solid neighbor on right
            else if (i == mx - 2 || nvert(k, j, i+1) > 0.1) {
                flux.x =
                    um * (coef * (-ucat(k,j,i+1).x - 2.0*ucat(k,j,i+1).x + 3.0*ucat(k,j,i  ).x) + ucat(k,j,i+1).x) +
                    up * (coef * (-ucat(k,j,i-1).x - 2.0*ucat(k,j,i  ).x + 3.0*ucat(k,j,i+1).x) + ucat(k,j,i  ).x);
                flux.y =
                    um * (coef * (-ucat(k,j,i+1).y - 2.0*ucat(k,j,i+1).y + 3.0*ucat(k,j,i  ).y) + ucat(k,j,i+1).y) +
                    up * (coef * (-ucat(k,j,i-1).y - 2.0*ucat(k,j,i  ).y + 3.0*ucat(k,j,i+1).y) + ucat(k,j,i  ).y);
                flux.z =
                    um * (coef * (-ucat(k,j,i+1).z - 2.0*ucat(k,j,i+1).z + 3.0*ucat(k,j,i  ).z) + ucat(k,j,i+1).z) +
                    up * (coef * (-ucat(k,j,i-1).z - 2.0*ucat(k,j,i  ).z + 3.0*ucat(k,j,i+1).z) + ucat(k,j,i  ).z);
            }

            fp1(k, j, i) = flux;
        }
    );
}

inline void ConvectionKernel::computeFluxJ(
    const VectorView& ucont,
    const VectorView& ucat,
    const ScalarView& nvert,
    VectorView& fp2,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const int my = domain.my;
    const double coef = QUICK_COEF;

    Kokkos::parallel_for("Convection_FluxJ",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys-1, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            double ucon = ucont(k, j, i).y * 0.5;
            double up = ucon + fabs(ucon);
            double um = ucon - fabs(ucon);

            Cmpnts3 flux;

            // Interior nodes
            if (j > 0 && j < my - 2 &&
                nvert(k, j+1, i) < 0.1 &&
                nvert(k, j-1, i) < 0.1) {
                flux.x =
                    um * (coef * (-ucat(k,j+2,i).x - 2.0*ucat(k,j+1,i).x + 3.0*ucat(k,j  ,i).x) + ucat(k,j+1,i).x) +
                    up * (coef * (-ucat(k,j-1,i).x - 2.0*ucat(k,j  ,i).x + 3.0*ucat(k,j+1,i).x) + ucat(k,j  ,i).x);
                flux.y =
                    um * (coef * (-ucat(k,j+2,i).y - 2.0*ucat(k,j+1,i).y + 3.0*ucat(k,j  ,i).y) + ucat(k,j+1,i).y) +
                    up * (coef * (-ucat(k,j-1,i).y - 2.0*ucat(k,j  ,i).y + 3.0*ucat(k,j+1,i).y) + ucat(k,j  ,i).y);
                flux.z =
                    um * (coef * (-ucat(k,j+2,i).z - 2.0*ucat(k,j+1,i).z + 3.0*ucat(k,j  ,i).z) + ucat(k,j+1,i).z) +
                    up * (coef * (-ucat(k,j-1,i).z - 2.0*ucat(k,j  ,i).z + 3.0*ucat(k,j+1,i).z) + ucat(k,j  ,i).z);
            }
            // Bottom boundary or solid neighbor below
            else if (j == 0 || nvert(k, j-1, i) > 0.1) {
                flux.x =
                    um * (coef * (-ucat(k,j+2,i).x - 2.0*ucat(k,j+1,i).x + 3.0*ucat(k,j  ,i).x) + ucat(k,j+1,i).x) +
                    up * (coef * (-ucat(k,j  ,i).x - 2.0*ucat(k,j  ,i).x + 3.0*ucat(k,j+1,i).x) + ucat(k,j  ,i).x);
                flux.y =
                    um * (coef * (-ucat(k,j+2,i).y - 2.0*ucat(k,j+1,i).y + 3.0*ucat(k,j  ,i).y) + ucat(k,j+1,i).y) +
                    up * (coef * (-ucat(k,j  ,i).y - 2.0*ucat(k,j  ,i).y + 3.0*ucat(k,j+1,i).y) + ucat(k,j  ,i).y);
                flux.z =
                    um * (coef * (-ucat(k,j+2,i).z - 2.0*ucat(k,j+1,i).z + 3.0*ucat(k,j  ,i).z) + ucat(k,j+1,i).z) +
                    up * (coef * (-ucat(k,j  ,i).z - 2.0*ucat(k,j  ,i).z + 3.0*ucat(k,j+1,i).z) + ucat(k,j  ,i).z);
            }
            // Top boundary or solid neighbor above
            else if (j == my - 2 || nvert(k, j+1, i) > 0.1) {
                flux.x =
                    um * (coef * (-ucat(k,j+1,i).x - 2.0*ucat(k,j+1,i).x + 3.0*ucat(k,j  ,i).x) + ucat(k,j+1,i).x) +
                    up * (coef * (-ucat(k,j-1,i).x - 2.0*ucat(k,j  ,i).x + 3.0*ucat(k,j+1,i).x) + ucat(k,j  ,i).x);
                flux.y =
                    um * (coef * (-ucat(k,j+1,i).y - 2.0*ucat(k,j+1,i).y + 3.0*ucat(k,j  ,i).y) + ucat(k,j+1,i).y) +
                    up * (coef * (-ucat(k,j-1,i).y - 2.0*ucat(k,j  ,i).y + 3.0*ucat(k,j+1,i).y) + ucat(k,j  ,i).y);
                flux.z =
                    um * (coef * (-ucat(k,j+1,i).z - 2.0*ucat(k,j+1,i).z + 3.0*ucat(k,j  ,i).z) + ucat(k,j+1,i).z) +
                    up * (coef * (-ucat(k,j-1,i).z - 2.0*ucat(k,j  ,i).z + 3.0*ucat(k,j+1,i).z) + ucat(k,j  ,i).z);
            }

            fp2(k, j, i) = flux;
        }
    );
}

inline void ConvectionKernel::computeFluxK(
    const VectorView& ucont,
    const VectorView& ucat,
    const ScalarView& nvert,
    VectorView& fp3,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const int mz = domain.mz;
    const double coef = QUICK_COEF;

    Kokkos::parallel_for("Convection_FluxK",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs-1, lys, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            double ucon = ucont(k, j, i).z * 0.5;
            double up = ucon + fabs(ucon);
            double um = ucon - fabs(ucon);

            Cmpnts3 flux;

            // Interior nodes
            if (k > 0 && k < mz - 2 &&
                nvert(k+1, j, i) < 0.1 &&
                nvert(k-1, j, i) < 0.1) {
                flux.x =
                    um * (coef * (-ucat(k+2,j,i).x - 2.0*ucat(k+1,j,i).x + 3.0*ucat(k  ,j,i).x) + ucat(k+1,j,i).x) +
                    up * (coef * (-ucat(k-1,j,i).x - 2.0*ucat(k  ,j,i).x + 3.0*ucat(k+1,j,i).x) + ucat(k  ,j,i).x);
                flux.y =
                    um * (coef * (-ucat(k+2,j,i).y - 2.0*ucat(k+1,j,i).y + 3.0*ucat(k  ,j,i).y) + ucat(k+1,j,i).y) +
                    up * (coef * (-ucat(k-1,j,i).y - 2.0*ucat(k  ,j,i).y + 3.0*ucat(k+1,j,i).y) + ucat(k  ,j,i).y);
                flux.z =
                    um * (coef * (-ucat(k+2,j,i).z - 2.0*ucat(k+1,j,i).z + 3.0*ucat(k  ,j,i).z) + ucat(k+1,j,i).z) +
                    up * (coef * (-ucat(k-1,j,i).z - 2.0*ucat(k  ,j,i).z + 3.0*ucat(k+1,j,i).z) + ucat(k  ,j,i).z);
            }
            // Back boundary or solid neighbor behind
            else if (k < mz - 2 && (k == 0 || nvert(k-1, j, i) > 0.1)) {
                flux.x =
                    um * (coef * (-ucat(k+2,j,i).x - 2.0*ucat(k+1,j,i).x + 3.0*ucat(k  ,j,i).x) + ucat(k+1,j,i).x) +
                    up * (coef * (-ucat(k  ,j,i).x - 2.0*ucat(k  ,j,i).x + 3.0*ucat(k+1,j,i).x) + ucat(k  ,j,i).x);
                flux.y =
                    um * (coef * (-ucat(k+2,j,i).y - 2.0*ucat(k+1,j,i).y + 3.0*ucat(k  ,j,i).y) + ucat(k+1,j,i).y) +
                    up * (coef * (-ucat(k  ,j,i).y - 2.0*ucat(k  ,j,i).y + 3.0*ucat(k+1,j,i).y) + ucat(k  ,j,i).y);
                flux.z =
                    um * (coef * (-ucat(k+2,j,i).z - 2.0*ucat(k+1,j,i).z + 3.0*ucat(k  ,j,i).z) + ucat(k+1,j,i).z) +
                    up * (coef * (-ucat(k  ,j,i).z - 2.0*ucat(k  ,j,i).z + 3.0*ucat(k+1,j,i).z) + ucat(k  ,j,i).z);
            }
            // Front boundary or solid neighbor in front
            else if (k > 0 && (k == mz - 2 || nvert(k+1, j, i) > 0.1)) {
                flux.x =
                    um * (coef * (-ucat(k+1,j,i).x - 2.0*ucat(k+1,j,i).x + 3.0*ucat(k  ,j,i).x) + ucat(k+1,j,i).x) +
                    up * (coef * (-ucat(k-1,j,i).x - 2.0*ucat(k  ,j,i).x + 3.0*ucat(k+1,j,i).x) + ucat(k  ,j,i).x);
                flux.y =
                    um * (coef * (-ucat(k+1,j,i).y - 2.0*ucat(k+1,j,i).y + 3.0*ucat(k  ,j,i).y) + ucat(k+1,j,i).y) +
                    up * (coef * (-ucat(k-1,j,i).y - 2.0*ucat(k  ,j,i).y + 3.0*ucat(k+1,j,i).y) + ucat(k  ,j,i).y);
                flux.z =
                    um * (coef * (-ucat(k+1,j,i).z - 2.0*ucat(k+1,j,i).z + 3.0*ucat(k  ,j,i).z) + ucat(k+1,j,i).z) +
                    up * (coef * (-ucat(k-1,j,i).z - 2.0*ucat(k  ,j,i).z + 3.0*ucat(k+1,j,i).z) + ucat(k  ,j,i).z);
            }

            fp3(k, j, i) = flux;
        }
    );
}

inline void ConvectionKernel::combineFluxes(
    const VectorView& fp1,
    const VectorView& fp2,
    const VectorView& fp3,
    VectorView& conv,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;

    Kokkos::parallel_for("Convection_Combine",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Divergence of fluxes: d(flux)/dx = flux(i) - flux(i-1)
            conv(k, j, i).x =
                fp1(k, j, i).x - fp1(k, j, i-1).x +
                fp2(k, j, i).x - fp2(k, j-1, i).x +
                fp3(k, j, i).x - fp3(k-1, j, i).x;

            conv(k, j, i).y =
                fp1(k, j, i).y - fp1(k, j, i-1).y +
                fp2(k, j, i).y - fp2(k, j-1, i).y +
                fp3(k, j, i).y - fp3(k-1, j, i).y;

            conv(k, j, i).z =
                fp1(k, j, i).z - fp1(k, j, i-1).z +
                fp2(k, j, i).z - fp2(k, j-1, i).z +
                fp3(k, j, i).z - fp3(k-1, j, i).z;
        }
    );
}

} // namespace kernels
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_CONVECTION_KERNEL_HPP
