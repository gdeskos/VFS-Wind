/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Pressure Gradient Kernel for VFS-Wind                      *
 * Computes pressure gradient on GPU via Kokkos                   *
 ******************************************************************/

#ifndef VFSWIND_PRESSURE_GRADIENT_KERNEL_HPP
#define VFSWIND_PRESSURE_GRADIENT_KERNEL_HPP

#ifdef ENABLE_GPU

#include "petsc_kokkos.hpp"
#include <Kokkos_Core.hpp>

namespace vfswind {
namespace gpu {
namespace kernels {

/**
 * @brief GPU Pressure Gradient kernel
 *
 * This kernel computes the pressure gradient term:
 *   dp = -grad(p)
 *
 * In curvilinear coordinates, the gradient uses the metric tensors
 * to transform from computational to physical space.
 */
class PressureGradientKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    /**
     * @brief Execute the pressure gradient kernel
     *
     * @param p Pressure field (input)
     * @param nvert Solid cell marker (input)
     * @param icsi, ieta, izet I-face metric tensors
     * @param jcsi, jeta, jzet J-face metric tensors
     * @param kcsi, keta, kzet K-face metric tensors
     * @param iaj, jaj, kaj Face Jacobians
     * @param dp Pressure gradient output (negative gradient)
     * @param domain Domain information
     */
    static void execute(
        const ScalarView& p,
        const ScalarView& nvert,
        const VectorView& icsi, const VectorView& ieta, const VectorView& izet,
        const VectorView& jcsi, const VectorView& jeta, const VectorView& jzet,
        const VectorView& kcsi, const VectorView& keta, const VectorView& kzet,
        const ScalarView& iaj, const ScalarView& jaj, const ScalarView& kaj,
        VectorView& dp,
        const KernelDomainInfo& domain
    );
};

// ============================================================================
// Implementation
// ============================================================================

inline void PressureGradientKernel::execute(
    const ScalarView& p,
    const ScalarView& nvert,
    const VectorView& icsi, const VectorView& ieta, const VectorView& izet,
    const VectorView& jcsi, const VectorView& jeta, const VectorView& jzet,
    const VectorView& kcsi, const VectorView& keta, const VectorView& kzet,
    const ScalarView& iaj, const ScalarView& jaj, const ScalarView& kaj,
    VectorView& dp,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const int mx = domain.mx;
    const int my = domain.my;
    const int mz = domain.mz;

    // Initialize output to zero
    Kokkos::deep_copy(dp, Cmpnts3());

    Kokkos::parallel_for("PressureGradient",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // I-face metric products
            double g11_i = icsi(k,j,i).x*icsi(k,j,i).x + icsi(k,j,i).y*icsi(k,j,i).y + icsi(k,j,i).z*icsi(k,j,i).z;
            double g12_i = ieta(k,j,i).x*icsi(k,j,i).x + ieta(k,j,i).y*icsi(k,j,i).y + ieta(k,j,i).z*icsi(k,j,i).z;
            double g13_i = izet(k,j,i).x*icsi(k,j,i).x + izet(k,j,i).y*icsi(k,j,i).y + izet(k,j,i).z*icsi(k,j,i).z;

            // J-face metric products
            double g21_j = jcsi(k,j,i).x*jeta(k,j,i).x + jcsi(k,j,i).y*jeta(k,j,i).y + jcsi(k,j,i).z*jeta(k,j,i).z;
            double g22_j = jeta(k,j,i).x*jeta(k,j,i).x + jeta(k,j,i).y*jeta(k,j,i).y + jeta(k,j,i).z*jeta(k,j,i).z;
            double g23_j = jzet(k,j,i).x*jeta(k,j,i).x + jzet(k,j,i).y*jeta(k,j,i).y + jzet(k,j,i).z*jeta(k,j,i).z;

            // K-face metric products
            double g31_k = kcsi(k,j,i).x*kzet(k,j,i).x + kcsi(k,j,i).y*kzet(k,j,i).y + kcsi(k,j,i).z*kzet(k,j,i).z;
            double g32_k = keta(k,j,i).x*kzet(k,j,i).x + keta(k,j,i).y*kzet(k,j,i).y + keta(k,j,i).z*kzet(k,j,i).z;
            double g33_k = kzet(k,j,i).x*kzet(k,j,i).x + kzet(k,j,i).y*kzet(k,j,i).y + kzet(k,j,i).z*kzet(k,j,i).z;

            // Pressure derivatives
            double dpdc = p(k, j, i+1) - p(k, j, i);

            // Eta-direction derivative with boundary handling
            double dpde;
            if (static_cast<int>(nvert(k,j+1,i) + 0.5) == 1 ||
                static_cast<int>(nvert(k,j+1,i+1) + 0.5) == 1 ||
                j == my - 2) {
                dpde = (p(k,j,i) - p(k,j-1,i) + p(k,j,i+1) - p(k,j-1,i+1)) * 0.5;
            }
            else if (static_cast<int>(nvert(k,j-1,i) + 0.5) == 1 ||
                     static_cast<int>(nvert(k,j-1,i+1) + 0.5) == 1 ||
                     j == 1) {
                dpde = (p(k,j+1,i) - p(k,j,i) + p(k,j+1,i+1) - p(k,j,i+1)) * 0.5;
            }
            else {
                dpde = (p(k,j+1,i) - p(k,j-1,i) + p(k,j+1,i+1) - p(k,j-1,i+1)) * 0.25;
            }

            // Zeta-direction derivative with boundary handling
            double dpdz;
            if (static_cast<int>(nvert(k+1,j,i) + 0.5) == 1 ||
                static_cast<int>(nvert(k+1,j,i+1) + 0.5) == 1 ||
                k == mz - 2) {
                dpdz = (p(k,j,i) - p(k-1,j,i) + p(k,j,i+1) - p(k-1,j,i+1)) * 0.5;
            }
            else if (static_cast<int>(nvert(k-1,j,i) + 0.5) == 1 ||
                     static_cast<int>(nvert(k-1,j,i+1) + 0.5) == 1 ||
                     k == 1) {
                dpdz = (p(k+1,j,i) - p(k,j,i) + p(k+1,j,i+1) - p(k,j,i+1)) * 0.5;
            }
            else {
                dpdz = (p(k+1,j,i) - p(k-1,j,i) + p(k+1,j,i+1) - p(k-1,j,i+1)) * 0.25;
            }

            // Compute pressure gradient contributions (negative gradient)
            // I-direction contribution
            double dpx_i = -(g11_i * dpdc + g12_i * dpde + g13_i * dpdz) * iaj(k,j,i);

            // J-direction contribution
            double dpdc_j = (p(k,j+1,i) - p(k,j-1,i) + p(k,j+1,i+1) - p(k,j-1,i+1)) * 0.25;
            double dpde_j = p(k,j+1,i) - p(k,j,i);
            double dpdz_j = (p(k+1,j,i) - p(k-1,j,i) + p(k+1,j+1,i) - p(k-1,j+1,i)) * 0.25;
            double dpy_j = -(g21_j * dpdc_j + g22_j * dpde_j + g23_j * dpdz_j) * jaj(k,j,i);

            // K-direction contribution
            double dpdc_k = (p(k+1,j,i) - p(k-1,j,i) + p(k+1,j,i+1) - p(k-1,j,i+1)) * 0.25;
            double dpde_k = (p(k+1,j+1,i) - p(k-1,j+1,i) + p(k+1,j,i) - p(k-1,j,i)) * 0.25;
            double dpdz_k = p(k+1,j,i) - p(k,j,i);
            double dpz_k = -(g31_k * dpdc_k + g32_k * dpde_k + g33_k * dpdz_k) * kaj(k,j,i);

            // Store result
            dp(k, j, i).x = dpx_i;
            dp(k, j, i).y = dpy_j;
            dp(k, j, i).z = dpz_k;
        }
    );

    Kokkos::fence();
}

} // namespace kernels
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_PRESSURE_GRADIENT_KERNEL_HPP
