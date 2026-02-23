/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * VFS-Wind GPU Kernels - Unified Header                          *
 ******************************************************************/

#ifndef VFSWIND_KERNELS_HPP
#define VFSWIND_KERNELS_HPP

#ifdef ENABLE_GPU

// PETSc-Kokkos integration utilities
#include "petsc_kokkos.hpp"

// Individual kernels
#include "convection_kernel.hpp"
#include "viscous_kernel.hpp"
#include "pressure_gradient_kernel.hpp"

namespace vfswind {
namespace gpu {

/**
 * @brief Execute all RHS kernels (Convection + Viscous)
 *
 * This is a convenience function that runs both the convection and
 * viscous kernels and combines them into a single RHS vector.
 *
 * @param ucont Contravariant velocity (input)
 * @param ucat Cartesian velocity (input)
 * @param nvert Solid marker (input)
 * @param icsi, ieta, izet I-face metrics (input)
 * @param jcsi, jeta, jzet J-face metrics (input)
 * @param kcsi, keta, kzet K-face metrics (input)
 * @param iaj, jaj, kaj Face Jacobians (input)
 * @param nu_t Turbulent viscosity (input, optional)
 * @param rhs Output: rhs = -conv + visc
 * @param domain Domain info
 */
inline void computeRHS(
    const kernels::ConvectionKernel::VectorView& ucont,
    const kernels::ConvectionKernel::VectorView& ucat,
    const kernels::ConvectionKernel::ScalarView& nvert,
    const kernels::ViscousKernel::VectorView& icsi,
    const kernels::ViscousKernel::VectorView& ieta,
    const kernels::ViscousKernel::VectorView& izet,
    const kernels::ViscousKernel::VectorView& jcsi,
    const kernels::ViscousKernel::VectorView& jeta,
    const kernels::ViscousKernel::VectorView& jzet,
    const kernels::ViscousKernel::VectorView& kcsi,
    const kernels::ViscousKernel::VectorView& keta,
    const kernels::ViscousKernel::VectorView& kzet,
    const kernels::ViscousKernel::ScalarView& iaj,
    const kernels::ViscousKernel::ScalarView& jaj,
    const kernels::ViscousKernel::ScalarView& kaj,
    const kernels::ViscousKernel::ScalarView* nu_t,
    kernels::ConvectionKernel::VectorView& rhs,
    const KernelDomainInfo& domain
) {
    using VectorView = kernels::ConvectionKernel::VectorView;

    // Allocate temporary arrays
    VectorView conv("conv", domain.mz, domain.my, domain.mx);
    VectorView visc("visc", domain.mz, domain.my, domain.mx);

    // Initialize
    Kokkos::deep_copy(conv, Cmpnts3());
    Kokkos::deep_copy(visc, Cmpnts3());

    // Compute convection term
    kernels::ConvectionKernel::execute(ucont, ucat, nvert, conv, domain);

    // Compute viscous term
    kernels::ViscousKernel::execute(
        ucat, nvert,
        icsi, ieta, izet,
        jcsi, jeta, jzet,
        kcsi, keta, kzet,
        iaj, jaj, kaj,
        nu_t, visc, domain
    );

    // Combine: rhs = -conv + visc
    const int lxs = domain.lxs, lxe = domain.lxe;
    const int lys = domain.lys, lye = domain.lye;
    const int lzs = domain.lzs, lze = domain.lze;

    Kokkos::parallel_for("CombineRHS",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            rhs(k, j, i).x = -conv(k, j, i).x + visc(k, j, i).x;
            rhs(k, j, i).y = -conv(k, j, i).y + visc(k, j, i).y;
            rhs(k, j, i).z = -conv(k, j, i).z + visc(k, j, i).z;
        }
    );

    Kokkos::fence();
}

} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_KERNELS_HPP
