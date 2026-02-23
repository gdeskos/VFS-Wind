/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Viscous Kernel for VFS-Wind                                *
 * Computes viscous diffusion terms on GPU via Kokkos             *
 ******************************************************************/

#ifndef VFSWIND_VISCOUS_KERNEL_HPP
#define VFSWIND_VISCOUS_KERNEL_HPP

#ifdef ENABLE_GPU

#include "petsc_kokkos.hpp"
#include <Kokkos_Core.hpp>

namespace vfswind {
namespace gpu {
namespace kernels {

/**
 * @brief GPU Viscous kernel
 *
 * This kernel computes the viscous terms for the Navier-Stokes equations:
 *   visc = nu * Laplacian(u)
 *
 * In curvilinear coordinates, this requires computing velocity gradients
 * using the metric tensors (csi, eta, zet) and Jacobian (aj).
 *
 * The computation is done in three phases:
 * 1. Compute i-direction viscous flux
 * 2. Compute j-direction viscous flux
 * 3. Compute k-direction viscous flux
 * 4. Combine to get final viscous term
 */
class ViscousKernel {
public:
    using ScalarView = ScalarView3D<DeviceMemorySpace>;
    using VectorView = VectorView3D<DeviceMemorySpace>;

    /**
     * @brief Execute the viscous kernel
     *
     * @param ucat Cartesian velocity components (input)
     * @param nvert Solid cell marker (input)
     * @param icsi, ieta, izet I-face metric tensors
     * @param jcsi, jeta, jzet J-face metric tensors
     * @param kcsi, keta, kzet K-face metric tensors
     * @param iaj, jaj, kaj Face Jacobians
     * @param nu_t Turbulent viscosity (optional, can be nullptr)
     * @param visc Viscous term output
     * @param domain Domain information
     */
    static void execute(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& icsi, const VectorView& ieta, const VectorView& izet,
        const VectorView& jcsi, const VectorView& jeta, const VectorView& jzet,
        const VectorView& kcsi, const VectorView& keta, const VectorView& kzet,
        const ScalarView& iaj, const ScalarView& jaj, const ScalarView& kaj,
        const ScalarView* nu_t,  // Can be nullptr if no turbulence model
        VectorView& visc,
        const KernelDomainInfo& domain
    );

private:
    static constexpr double SOLID_THRESHOLD = 0.5;

    /**
     * @brief Compute i-direction viscous flux
     */
    static void computeFluxI(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& icsi, const VectorView& ieta, const VectorView& izet,
        const ScalarView& iaj,
        const ScalarView* nu_t,
        VectorView& fp1,
        const KernelDomainInfo& domain
    );

    /**
     * @brief Compute j-direction viscous flux
     */
    static void computeFluxJ(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& jcsi, const VectorView& jeta, const VectorView& jzet,
        const ScalarView& jaj,
        const ScalarView* nu_t,
        VectorView& fp2,
        const KernelDomainInfo& domain
    );

    /**
     * @brief Compute k-direction viscous flux
     */
    static void computeFluxK(
        const VectorView& ucat,
        const ScalarView& nvert,
        const VectorView& kcsi, const VectorView& keta, const VectorView& kzet,
        const ScalarView& kaj,
        const ScalarView* nu_t,
        VectorView& fp3,
        const KernelDomainInfo& domain
    );

    /**
     * @brief Combine directional fluxes
     */
    static void combineFluxes(
        const VectorView& fp1,
        const VectorView& fp2,
        const VectorView& fp3,
        VectorView& visc,
        const KernelDomainInfo& domain
    );
};

// ============================================================================
// Implementation
// ============================================================================

inline void ViscousKernel::execute(
    const VectorView& ucat,
    const ScalarView& nvert,
    const VectorView& icsi, const VectorView& ieta, const VectorView& izet,
    const VectorView& jcsi, const VectorView& jeta, const VectorView& jzet,
    const VectorView& kcsi, const VectorView& keta, const VectorView& kzet,
    const ScalarView& iaj, const ScalarView& jaj, const ScalarView& kaj,
    const ScalarView* nu_t,
    VectorView& visc,
    const KernelDomainInfo& domain
) {
    // Allocate temporary flux arrays
    VectorView fp1("visc_fp1", domain.mz, domain.my, domain.mx);
    VectorView fp2("visc_fp2", domain.mz, domain.my, domain.mx);
    VectorView fp3("visc_fp3", domain.mz, domain.my, domain.mx);

    // Initialize to zero
    Kokkos::deep_copy(fp1, Cmpnts3());
    Kokkos::deep_copy(fp2, Cmpnts3());
    Kokkos::deep_copy(fp3, Cmpnts3());

    // Compute directional fluxes
    computeFluxI(ucat, nvert, icsi, ieta, izet, iaj, nu_t, fp1, domain);
    computeFluxJ(ucat, nvert, jcsi, jeta, jzet, jaj, nu_t, fp2, domain);
    computeFluxK(ucat, nvert, kcsi, keta, kzet, kaj, nu_t, fp3, domain);

    // Combine fluxes
    combineFluxes(fp1, fp2, fp3, visc, domain);

    Kokkos::fence();
}

inline void ViscousKernel::computeFluxI(
    const VectorView& ucat,
    const ScalarView& nvert,
    const VectorView& icsi, const VectorView& ieta, const VectorView& izet,
    const ScalarView& iaj,
    const ScalarView* nu_t,
    VectorView& fp1,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const double nu = 1.0 / domain.ren;
    const double solid = SOLID_THRESHOLD;
    const bool has_nu_t = (nu_t != nullptr && (domain.les || domain.rans));

    // Capture nu_t for lambda
    ScalarView nu_t_view;
    if (has_nu_t) {
        nu_t_view = *nu_t;
    }

    Kokkos::parallel_for("Viscous_FluxI",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs-1}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Compute velocity derivatives
            double dudc = ucat(k, j, i+1).x - ucat(k, j, i).x;
            double dvdc = ucat(k, j, i+1).y - ucat(k, j, i).y;
            double dwdc = ucat(k, j, i+1).z - ucat(k, j, i).z;

            // Eta-direction derivatives (j-direction)
            double dude, dvde, dwde;
            if (nvert(k, j+1, i) > solid || nvert(k, j+1, i+1) > solid) {
                dude = (ucat(k,j  ,i+1).x + ucat(k,j  ,i).x - ucat(k,j-1,i+1).x - ucat(k,j-1,i).x) * 0.5;
                dvde = (ucat(k,j  ,i+1).y + ucat(k,j  ,i).y - ucat(k,j-1,i+1).y - ucat(k,j-1,i).y) * 0.5;
                dwde = (ucat(k,j  ,i+1).z + ucat(k,j  ,i).z - ucat(k,j-1,i+1).z - ucat(k,j-1,i).z) * 0.5;
            }
            else if (nvert(k, j-1, i) > solid || nvert(k, j-1, i+1) > solid) {
                dude = (ucat(k,j+1,i+1).x + ucat(k,j+1,i).x - ucat(k,j  ,i+1).x - ucat(k,j  ,i).x) * 0.5;
                dvde = (ucat(k,j+1,i+1).y + ucat(k,j+1,i).y - ucat(k,j  ,i+1).y - ucat(k,j  ,i).y) * 0.5;
                dwde = (ucat(k,j+1,i+1).z + ucat(k,j+1,i).z - ucat(k,j  ,i+1).z - ucat(k,j  ,i).z) * 0.5;
            }
            else {
                dude = (ucat(k,j+1,i+1).x + ucat(k,j+1,i).x - ucat(k,j-1,i+1).x - ucat(k,j-1,i).x) * 0.25;
                dvde = (ucat(k,j+1,i+1).y + ucat(k,j+1,i).y - ucat(k,j-1,i+1).y - ucat(k,j-1,i).y) * 0.25;
                dwde = (ucat(k,j+1,i+1).z + ucat(k,j+1,i).z - ucat(k,j-1,i+1).z - ucat(k,j-1,i).z) * 0.25;
            }

            // Zeta-direction derivatives (k-direction)
            double dudz, dvdz, dwdz;
            if (nvert(k+1, j, i) > solid || nvert(k+1, j, i+1) > solid) {
                dudz = (ucat(k  ,j,i+1).x + ucat(k  ,j,i).x - ucat(k-1,j,i+1).x - ucat(k-1,j,i).x) * 0.5;
                dvdz = (ucat(k  ,j,i+1).y + ucat(k  ,j,i).y - ucat(k-1,j,i+1).y - ucat(k-1,j,i).y) * 0.5;
                dwdz = (ucat(k  ,j,i+1).z + ucat(k  ,j,i).z - ucat(k-1,j,i+1).z - ucat(k-1,j,i).z) * 0.5;
            }
            else if (nvert(k-1, j, i) > solid || nvert(k-1, j, i+1) > solid) {
                dudz = (ucat(k+1,j,i+1).x + ucat(k+1,j,i).x - ucat(k  ,j,i+1).x - ucat(k  ,j,i).x) * 0.5;
                dvdz = (ucat(k+1,j,i+1).y + ucat(k+1,j,i).y - ucat(k  ,j,i+1).y - ucat(k  ,j,i).y) * 0.5;
                dwdz = (ucat(k+1,j,i+1).z + ucat(k+1,j,i).z - ucat(k  ,j,i+1).z - ucat(k  ,j,i).z) * 0.5;
            }
            else {
                dudz = (ucat(k+1,j,i+1).x + ucat(k+1,j,i).x - ucat(k-1,j,i+1).x - ucat(k-1,j,i).x) * 0.25;
                dvdz = (ucat(k+1,j,i+1).y + ucat(k+1,j,i).y - ucat(k-1,j,i+1).y - ucat(k-1,j,i).y) * 0.25;
                dwdz = (ucat(k+1,j,i+1).z + ucat(k+1,j,i).z - ucat(k-1,j,i+1).z - ucat(k-1,j,i).z) * 0.25;
            }

            // Metric tensor components at i-face
            double csi0 = icsi(k,j,i).x, csi1 = icsi(k,j,i).y, csi2 = icsi(k,j,i).z;
            double eta0 = ieta(k,j,i).x, eta1 = ieta(k,j,i).y, eta2 = ieta(k,j,i).z;
            double zet0 = izet(k,j,i).x, zet1 = izet(k,j,i).y, zet2 = izet(k,j,i).z;

            // Metric products
            double g11 = csi0*csi0 + csi1*csi1 + csi2*csi2;
            double g21 = eta0*csi0 + eta1*csi1 + eta2*csi2;
            double g31 = zet0*csi0 + zet1*csi1 + zet2*csi2;

            // Velocity gradient in physical coords
            double r11 = dudc*csi0 + dude*eta0 + dudz*zet0;
            double r21 = dvdc*csi0 + dvde*eta0 + dvdz*zet0;
            double r31 = dwdc*csi0 + dwde*eta0 + dwdz*zet0;

            double r12 = dudc*csi1 + dude*eta1 + dudz*zet1;
            double r22 = dvdc*csi1 + dvde*eta1 + dvdz*zet1;
            double r32 = dwdc*csi1 + dwde*eta1 + dwdz*zet1;

            double r13 = dudc*csi2 + dude*eta2 + dudz*zet2;
            double r23 = dvdc*csi2 + dvde*eta2 + dvdz*zet2;
            double r33 = dwdc*csi2 + dwde*eta2 + dwdz*zet2;

            // Jacobian at face
            double ajc = iaj(k, j, i);

            // Effective viscosity
            double nu_eff = nu;
            if (has_nu_t) {
                nu_eff += 0.5 * (nu_t_view(k,j,i) + nu_t_view(k,j,i+1));
            }

            // Viscous flux
            fp1(k,j,i).x = (g11*dudc + g21*dude + g31*dudz + r11*csi0 + r21*csi1 + r31*csi2) * ajc * nu_eff;
            fp1(k,j,i).y = (g11*dvdc + g21*dvde + g31*dvdz + r12*csi0 + r22*csi1 + r32*csi2) * ajc * nu_eff;
            fp1(k,j,i).z = (g11*dwdc + g21*dwde + g31*dwdz + r13*csi0 + r23*csi1 + r33*csi2) * ajc * nu_eff;
        }
    );
}

inline void ViscousKernel::computeFluxJ(
    const VectorView& ucat,
    const ScalarView& nvert,
    const VectorView& jcsi, const VectorView& jeta, const VectorView& jzet,
    const ScalarView& jaj,
    const ScalarView* nu_t,
    VectorView& fp2,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const double nu = 1.0 / domain.ren;
    const double solid = SOLID_THRESHOLD;
    const bool has_nu_t = (nu_t != nullptr && (domain.les || domain.rans));

    ScalarView nu_t_view;
    if (has_nu_t) {
        nu_t_view = *nu_t;
    }

    Kokkos::parallel_for("Viscous_FluxJ",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys-1, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Xi-direction derivatives
            double dudc, dvdc, dwdc;
            if (nvert(k, j, i+1) > solid || nvert(k, j+1, i+1) > solid) {
                dudc = (ucat(k,j+1,i  ).x + ucat(k,j,i  ).x - ucat(k,j+1,i-1).x - ucat(k,j,i-1).x) * 0.5;
                dvdc = (ucat(k,j+1,i  ).y + ucat(k,j,i  ).y - ucat(k,j+1,i-1).y - ucat(k,j,i-1).y) * 0.5;
                dwdc = (ucat(k,j+1,i  ).z + ucat(k,j,i  ).z - ucat(k,j+1,i-1).z - ucat(k,j,i-1).z) * 0.5;
            }
            else if (nvert(k, j, i-1) > solid || nvert(k, j+1, i-1) > solid) {
                dudc = (ucat(k,j+1,i+1).x + ucat(k,j,i+1).x - ucat(k,j+1,i  ).x - ucat(k,j,i  ).x) * 0.5;
                dvdc = (ucat(k,j+1,i+1).y + ucat(k,j,i+1).y - ucat(k,j+1,i  ).y - ucat(k,j,i  ).y) * 0.5;
                dwdc = (ucat(k,j+1,i+1).z + ucat(k,j,i+1).z - ucat(k,j+1,i  ).z - ucat(k,j,i  ).z) * 0.5;
            }
            else {
                dudc = (ucat(k,j+1,i+1).x + ucat(k,j,i+1).x - ucat(k,j+1,i-1).x - ucat(k,j,i-1).x) * 0.25;
                dvdc = (ucat(k,j+1,i+1).y + ucat(k,j,i+1).y - ucat(k,j+1,i-1).y - ucat(k,j,i-1).y) * 0.25;
                dwdc = (ucat(k,j+1,i+1).z + ucat(k,j,i+1).z - ucat(k,j+1,i-1).z - ucat(k,j,i-1).z) * 0.25;
            }

            // Eta-direction (direct)
            double dude = ucat(k, j+1, i).x - ucat(k, j, i).x;
            double dvde = ucat(k, j+1, i).y - ucat(k, j, i).y;
            double dwde = ucat(k, j+1, i).z - ucat(k, j, i).z;

            // Zeta-direction derivatives
            double dudz, dvdz, dwdz;
            if (nvert(k+1, j, i) > solid || nvert(k+1, j+1, i) > solid) {
                dudz = (ucat(k  ,j+1,i).x + ucat(k  ,j,i).x - ucat(k-1,j+1,i).x - ucat(k-1,j,i).x) * 0.5;
                dvdz = (ucat(k  ,j+1,i).y + ucat(k  ,j,i).y - ucat(k-1,j+1,i).y - ucat(k-1,j,i).y) * 0.5;
                dwdz = (ucat(k  ,j+1,i).z + ucat(k  ,j,i).z - ucat(k-1,j+1,i).z - ucat(k-1,j,i).z) * 0.5;
            }
            else if (nvert(k-1, j, i) > solid || nvert(k-1, j+1, i) > solid) {
                dudz = (ucat(k+1,j+1,i).x + ucat(k+1,j,i).x - ucat(k  ,j+1,i).x - ucat(k  ,j,i).x) * 0.5;
                dvdz = (ucat(k+1,j+1,i).y + ucat(k+1,j,i).y - ucat(k  ,j+1,i).y - ucat(k  ,j,i).y) * 0.5;
                dwdz = (ucat(k+1,j+1,i).z + ucat(k+1,j,i).z - ucat(k  ,j+1,i).z - ucat(k  ,j,i).z) * 0.5;
            }
            else {
                dudz = (ucat(k+1,j+1,i).x + ucat(k+1,j,i).x - ucat(k-1,j+1,i).x - ucat(k-1,j,i).x) * 0.25;
                dvdz = (ucat(k+1,j+1,i).y + ucat(k+1,j,i).y - ucat(k-1,j+1,i).y - ucat(k-1,j,i).y) * 0.25;
                dwdz = (ucat(k+1,j+1,i).z + ucat(k+1,j,i).z - ucat(k-1,j+1,i).z - ucat(k-1,j,i).z) * 0.25;
            }

            // Metric tensor at j-face
            double csi0 = jcsi(k,j,i).x, csi1 = jcsi(k,j,i).y, csi2 = jcsi(k,j,i).z;
            double eta0 = jeta(k,j,i).x, eta1 = jeta(k,j,i).y, eta2 = jeta(k,j,i).z;
            double zet0 = jzet(k,j,i).x, zet1 = jzet(k,j,i).y, zet2 = jzet(k,j,i).z;

            // Metric products for j-direction
            double g11 = csi0*eta0 + csi1*eta1 + csi2*eta2;
            double g21 = eta0*eta0 + eta1*eta1 + eta2*eta2;
            double g31 = zet0*eta0 + zet1*eta1 + zet2*eta2;

            // Velocity gradient
            double r11 = dudc*csi0 + dude*eta0 + dudz*zet0;
            double r21 = dvdc*csi0 + dvde*eta0 + dvdz*zet0;
            double r31 = dwdc*csi0 + dwde*eta0 + dwdz*zet0;

            double r12 = dudc*csi1 + dude*eta1 + dudz*zet1;
            double r22 = dvdc*csi1 + dvde*eta1 + dvdz*zet1;
            double r32 = dwdc*csi1 + dwde*eta1 + dwdz*zet1;

            double r13 = dudc*csi2 + dude*eta2 + dudz*zet2;
            double r23 = dvdc*csi2 + dvde*eta2 + dvdz*zet2;
            double r33 = dwdc*csi2 + dwde*eta2 + dwdz*zet2;

            double ajc = jaj(k, j, i);

            double nu_eff = nu;
            if (has_nu_t) {
                nu_eff += 0.5 * (nu_t_view(k,j,i) + nu_t_view(k,j+1,i));
            }

            fp2(k,j,i).x = (g11*dudc + g21*dude + g31*dudz + r11*eta0 + r21*eta1 + r31*eta2) * ajc * nu_eff;
            fp2(k,j,i).y = (g11*dvdc + g21*dvde + g31*dvdz + r12*eta0 + r22*eta1 + r32*eta2) * ajc * nu_eff;
            fp2(k,j,i).z = (g11*dwdc + g21*dwde + g31*dwdz + r13*eta0 + r23*eta1 + r33*eta2) * ajc * nu_eff;
        }
    );
}

inline void ViscousKernel::computeFluxK(
    const VectorView& ucat,
    const ScalarView& nvert,
    const VectorView& kcsi, const VectorView& keta, const VectorView& kzet,
    const ScalarView& kaj,
    const ScalarView* nu_t,
    VectorView& fp3,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;
    const double nu = 1.0 / domain.ren;
    const double solid = SOLID_THRESHOLD;
    const bool has_nu_t = (nu_t != nullptr && (domain.les || domain.rans));

    ScalarView nu_t_view;
    if (has_nu_t) {
        nu_t_view = *nu_t;
    }

    Kokkos::parallel_for("Viscous_FluxK",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs-1, lys, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Xi-direction derivatives
            double dudc, dvdc, dwdc;
            if (nvert(k, j, i+1) > solid || nvert(k+1, j, i+1) > solid) {
                dudc = (ucat(k+1,j,i  ).x + ucat(k,j,i  ).x - ucat(k+1,j,i-1).x - ucat(k,j,i-1).x) * 0.5;
                dvdc = (ucat(k+1,j,i  ).y + ucat(k,j,i  ).y - ucat(k+1,j,i-1).y - ucat(k,j,i-1).y) * 0.5;
                dwdc = (ucat(k+1,j,i  ).z + ucat(k,j,i  ).z - ucat(k+1,j,i-1).z - ucat(k,j,i-1).z) * 0.5;
            }
            else if (nvert(k, j, i-1) > solid || nvert(k+1, j, i-1) > solid) {
                dudc = (ucat(k+1,j,i+1).x + ucat(k,j,i+1).x - ucat(k+1,j,i  ).x - ucat(k,j,i  ).x) * 0.5;
                dvdc = (ucat(k+1,j,i+1).y + ucat(k,j,i+1).y - ucat(k+1,j,i  ).y - ucat(k,j,i  ).y) * 0.5;
                dwdc = (ucat(k+1,j,i+1).z + ucat(k,j,i+1).z - ucat(k+1,j,i  ).z - ucat(k,j,i  ).z) * 0.5;
            }
            else {
                dudc = (ucat(k+1,j,i+1).x + ucat(k,j,i+1).x - ucat(k+1,j,i-1).x - ucat(k,j,i-1).x) * 0.25;
                dvdc = (ucat(k+1,j,i+1).y + ucat(k,j,i+1).y - ucat(k+1,j,i-1).y - ucat(k,j,i-1).y) * 0.25;
                dwdc = (ucat(k+1,j,i+1).z + ucat(k,j,i+1).z - ucat(k+1,j,i-1).z - ucat(k,j,i-1).z) * 0.25;
            }

            // Eta-direction derivatives
            double dude, dvde, dwde;
            if (nvert(k, j+1, i) > solid || nvert(k+1, j+1, i) > solid) {
                dude = (ucat(k+1,j  ,i).x + ucat(k,j  ,i).x - ucat(k+1,j-1,i).x - ucat(k,j-1,i).x) * 0.5;
                dvde = (ucat(k+1,j  ,i).y + ucat(k,j  ,i).y - ucat(k+1,j-1,i).y - ucat(k,j-1,i).y) * 0.5;
                dwde = (ucat(k+1,j  ,i).z + ucat(k,j  ,i).z - ucat(k+1,j-1,i).z - ucat(k,j-1,i).z) * 0.5;
            }
            else if (nvert(k, j-1, i) > solid || nvert(k+1, j-1, i) > solid) {
                dude = (ucat(k+1,j+1,i).x + ucat(k,j+1,i).x - ucat(k+1,j  ,i).x - ucat(k,j  ,i).x) * 0.5;
                dvde = (ucat(k+1,j+1,i).y + ucat(k,j+1,i).y - ucat(k+1,j  ,i).y - ucat(k,j  ,i).y) * 0.5;
                dwde = (ucat(k+1,j+1,i).z + ucat(k,j+1,i).z - ucat(k+1,j  ,i).z - ucat(k,j  ,i).z) * 0.5;
            }
            else {
                dude = (ucat(k+1,j+1,i).x + ucat(k,j+1,i).x - ucat(k+1,j-1,i).x - ucat(k,j-1,i).x) * 0.25;
                dvde = (ucat(k+1,j+1,i).y + ucat(k,j+1,i).y - ucat(k+1,j-1,i).y - ucat(k,j-1,i).y) * 0.25;
                dwde = (ucat(k+1,j+1,i).z + ucat(k,j+1,i).z - ucat(k+1,j-1,i).z - ucat(k,j-1,i).z) * 0.25;
            }

            // Zeta-direction (direct)
            double dudz = ucat(k+1, j, i).x - ucat(k, j, i).x;
            double dvdz = ucat(k+1, j, i).y - ucat(k, j, i).y;
            double dwdz = ucat(k+1, j, i).z - ucat(k, j, i).z;

            // Metric tensor at k-face
            double csi0 = kcsi(k,j,i).x, csi1 = kcsi(k,j,i).y, csi2 = kcsi(k,j,i).z;
            double eta0 = keta(k,j,i).x, eta1 = keta(k,j,i).y, eta2 = keta(k,j,i).z;
            double zet0 = kzet(k,j,i).x, zet1 = kzet(k,j,i).y, zet2 = kzet(k,j,i).z;

            // Metric products for k-direction
            double g11 = csi0*zet0 + csi1*zet1 + csi2*zet2;
            double g21 = eta0*zet0 + eta1*zet1 + eta2*zet2;
            double g31 = zet0*zet0 + zet1*zet1 + zet2*zet2;

            // Velocity gradient
            double r11 = dudc*csi0 + dude*eta0 + dudz*zet0;
            double r21 = dvdc*csi0 + dvde*eta0 + dvdz*zet0;
            double r31 = dwdc*csi0 + dwde*eta0 + dwdz*zet0;

            double r12 = dudc*csi1 + dude*eta1 + dudz*zet1;
            double r22 = dvdc*csi1 + dvde*eta1 + dvdz*zet1;
            double r32 = dwdc*csi1 + dwde*eta1 + dwdz*zet1;

            double r13 = dudc*csi2 + dude*eta2 + dudz*zet2;
            double r23 = dvdc*csi2 + dvde*eta2 + dvdz*zet2;
            double r33 = dwdc*csi2 + dwde*eta2 + dwdz*zet2;

            double ajc = kaj(k, j, i);

            double nu_eff = nu;
            if (has_nu_t) {
                nu_eff += 0.5 * (nu_t_view(k,j,i) + nu_t_view(k+1,j,i));
            }

            fp3(k,j,i).x = (g11*dudc + g21*dude + g31*dudz + r11*zet0 + r21*zet1 + r31*zet2) * ajc * nu_eff;
            fp3(k,j,i).y = (g11*dvdc + g21*dvde + g31*dvdz + r12*zet0 + r22*zet1 + r32*zet2) * ajc * nu_eff;
            fp3(k,j,i).z = (g11*dwdc + g21*dwde + g31*dwdz + r13*zet0 + r23*zet1 + r33*zet2) * ajc * nu_eff;
        }
    );
}

inline void ViscousKernel::combineFluxes(
    const VectorView& fp1,
    const VectorView& fp2,
    const VectorView& fp3,
    VectorView& visc,
    const KernelDomainInfo& domain
) {
    const int lxs = domain.lxs;
    const int lxe = domain.lxe;
    const int lys = domain.lys;
    const int lye = domain.lye;
    const int lzs = domain.lzs;
    const int lze = domain.lze;

    Kokkos::parallel_for("Viscous_Combine",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            visc(k, j, i).x =
                fp1(k, j, i).x - fp1(k, j, i-1).x +
                fp2(k, j, i).x - fp2(k, j-1, i).x +
                fp3(k, j, i).x - fp3(k-1, j, i).x;

            visc(k, j, i).y =
                fp1(k, j, i).y - fp1(k, j, i-1).y +
                fp2(k, j, i).y - fp2(k, j-1, i).y +
                fp3(k, j, i).y - fp3(k-1, j, i).y;

            visc(k, j, i).z =
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
#endif // VFSWIND_VISCOUS_KERNEL_HPP
