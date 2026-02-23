/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * IBM Interpolation Kernel for VFS-Wind                          *
 * Phase 3: Immersed Boundary Method GPU Acceleration             *
 ******************************************************************/

#ifndef VFSWIND_IBM_INTERPOLATION_KERNEL_HPP
#define VFSWIND_IBM_INTERPOLATION_KERNEL_HPP

#ifdef ENABLE_GPU

#include "ibm_types.hpp"
#include "petsc_kokkos.hpp"
#include <Kokkos_Core.hpp>

namespace vfswind {
namespace gpu {
namespace ibm {

/**
 * @brief IBM Velocity Interpolation Kernel
 *
 * Interpolates velocities from the Eulerian grid to IBM Lagrangian points.
 * This implements the velocity interpolation part of ibm_interpolation_advanced.
 */
class IBMInterpolationKernel {
public:
    using VectorView = VectorView3D<DeviceMemorySpace>;
    using ScalarView = ScalarView3D<DeviceMemorySpace>;

    /**
     * @brief Interpolate velocity to IBM points
     *
     * For each IBM interpolation point, computes the interpolated velocity
     * using the three-point stencil and interpolation coefficients.
     *
     * @param ucat Cartesian velocity field [k][j][i]
     * @param interp Interpolation data (stencil indices, coefficients)
     * @param forces Output: interpolated velocities stored in U_x, U_y, U_z
     * @param domain Domain information
     */
    static void interpolateVelocity(
        const VectorView& ucat,
        const IBMInterpData<DeviceMemorySpace>& interp,
        IBMForceData<DeviceMemorySpace>& forces,
        const KernelDomainInfo& domain
    ) {
        const int n_points = interp.n_points;

        // Get views for interpolation data
        auto cell_i = interp.cell_i;
        auto cell_j = interp.cell_j;
        auto cell_k = interp.cell_k;

        auto i1 = interp.i1; auto j1 = interp.j1; auto k1 = interp.k1;
        auto i2 = interp.i2; auto j2 = interp.j2; auto k2 = interp.k2;
        auto i3 = interp.i3; auto j3 = interp.j3; auto k3 = interp.k3;

        auto cr1 = interp.cr1;
        auto cr2 = interp.cr2;
        auto cr3 = interp.cr3;

        auto mode = interp.mode;

        // Output views
        auto U_x = forces.U_x;
        auto U_y = forces.U_y;
        auto U_z = forces.U_z;

        Kokkos::parallel_for("IBMInterpolateVelocity",
            Kokkos::RangePolicy<>(0, n_points),
            KOKKOS_LAMBDA(int p) {
                // Skip invalid points
                if (mode(p) == 0) {
                    U_x(p) = 0.0;
                    U_y(p) = 0.0;
                    U_z(p) = 0.0;
                    return;
                }

                // Get stencil indices
                const int ip1 = i1(p), jp1 = j1(p), kp1 = k1(p);
                const int ip2 = i2(p), jp2 = j2(p), kp2 = k2(p);
                const int ip3 = i3(p), jp3 = j3(p), kp3 = k3(p);

                // Get interpolation coefficients
                const double c1 = cr1(p);
                const double c2 = cr2(p);
                const double c3 = cr3(p);

                // Interpolate velocity
                U_x(p) = ucat(kp1, jp1, ip1).x * c1 +
                         ucat(kp2, jp2, ip2).x * c2 +
                         ucat(kp3, jp3, ip3).x * c3;

                U_y(p) = ucat(kp1, jp1, ip1).y * c1 +
                         ucat(kp2, jp2, ip2).y * c2 +
                         ucat(kp3, jp3, ip3).y * c3;

                U_z(p) = ucat(kp1, jp1, ip1).z * c1 +
                         ucat(kp2, jp2, ip2).z * c2 +
                         ucat(kp3, jp3, ip3).z * c3;
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Apply no-slip boundary condition at IBM points
     *
     * Modifies the velocity at IBM cell centers to satisfy no-slip
     * using linear interpolation between surface and interior.
     *
     * @param ucat Cartesian velocity field (modified in place)
     * @param surface IBM surface mesh (for boundary velocities)
     * @param interp Interpolation data
     * @param domain Domain information
     */
    static void applyNoSlip(
        VectorView& ucat,
        const IBMSurfaceMesh<DeviceMemorySpace>& surface,
        const IBMInterpData<DeviceMemorySpace>& interp,
        const KernelDomainInfo& domain
    ) {
        const int n_points = interp.n_points;

        // Interpolation data
        auto cell_i = interp.cell_i;
        auto cell_j = interp.cell_j;
        auto cell_k = interp.cell_k;
        auto element_idx = interp.element_idx;

        auto i1 = interp.i1; auto j1 = interp.j1; auto k1 = interp.k1;
        auto i2 = interp.i2; auto j2 = interp.j2; auto k2 = interp.k2;
        auto i3 = interp.i3; auto j3 = interp.j3; auto k3 = interp.k3;

        auto cr1 = interp.cr1;
        auto cr2 = interp.cr2;
        auto cr3 = interp.cr3;

        auto cs1 = interp.cs1;
        auto cs2 = interp.cs2;
        auto cs3 = interp.cs3;

        auto d_surface = interp.d_surface;
        auto d_interp = interp.d_interp;
        auto mode = interp.mode;

        // Surface mesh data
        auto nv1 = surface.nv1;
        auto nv2 = surface.nv2;
        auto nv3 = surface.nv3;
        auto surf_ux = surface.u_x;
        auto surf_uy = surface.u_y;
        auto surf_uz = surface.u_z;

        Kokkos::parallel_for("IBMApplyNoSlip",
            Kokkos::RangePolicy<>(0, n_points),
            KOKKOS_LAMBDA(int p) {
                // Skip invalid points
                if (mode(p) == 0) return;

                // Get cell indices
                const int i = cell_i(p);
                const int j = cell_j(p);
                const int k = cell_k(p);

                // Get surface element and vertices
                const int elem = element_idx(p);
                if (elem < 0) return;

                const int v1 = nv1(elem);
                const int v2 = nv2(elem);
                const int v3 = nv3(elem);

                // Surface velocity (interpolated on triangle)
                const double c1 = cs1(p);
                const double c2 = cs2(p);
                const double c3 = cs3(p);

                const double Ua_x = surf_ux(v1) * c1 + surf_ux(v2) * c2 + surf_ux(v3) * c3;
                const double Ua_y = surf_uy(v1) * c1 + surf_uy(v2) * c2 + surf_uy(v3) * c3;
                const double Ua_z = surf_uz(v1) * c1 + surf_uz(v2) * c2 + surf_uz(v3) * c3;

                // Stencil point velocity (Uc)
                const int ip1 = i1(p), jp1 = j1(p), kp1 = k1(p);
                const int ip2 = i2(p), jp2 = j2(p), kp2 = k2(p);
                const int ip3 = i3(p), jp3 = j3(p), kp3 = k3(p);

                const double sk1 = cr1(p);
                const double sk2 = cr2(p);
                const double sk3 = cr3(p);

                const double Uc_x = ucat(kp1, jp1, ip1).x * sk1 +
                                    ucat(kp2, jp2, ip2).x * sk2 +
                                    ucat(kp3, jp3, ip3).x * sk3;
                const double Uc_y = ucat(kp1, jp1, ip1).y * sk1 +
                                    ucat(kp2, jp2, ip2).y * sk2 +
                                    ucat(kp3, jp3, ip3).y * sk3;
                const double Uc_z = ucat(kp1, jp1, ip1).z * sk1 +
                                    ucat(kp2, jp2, ip2).z * sk2 +
                                    ucat(kp3, jp3, ip3).z * sk3;

                // Linear interpolation for no-slip
                // sb = distance to surface, sc = sb + d_interp
                const double sb = d_surface(p);
                const double sc = sb + d_interp(p);

                // Avoid division by zero
                if (sc < 1e-10) return;

                // No-slip: U_cell = Ua + (Uc - Ua) * sb / sc
                const double ratio = sb / sc;
                ucat(k, j, i).x = Ua_x + (Uc_x - Ua_x) * ratio;
                ucat(k, j, i).y = Ua_y + (Uc_y - Ua_y) * ratio;
                ucat(k, j, i).z = Ua_z + (Uc_z - Ua_z) * ratio;
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Apply free-slip boundary condition at IBM points
     *
     * Similar to no-slip but removes only the normal component
     *
     * @param ucat Cartesian velocity field (modified in place)
     * @param surface IBM surface mesh
     * @param interp Interpolation data
     * @param domain Domain information
     */
    static void applyFreeSlip(
        VectorView& ucat,
        const IBMSurfaceMesh<DeviceMemorySpace>& surface,
        const IBMInterpData<DeviceMemorySpace>& interp,
        const KernelDomainInfo& domain
    ) {
        const int n_points = interp.n_points;

        auto cell_i = interp.cell_i;
        auto cell_j = interp.cell_j;
        auto cell_k = interp.cell_k;
        auto element_idx = interp.element_idx;

        auto i1 = interp.i1; auto j1 = interp.j1; auto k1 = interp.k1;
        auto i2 = interp.i2; auto j2 = interp.j2; auto k2 = interp.k2;
        auto i3 = interp.i3; auto j3 = interp.j3; auto k3 = interp.k3;

        auto cr1 = interp.cr1;
        auto cr2 = interp.cr2;
        auto cr3 = interp.cr3;

        auto cs1 = interp.cs1;
        auto cs2 = interp.cs2;
        auto cs3 = interp.cs3;

        auto d_surface = interp.d_surface;
        auto d_interp = interp.d_interp;
        auto mode = interp.mode;

        // Surface normals
        auto nf_x = surface.nf_x;
        auto nf_y = surface.nf_y;
        auto nf_z = surface.nf_z;

        // Surface velocities
        auto nv1 = surface.nv1;
        auto nv2 = surface.nv2;
        auto nv3 = surface.nv3;
        auto surf_ux = surface.u_x;
        auto surf_uy = surface.u_y;
        auto surf_uz = surface.u_z;

        Kokkos::parallel_for("IBMApplyFreeSlip",
            Kokkos::RangePolicy<>(0, n_points),
            KOKKOS_LAMBDA(int p) {
                if (mode(p) == 0) return;

                const int i = cell_i(p);
                const int j = cell_j(p);
                const int k = cell_k(p);

                const int elem = element_idx(p);
                if (elem < 0) return;

                // Surface normal
                const double nx = nf_x(elem);
                const double ny = nf_y(elem);
                const double nz = nf_z(elem);

                // Surface velocity
                const int v1 = nv1(elem);
                const int v2 = nv2(elem);
                const int v3 = nv3(elem);

                const double c1 = cs1(p);
                const double c2 = cs2(p);
                const double c3 = cs3(p);

                const double Ua_x = surf_ux(v1) * c1 + surf_ux(v2) * c2 + surf_ux(v3) * c3;
                const double Ua_y = surf_uy(v1) * c1 + surf_uy(v2) * c2 + surf_uy(v3) * c3;
                const double Ua_z = surf_uz(v1) * c1 + surf_uz(v2) * c2 + surf_uz(v3) * c3;

                // Normal component of surface velocity
                const double Ua_n = Ua_x * nx + Ua_y * ny + Ua_z * nz;

                // Stencil point velocity
                const int ip1 = i1(p), jp1 = j1(p), kp1 = k1(p);
                const int ip2 = i2(p), jp2 = j2(p), kp2 = k2(p);
                const int ip3 = i3(p), jp3 = j3(p), kp3 = k3(p);

                const double sk1 = cr1(p);
                const double sk2 = cr2(p);
                const double sk3 = cr3(p);

                const double Uc_x = ucat(kp1, jp1, ip1).x * sk1 +
                                    ucat(kp2, jp2, ip2).x * sk2 +
                                    ucat(kp3, jp3, ip3).x * sk3;
                const double Uc_y = ucat(kp1, jp1, ip1).y * sk1 +
                                    ucat(kp2, jp2, ip2).y * sk2 +
                                    ucat(kp3, jp3, ip3).y * sk3;
                const double Uc_z = ucat(kp1, jp1, ip1).z * sk1 +
                                    ucat(kp2, jp2, ip2).z * sk2 +
                                    ucat(kp3, jp3, ip3).z * sk3;

                // Normal component of stencil velocity
                const double Uc_n = Uc_x * nx + Uc_y * ny + Uc_z * nz;

                const double sb = d_surface(p);
                const double sc = sb + d_interp(p);

                if (sc < 1e-10) return;

                const double ratio = sb / sc;

                // Free-slip: interpolate only normal component, keep tangential
                const double Un_cell = Ua_n + (Uc_n - Ua_n) * ratio;

                // Cell velocity = Uc - (Uc_n - Un_cell) * n
                ucat(k, j, i).x = Uc_x - (Uc_n - Un_cell) * nx;
                ucat(k, j, i).y = Uc_y - (Uc_n - Un_cell) * ny;
                ucat(k, j, i).z = Uc_z - (Uc_n - Un_cell) * nz;
            }
        );

        Kokkos::fence();
    }
};

} // namespace ibm
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_IBM_INTERPOLATION_KERNEL_HPP
