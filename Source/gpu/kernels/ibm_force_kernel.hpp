/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * IBM Force Spreading Kernel for VFS-Wind                        *
 * Phase 3: Immersed Boundary Method GPU Acceleration             *
 ******************************************************************/

#ifndef VFSWIND_IBM_FORCE_KERNEL_HPP
#define VFSWIND_IBM_FORCE_KERNEL_HPP

#ifdef ENABLE_GPU

#include "ibm_types.hpp"
#include "petsc_kokkos.hpp"
#include <Kokkos_Core.hpp>

namespace vfswind {
namespace gpu {
namespace ibm {

/**
 * @brief IBM Force Spreading Kernel
 *
 * Spreads forces from IBM Lagrangian points to the Eulerian grid.
 * Uses regularized delta functions for smooth force distribution.
 */
class IBMForceKernel {
public:
    using VectorView = VectorView3D<DeviceMemorySpace>;
    using ScalarView = ScalarView3D<DeviceMemorySpace>;

    /**
     * @brief Compute forces at Lagrangian points using direct forcing
     *
     * F = (U_desired - U_interpolated) / dt
     *
     * @param surface IBM surface mesh (contains desired velocities)
     * @param interp Interpolation data
     * @param forces Force data (U_x,y,z contains interpolated velocities)
     * @param dt Time step
     */
    static void computeDirectForcing(
        const IBMSurfaceMesh<DeviceMemorySpace>& surface,
        const IBMInterpData<DeviceMemorySpace>& interp,
        IBMForceData<DeviceMemorySpace>& forces,
        double dt
    ) {
        const int n_points = interp.n_points;

        auto element_idx = interp.element_idx;
        auto mode = interp.mode;

        auto cs1 = interp.cs1;
        auto cs2 = interp.cs2;
        auto cs3 = interp.cs3;

        // Surface data
        auto nv1 = surface.nv1;
        auto nv2 = surface.nv2;
        auto nv3 = surface.nv3;
        auto surf_ux = surface.u_x;
        auto surf_uy = surface.u_y;
        auto surf_uz = surface.u_z;

        // Force data
        auto U_x = forces.U_x;
        auto U_y = forces.U_y;
        auto U_z = forces.U_z;
        auto F_x = forces.F_x;
        auto F_y = forces.F_y;
        auto F_z = forces.F_z;

        const double inv_dt = 1.0 / dt;

        Kokkos::parallel_for("IBMComputeDirectForcing",
            Kokkos::RangePolicy<>(0, n_points),
            KOKKOS_LAMBDA(int p) {
                if (mode(p) == 0) {
                    F_x(p) = 0.0;
                    F_y(p) = 0.0;
                    F_z(p) = 0.0;
                    return;
                }

                const int elem = element_idx(p);
                if (elem < 0) {
                    F_x(p) = 0.0;
                    F_y(p) = 0.0;
                    F_z(p) = 0.0;
                    return;
                }

                // Get surface vertices
                const int v1 = nv1(elem);
                const int v2 = nv2(elem);
                const int v3 = nv3(elem);

                // Barycentric interpolation of desired velocity
                const double c1 = cs1(p);
                const double c2 = cs2(p);
                const double c3 = cs3(p);

                const double U_des_x = surf_ux(v1) * c1 + surf_ux(v2) * c2 + surf_ux(v3) * c3;
                const double U_des_y = surf_uy(v1) * c1 + surf_uy(v2) * c2 + surf_uy(v3) * c3;
                const double U_des_z = surf_uz(v1) * c1 + surf_uz(v2) * c2 + surf_uz(v3) * c3;

                // Direct forcing: F = (U_desired - U_interpolated) / dt
                F_x(p) = (U_des_x - U_x(p)) * inv_dt;
                F_y(p) = (U_des_y - U_y(p)) * inv_dt;
                F_z(p) = (U_des_z - U_z(p)) * inv_dt;
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Regularized delta function (Roma et al. 1999)
     *
     * 3-point hat function: phi(r) for |r| <= 1.5h
     */
    KOKKOS_INLINE_FUNCTION
    static double deltaFunction(double r, double h) {
        const double s = r / h;
        const double abs_s = Kokkos::fabs(s);

        if (abs_s >= 1.5) {
            return 0.0;
        } else if (abs_s >= 0.5) {
            // 0.5 <= |s| < 1.5
            const double t = 1.5 - abs_s;
            return (1.0 / (6.0 * h)) * t * t;
        } else {
            // |s| < 0.5
            return (1.0 / h) * (2.0/3.0 - s*s + 0.5 * abs_s * abs_s * abs_s);
        }
    }

    /**
     * @brief Spread forces from Lagrangian points to Eulerian grid
     *
     * Uses regularized delta functions for smooth spreading.
     * f_euler(x) = sum_l F_lagr(l) * delta(x - X_l) * dA_l
     *
     * Note: This kernel uses atomic operations for thread safety
     * when multiple Lagrangian points contribute to the same Eulerian cell.
     *
     * @param forces Lagrangian force data
     * @param surface Surface mesh (for element areas and centroids)
     * @param interp Interpolation data
     * @param f_body Output: body force field on Eulerian grid
     * @param h Grid spacing (assumed uniform)
     * @param domain Domain info
     */
    static void spreadForces(
        const IBMForceData<DeviceMemorySpace>& forces,
        const IBMSurfaceMesh<DeviceMemorySpace>& surface,
        const IBMInterpData<DeviceMemorySpace>& interp,
        VectorView& f_body,
        double h,
        const KernelDomainInfo& domain
    ) {
        const int n_points = interp.n_points;

        // Get interpolation point locations (cell centers where IBM applies)
        auto cell_i = interp.cell_i;
        auto cell_j = interp.cell_j;
        auto cell_k = interp.cell_k;
        auto element_idx = interp.element_idx;
        auto mode = interp.mode;

        // Force data
        auto F_x = forces.F_x;
        auto F_y = forces.F_y;
        auto F_z = forces.F_z;

        // Surface areas
        auto dA = surface.dA;

        // First, zero the body force field
        Kokkos::deep_copy(f_body, Cmpnts3());

        // Spread forces using atomic operations
        // For each Lagrangian point, add its contribution to nearby Eulerian cells
        Kokkos::parallel_for("IBMSpreadForces",
            Kokkos::RangePolicy<>(0, n_points),
            KOKKOS_LAMBDA(int p) {
                if (mode(p) == 0) return;

                const int i = cell_i(p);
                const int j = cell_j(p);
                const int k = cell_k(p);

                const int elem = element_idx(p);
                if (elem < 0) return;

                // Get force and element area
                const double fx = F_x(p);
                const double fy = F_y(p);
                const double fz = F_z(p);
                const double area = dA(elem);

                // For now, use simple point-to-cell spreading
                // (direct assignment to nearest cell)
                // A more sophisticated approach would use delta functions
                // to spread to neighboring cells as well

                // Atomic add to handle multiple points per cell
                Kokkos::atomic_add(&f_body(k, j, i).x, fx * area);
                Kokkos::atomic_add(&f_body(k, j, i).y, fy * area);
                Kokkos::atomic_add(&f_body(k, j, i).z, fz * area);
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Spread forces with regularized delta function support
     *
     * More accurate spreading using 3-point delta function stencil.
     * Each Lagrangian point spreads to a 3x3x3 neighborhood.
     *
     * @param forces Lagrangian force data
     * @param surface Surface mesh
     * @param cent_x, cent_y, cent_z Element centroid coordinates
     * @param f_body Output: body force field
     * @param x_grid, y_grid, z_grid Grid coordinate arrays
     * @param hx, hy, hz Grid spacings
     * @param domain Domain info
     */
    static void spreadForcesRegularized(
        const IBMForceData<DeviceMemorySpace>& forces,
        const IBMSurfaceMesh<DeviceMemorySpace>& surface,
        const ScalarView3D<DeviceMemorySpace>& x_grid,
        const ScalarView3D<DeviceMemorySpace>& y_grid,
        const ScalarView3D<DeviceMemorySpace>& z_grid,
        VectorView& f_body,
        double hx, double hy, double hz,
        const KernelDomainInfo& domain
    ) {
        const int n_elements = surface.n_elements;

        auto cent_x = surface.cent_x;
        auto cent_y = surface.cent_y;
        auto cent_z = surface.cent_z;
        auto dA = surface.dA;

        auto F_x = forces.F_x;
        auto F_y = forces.F_y;
        auto F_z = forces.F_z;

        const int mx = domain.mx;
        const int my = domain.my;
        const int mz = domain.mz;

        // Zero body force
        Kokkos::deep_copy(f_body, Cmpnts3());

        // For each surface element, spread force to nearby cells
        Kokkos::parallel_for("IBMSpreadForcesRegularized",
            Kokkos::RangePolicy<>(0, n_elements),
            KOKKOS_LAMBDA(int e) {
                const double px = cent_x(e);
                const double py = cent_y(e);
                const double pz = cent_z(e);

                const double fx = F_x(e);
                const double fy = F_y(e);
                const double fz = F_z(e);
                const double area = dA(e);

                // Find the cell containing this point
                // This is approximate - assumes uniform grid starting at origin
                const int i0 = static_cast<int>(px / hx);
                const int j0 = static_cast<int>(py / hy);
                const int k0 = static_cast<int>(pz / hz);

                // Spread to 3x3x3 neighborhood
                for (int dk = -1; dk <= 1; dk++) {
                    int k = k0 + dk;
                    if (k < 1 || k >= mz - 1) continue;

                    const double zk = k * hz;
                    const double delta_z = deltaFunction(pz - zk, hz);
                    if (delta_z < 1e-12) continue;

                    for (int dj = -1; dj <= 1; dj++) {
                        int j = j0 + dj;
                        if (j < 1 || j >= my - 1) continue;

                        const double yj = j * hy;
                        const double delta_y = deltaFunction(py - yj, hy);
                        if (delta_y < 1e-12) continue;

                        for (int di = -1; di <= 1; di++) {
                            int i = i0 + di;
                            if (i < 1 || i >= mx - 1) continue;

                            const double xi = i * hx;
                            const double delta_x = deltaFunction(px - xi, hx);
                            if (delta_x < 1e-12) continue;

                            // 3D delta function
                            const double delta = delta_x * delta_y * delta_z;

                            // Spread force
                            Kokkos::atomic_add(&f_body(k, j, i).x, fx * area * delta);
                            Kokkos::atomic_add(&f_body(k, j, i).y, fy * area * delta);
                            Kokkos::atomic_add(&f_body(k, j, i).z, fz * area * delta);
                        }
                    }
                }
            }
        );

        Kokkos::fence();
    }

    /**
     * @brief Add body forces to RHS
     *
     * @param rhs RHS vector (modified in place)
     * @param f_body Body force field
     * @param nvert Solid cell marker
     * @param domain Domain info
     */
    static void addBodyForceToRHS(
        VectorView& rhs,
        const VectorView& f_body,
        const ScalarView& nvert,
        const KernelDomainInfo& domain
    ) {
        const int lxs = domain.lxs, lxe = domain.lxe;
        const int lys = domain.lys, lye = domain.lye;
        const int lzs = domain.lzs, lze = domain.lze;

        Kokkos::parallel_for("IBMAddBodyForce",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({lzs, lys, lxs}, {lze, lye, lxe}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                // Skip solid cells
                if (static_cast<int>(nvert(k, j, i) + 0.5) >= 1) return;

                rhs(k, j, i).x += f_body(k, j, i).x;
                rhs(k, j, i).y += f_body(k, j, i).y;
                rhs(k, j, i).z += f_body(k, j, i).z;
            }
        );

        Kokkos::fence();
    }
};

} // namespace ibm
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_IBM_FORCE_KERNEL_HPP
