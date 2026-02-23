/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * IBM GPU Data Structures for VFS-Wind                           *
 * Phase 3: Immersed Boundary Method GPU Acceleration             *
 ******************************************************************/

#ifndef VFSWIND_IBM_TYPES_HPP
#define VFSWIND_IBM_TYPES_HPP

#ifdef ENABLE_GPU

#include "../gpu_config.hpp"
#include "petsc_kokkos.hpp"
#include <Kokkos_Core.hpp>

namespace vfswind {
namespace gpu {
namespace ibm {

// ============================================================================
// GPU-Friendly IBM Surface Mesh Data (Structure of Arrays)
// ============================================================================

/**
 * @brief GPU-friendly IBM surface mesh storage
 *
 * Uses Structure of Arrays (SoA) layout for coalesced memory access.
 * All arrays are indexed by vertex or element number.
 */
template<typename Space = DeviceMemorySpace>
struct IBMSurfaceMesh {
    // Number of vertices and elements
    int n_vertices;
    int n_elements;

    // Vertex coordinates (size: n_vertices)
    Kokkos::View<double*, Space> x_bp;
    Kokkos::View<double*, Space> y_bp;
    Kokkos::View<double*, Space> z_bp;

    // Element connectivity - indices into vertex arrays (size: n_elements)
    Kokkos::View<int*, Space> nv1;
    Kokkos::View<int*, Space> nv2;
    Kokkos::View<int*, Space> nv3;

    // Element normals (size: n_elements)
    Kokkos::View<double*, Space> nf_x;
    Kokkos::View<double*, Space> nf_y;
    Kokkos::View<double*, Space> nf_z;

    // Element centroids (size: n_elements)
    Kokkos::View<double*, Space> cent_x;
    Kokkos::View<double*, Space> cent_y;
    Kokkos::View<double*, Space> cent_z;

    // Element areas (size: n_elements)
    Kokkos::View<double*, Space> dA;

    // Vertex velocities (for moving bodies, size: n_vertices)
    Kokkos::View<double*, Space> u_x;
    Kokkos::View<double*, Space> u_y;
    Kokkos::View<double*, Space> u_z;

    // Default constructor
    IBMSurfaceMesh() : n_vertices(0), n_elements(0) {}

    // Allocate with given sizes
    void allocate(int nv, int ne) {
        n_vertices = nv;
        n_elements = ne;

        // Vertex data
        x_bp = Kokkos::View<double*, Space>("ibm_x_bp", nv);
        y_bp = Kokkos::View<double*, Space>("ibm_y_bp", nv);
        z_bp = Kokkos::View<double*, Space>("ibm_z_bp", nv);
        u_x = Kokkos::View<double*, Space>("ibm_u_x", nv);
        u_y = Kokkos::View<double*, Space>("ibm_u_y", nv);
        u_z = Kokkos::View<double*, Space>("ibm_u_z", nv);

        // Element data
        nv1 = Kokkos::View<int*, Space>("ibm_nv1", ne);
        nv2 = Kokkos::View<int*, Space>("ibm_nv2", ne);
        nv3 = Kokkos::View<int*, Space>("ibm_nv3", ne);
        nf_x = Kokkos::View<double*, Space>("ibm_nf_x", ne);
        nf_y = Kokkos::View<double*, Space>("ibm_nf_y", ne);
        nf_z = Kokkos::View<double*, Space>("ibm_nf_z", ne);
        cent_x = Kokkos::View<double*, Space>("ibm_cent_x", ne);
        cent_y = Kokkos::View<double*, Space>("ibm_cent_y", ne);
        cent_z = Kokkos::View<double*, Space>("ibm_cent_z", ne);
        dA = Kokkos::View<double*, Space>("ibm_dA", ne);
    }
};

// ============================================================================
// IBM Interpolation Data
// ============================================================================

/**
 * @brief Interpolation point data for a single IBM point
 *
 * Contains the stencil information needed to interpolate from
 * the Eulerian grid to a Lagrangian IBM point.
 */
struct IBMInterpPoint {
    // Grid cell containing the IBM point
    int i, j, k;

    // Nearest surface element index
    int element_idx;

    // Three interpolation stencil points (grid indices)
    int i1, j1, k1;
    int i2, j2, k2;
    int i3, j3, k3;

    // Interpolation coefficients for the three points
    double cr1, cr2, cr3;

    // Barycentric coordinates on the surface element
    double cs1, cs2, cs3;

    // Distance to surface and to interpolation point
    double d_surface;
    double d_interp;

    // Interpolation mode (1=valid, 0=invalid)
    int mode;
};

/**
 * @brief GPU storage for all IBM interpolation points
 *
 * Uses SoA layout for efficient GPU access.
 */
template<typename Space = DeviceMemorySpace>
struct IBMInterpData {
    // Number of interpolation points
    int n_points;

    // Grid cell indices (size: n_points)
    Kokkos::View<int*, Space> cell_i;
    Kokkos::View<int*, Space> cell_j;
    Kokkos::View<int*, Space> cell_k;

    // Surface element index (size: n_points)
    Kokkos::View<int*, Space> element_idx;

    // First interpolation point (size: n_points)
    Kokkos::View<int*, Space> i1, j1, k1;
    Kokkos::View<double*, Space> cr1;

    // Second interpolation point (size: n_points)
    Kokkos::View<int*, Space> i2, j2, k2;
    Kokkos::View<double*, Space> cr2;

    // Third interpolation point (size: n_points)
    Kokkos::View<int*, Space> i3, j3, k3;
    Kokkos::View<double*, Space> cr3;

    // Barycentric coordinates on surface element (size: n_points)
    Kokkos::View<double*, Space> cs1, cs2, cs3;

    // Distances (size: n_points)
    Kokkos::View<double*, Space> d_surface;
    Kokkos::View<double*, Space> d_interp;

    // Mode flag (size: n_points)
    Kokkos::View<int*, Space> mode;

    // Default constructor
    IBMInterpData() : n_points(0) {}

    // Allocate with given size
    void allocate(int np) {
        n_points = np;

        cell_i = Kokkos::View<int*, Space>("ibm_cell_i", np);
        cell_j = Kokkos::View<int*, Space>("ibm_cell_j", np);
        cell_k = Kokkos::View<int*, Space>("ibm_cell_k", np);
        element_idx = Kokkos::View<int*, Space>("ibm_element_idx", np);

        i1 = Kokkos::View<int*, Space>("ibm_i1", np);
        j1 = Kokkos::View<int*, Space>("ibm_j1", np);
        k1 = Kokkos::View<int*, Space>("ibm_k1", np);
        cr1 = Kokkos::View<double*, Space>("ibm_cr1", np);

        i2 = Kokkos::View<int*, Space>("ibm_i2", np);
        j2 = Kokkos::View<int*, Space>("ibm_j2", np);
        k2 = Kokkos::View<int*, Space>("ibm_k2", np);
        cr2 = Kokkos::View<double*, Space>("ibm_cr2", np);

        i3 = Kokkos::View<int*, Space>("ibm_i3", np);
        j3 = Kokkos::View<int*, Space>("ibm_j3", np);
        k3 = Kokkos::View<int*, Space>("ibm_k3", np);
        cr3 = Kokkos::View<double*, Space>("ibm_cr3", np);

        cs1 = Kokkos::View<double*, Space>("ibm_cs1", np);
        cs2 = Kokkos::View<double*, Space>("ibm_cs2", np);
        cs3 = Kokkos::View<double*, Space>("ibm_cs3", np);

        d_surface = Kokkos::View<double*, Space>("ibm_d_surface", np);
        d_interp = Kokkos::View<double*, Space>("ibm_d_interp", np);

        mode = Kokkos::View<int*, Space>("ibm_mode", np);
    }
};

// ============================================================================
// IBM Force Data
// ============================================================================

/**
 * @brief Force data at IBM Lagrangian points
 */
template<typename Space = DeviceMemorySpace>
struct IBMForceData {
    // Number of force points (typically = n_elements)
    int n_points;

    // Lagrangian forces (size: n_points)
    Kokkos::View<double*, Space> F_x;
    Kokkos::View<double*, Space> F_y;
    Kokkos::View<double*, Space> F_z;

    // Interpolated velocities at Lagrangian points (size: n_points)
    Kokkos::View<double*, Space> U_x;
    Kokkos::View<double*, Space> U_y;
    Kokkos::View<double*, Space> U_z;

    // Default constructor
    IBMForceData() : n_points(0) {}

    // Allocate with given size
    void allocate(int np) {
        n_points = np;

        F_x = Kokkos::View<double*, Space>("ibm_F_x", np);
        F_y = Kokkos::View<double*, Space>("ibm_F_y", np);
        F_z = Kokkos::View<double*, Space>("ibm_F_z", np);

        U_x = Kokkos::View<double*, Space>("ibm_U_x", np);
        U_y = Kokkos::View<double*, Space>("ibm_U_y", np);
        U_z = Kokkos::View<double*, Space>("ibm_U_z", np);
    }

    // Zero all forces
    void zeroForces() {
        Kokkos::deep_copy(F_x, 0.0);
        Kokkos::deep_copy(F_y, 0.0);
        Kokkos::deep_copy(F_z, 0.0);
    }
};

// ============================================================================
// Type Aliases for Convenience
// ============================================================================

using DeviceIBMSurfaceMesh = IBMSurfaceMesh<DeviceMemorySpace>;
using HostIBMSurfaceMesh = IBMSurfaceMesh<Kokkos::HostSpace>;

using DeviceIBMInterpData = IBMInterpData<DeviceMemorySpace>;
using HostIBMInterpData = IBMInterpData<Kokkos::HostSpace>;

using DeviceIBMForceData = IBMForceData<DeviceMemorySpace>;
using HostIBMForceData = IBMForceData<Kokkos::HostSpace>;

} // namespace ibm
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_IBM_TYPES_HPP
