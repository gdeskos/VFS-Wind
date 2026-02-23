/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * IBM GPU Kernels - Unified Header                               *
 * Phase 3: Immersed Boundary Method GPU Acceleration             *
 ******************************************************************/

#ifndef VFSWIND_IBM_KERNELS_HPP
#define VFSWIND_IBM_KERNELS_HPP

#ifdef ENABLE_GPU

// IBM data structures
#include "ibm_types.hpp"

// IBM kernels
#include "ibm_interpolation_kernel.hpp"
#include "ibm_force_kernel.hpp"

namespace vfswind {
namespace gpu {
namespace ibm {

/**
 * @brief Execute complete IBM step
 *
 * This is a convenience function that performs all IBM operations:
 * 1. Interpolate velocity to Lagrangian points
 * 2. Compute forces at Lagrangian points (direct forcing)
 * 3. Spread forces to Eulerian grid
 * 4. Apply boundary conditions (no-slip or free-slip)
 *
 * @param ucat Cartesian velocity field (modified for BC)
 * @param rhs RHS vector (modified with body forces)
 * @param nvert Solid cell marker
 * @param surface IBM surface mesh
 * @param interp Interpolation data
 * @param forces Force data (workspace)
 * @param f_body Body force field (workspace)
 * @param dt Time step
 * @param h Grid spacing
 * @param domain Domain info
 * @param use_noslip Use no-slip (true) or free-slip (false)
 */
inline void executeIBMStep(
    VectorView3D<DeviceMemorySpace>& ucat,
    VectorView3D<DeviceMemorySpace>& rhs,
    const ScalarView3D<DeviceMemorySpace>& nvert,
    const IBMSurfaceMesh<DeviceMemorySpace>& surface,
    const IBMInterpData<DeviceMemorySpace>& interp,
    IBMForceData<DeviceMemorySpace>& forces,
    VectorView3D<DeviceMemorySpace>& f_body,
    double dt,
    double h,
    const KernelDomainInfo& domain,
    bool use_noslip = true
) {
    // Step 1: Interpolate velocity to Lagrangian points
    IBMInterpolationKernel::interpolateVelocity(ucat, interp, forces, domain);

    // Step 2: Compute forces using direct forcing
    IBMForceKernel::computeDirectForcing(surface, interp, forces, dt);

    // Step 3: Spread forces to Eulerian grid
    IBMForceKernel::spreadForces(forces, surface, interp, f_body, h, domain);

    // Step 4: Add body forces to RHS
    IBMForceKernel::addBodyForceToRHS(rhs, f_body, nvert, domain);

    // Step 5: Apply boundary condition at IBM cells
    if (use_noslip) {
        IBMInterpolationKernel::applyNoSlip(ucat, surface, interp, domain);
    } else {
        IBMInterpolationKernel::applyFreeSlip(ucat, surface, interp, domain);
    }
}

/**
 * @brief Helper to allocate IBM workspace
 */
inline void allocateIBMWorkspace(
    IBMForceData<DeviceMemorySpace>& forces,
    VectorView3D<DeviceMemorySpace>& f_body,
    int n_interp_points,
    int mz, int my, int mx
) {
    forces.allocate(n_interp_points);
    f_body = VectorView3D<DeviceMemorySpace>("ibm_f_body", mz, my, mx);
    Kokkos::deep_copy(f_body, Cmpnts3());
}

} // namespace ibm
} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_IBM_KERNELS_HPP
