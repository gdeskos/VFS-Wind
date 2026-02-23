/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * PETSc-Kokkos Integration for VFS-Wind                          *
 * Provides utilities to convert between PETSc and Kokkos Views   *
 ******************************************************************/

#ifndef VFSWIND_PETSC_KOKKOS_HPP
#define VFSWIND_PETSC_KOKKOS_HPP

#ifdef ENABLE_GPU

#include "../gpu_config.hpp"
#include <Kokkos_Core.hpp>

// PETSc headers - must NOT be inside extern "C" for C++
#include "petscdmda.h"
#include "petscvec.h"

namespace vfswind {
namespace gpu {

// ============================================================================
// Cmpnts structure for GPU (mirrors variables.h Cmpnts)
// ============================================================================

struct Cmpnts3 {
    double x, y, z;

    KOKKOS_INLINE_FUNCTION
    Cmpnts3() : x(0.0), y(0.0), z(0.0) {}

    KOKKOS_INLINE_FUNCTION
    Cmpnts3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

// ============================================================================
// Domain Information
// ============================================================================

struct KernelDomainInfo {
    // Local owned range (interior points)
    int xs, xe, ys, ye, zs, ze;

    // Global dimensions
    int mx, my, mz;

    // Loop bounds (excluding boundaries)
    int lxs, lxe, lys, lye, lzs, lze;

    // Reynolds number (for viscous term)
    double ren;

    // Flags
    bool les;
    bool rans;
    int ti;  // time step
};

// Initialize domain info from PETSc DMDA
inline KernelDomainInfo createDomainInfo(DM da, double ren = 1000.0) {
    KernelDomainInfo info;
    DMDALocalInfo dainfo;
    DMDAGetLocalInfo(da, &dainfo);

    info.mx = dainfo.mx;
    info.my = dainfo.my;
    info.mz = dainfo.mz;
    info.xs = dainfo.xs;
    info.xe = dainfo.xs + dainfo.xm;
    info.ys = dainfo.ys;
    info.ye = dainfo.ys + dainfo.ym;
    info.zs = dainfo.zs;
    info.ze = dainfo.zs + dainfo.zm;

    // Compute interior loop bounds
    info.lxs = info.xs;
    info.lxe = info.xe;
    info.lys = info.ys;
    info.lye = info.ye;
    info.lzs = info.zs;
    info.lze = info.ze;

    if (info.xs == 0) info.lxs = info.xs + 1;
    if (info.ys == 0) info.lys = info.ys + 1;
    if (info.zs == 0) info.lzs = info.zs + 1;
    if (info.xe == info.mx) info.lxe = info.xe - 1;
    if (info.ye == info.my) info.lye = info.ye - 1;
    if (info.ze == info.mz) info.lze = info.ze - 1;

    info.ren = ren;
    info.les = false;
    info.rans = false;
    info.ti = 0;

    return info;
}

// ============================================================================
// Kokkos View Type Aliases for VFS-Wind
// ============================================================================

// Scalar field: [k][j][i]
template<typename Space = DeviceMemorySpace>
using ScalarView3D = Kokkos::View<double***, Kokkos::LayoutRight, Space>;

// Vector field: [k][j][i] of Cmpnts3
template<typename Space = DeviceMemorySpace>
using VectorView3D = Kokkos::View<Cmpnts3***, Kokkos::LayoutRight, Space>;

// Alternative vector field: [k][j][i][component]
template<typename Space = DeviceMemorySpace>
using VectorView4D = Kokkos::View<double****, Kokkos::LayoutRight, Space>;

// ============================================================================
// PETSc to Kokkos Conversion Utilities
// ============================================================================

/**
 * @brief Copy a PETSc scalar Vec to a Kokkos View
 *
 * @param da PETSc DMDA for the scalar field
 * @param vec PETSc Vec (local vector with ghosts)
 * @param view Kokkos View to copy into (must be pre-allocated)
 */
template<typename Space>
void copyPetscScalarToView(DM da, Vec vec, ScalarView3D<Space>& view) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    double*** array;
    DMDAVecGetArray(da, vec, &array);

    // Create host mirror
    auto h_view = Kokkos::create_mirror_view(view);

    // Copy data
    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                h_view(k, j, i) = array[k][j][i];
            }
        }
    }

    // Copy to device
    Kokkos::deep_copy(view, h_view);

    DMDAVecRestoreArray(da, vec, &array);
}

/**
 * @brief Copy a PETSc Cmpnts Vec to a Kokkos VectorView3D
 *
 * @param fda PETSc DMDA for the vector field (dof=3)
 * @param vec PETSc Vec (local vector with ghosts)
 * @param view Kokkos View to copy into (must be pre-allocated)
 */
template<typename Space>
void copyPetscVectorToView(DM fda, Vec vec, VectorView3D<Space>& view) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(fda, &info);

    // Get array as Cmpnts (assumes PETSc struct matches our Cmpnts3)
    struct PetscCmpnts { double x, y, z; };
    PetscCmpnts*** array;
    DMDAVecGetArray(fda, vec, &array);

    // Create host mirror
    auto h_view = Kokkos::create_mirror_view(view);

    // Copy data
    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                h_view(k, j, i).x = array[k][j][i].x;
                h_view(k, j, i).y = array[k][j][i].y;
                h_view(k, j, i).z = array[k][j][i].z;
            }
        }
    }

    // Copy to device
    Kokkos::deep_copy(view, h_view);

    DMDAVecRestoreArray(fda, vec, &array);
}

/**
 * @brief Copy a Kokkos VectorView3D back to PETSc Vec
 *
 * @param fda PETSc DMDA for the vector field (dof=3)
 * @param view Kokkos View source
 * @param vec PETSc Vec destination
 */
template<typename Space>
void copyViewToPetscVector(DM fda, const VectorView3D<Space>& view, Vec vec) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(fda, &info);

    // Copy to host first
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    // Get PETSc array
    struct PetscCmpnts { double x, y, z; };
    PetscCmpnts*** array;
    DMDAVecGetArray(fda, vec, &array);

    // Copy data
    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                array[k][j][i].x = h_view(k, j, i).x;
                array[k][j][i].y = h_view(k, j, i).y;
                array[k][j][i].z = h_view(k, j, i).z;
            }
        }
    }

    DMDAVecRestoreArray(fda, vec, &array);
}

/**
 * @brief Copy a Kokkos ScalarView3D back to PETSc Vec
 */
template<typename Space>
void copyViewToPetscScalar(DM da, const ScalarView3D<Space>& view, Vec vec) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    // Copy to host first
    auto h_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);

    // Get PETSc array
    double*** array;
    DMDAVecGetArray(da, vec, &array);

    // Copy data
    for (int k = info.zs; k < info.zs + info.zm; ++k) {
        for (int j = info.ys; j < info.ys + info.ym; ++j) {
            for (int i = info.xs; i < info.xs + info.xm; ++i) {
                array[k][j][i] = h_view(k, j, i);
            }
        }
    }

    DMDAVecRestoreArray(da, vec, &array);
}

// ============================================================================
// View Allocation Helpers
// ============================================================================

/**
 * @brief Allocate a scalar view matching DMDA dimensions
 */
template<typename Space = DeviceMemorySpace>
ScalarView3D<Space> allocateScalarView(DM da, const char* label) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    return ScalarView3D<Space>(label, info.mz, info.my, info.mx);
}

/**
 * @brief Allocate a vector view matching DMDA dimensions
 */
template<typename Space = DeviceMemorySpace>
VectorView3D<Space> allocateVectorView(DM fda, const char* label) {
    DMDALocalInfo info;
    DMDAGetLocalInfo(fda, &info);
    return VectorView3D<Space>(label, info.mz, info.my, info.mx);
}

} // namespace gpu
} // namespace vfswind

#endif // ENABLE_GPU
#endif // VFSWIND_PETSC_KOKKOS_HPP
