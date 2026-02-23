/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Configuration Header for VFS-Wind                          *
 * Provides Kokkos-based GPU portability layer                    *
 ******************************************************************/

#ifndef VFSWIND_GPU_CONFIG_HPP
#define VFSWIND_GPU_CONFIG_HPP

#ifdef ENABLE_GPU

#include <Kokkos_Core.hpp>

// ============================================================================
// Execution Space Aliases
// ============================================================================

// Default execution space (GPU if available, otherwise CPU)
using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

// Host execution space (always CPU)
using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemorySpace = Kokkos::HostSpace;

// Device execution space (same as default for GPU builds)
using DeviceExecutionSpace = DefaultExecutionSpace;
using DeviceMemorySpace = DefaultMemorySpace;

// ============================================================================
// View Type Aliases
// ============================================================================

// Scalar field (3D array of PetscReal/double)
template<typename Space = DeviceMemorySpace>
using ScalarFieldView = Kokkos::View<double***, Kokkos::LayoutRight, Space>;

// Vector field (3D array of 3-component vectors)
// Layout: [k][j][i][component] where component = 0,1,2 for x,y,z
template<typename Space = DeviceMemorySpace>
using VectorFieldView = Kokkos::View<double****, Kokkos::LayoutRight, Space>;

// 1D arrays for IBM/FSI data
template<typename T, typename Space = DeviceMemorySpace>
using Array1DView = Kokkos::View<T*, Space>;

// ============================================================================
// Memory Management Helpers
// ============================================================================

// Create a mirror view on the host
template<typename ViewType>
auto createHostMirror(const ViewType& device_view) {
    return Kokkos::create_mirror_view(device_view);
}

// Deep copy from host to device
template<typename DstView, typename SrcView>
void copyToDevice(DstView& dst, const SrcView& src) {
    Kokkos::deep_copy(dst, src);
}

// Deep copy from device to host
template<typename DstView, typename SrcView>
void copyToHost(DstView& dst, const SrcView& src) {
    Kokkos::deep_copy(dst, src);
}

// ============================================================================
// Parallel Execution Helpers
// ============================================================================

// 3D parallel loop policy (for structured grid loops)
using MDRangePolicy3D = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

// Create a 3D range policy
inline MDRangePolicy3D make3DPolicy(int zs, int ze, int ys, int ye, int xs, int xe) {
    return MDRangePolicy3D({zs, ys, xs}, {ze, ye, xe});
}

// ============================================================================
// Domain Information Structure
// ============================================================================

struct DomainInfo {
    // Local owned range (interior points)
    int xs, xe, ys, ye, zs, ze;

    // Local range with ghosts
    int gxs, gxe, gys, gye, gzs, gze;

    // Global dimensions
    int mx, my, mz;

    // Loop bounds (excluding boundaries)
    int lxs, lxe, lys, lye, lzs, lze;
};

// ============================================================================
// GPU Kernel Execution Wrapper
// ============================================================================

// Fence to ensure all GPU work is complete
inline void gpuFence() {
    Kokkos::fence();
}

// Check if running on GPU
inline bool isGPUEnabled() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    return true;
#else
    return false;
#endif
}

// Get backend name as string
inline const char* getBackendName() {
#if defined(KOKKOS_ENABLE_CUDA)
    return "CUDA";
#elif defined(KOKKOS_ENABLE_HIP)
    return "HIP";
#elif defined(KOKKOS_ENABLE_SYCL)
    return "SYCL";
#elif defined(KOKKOS_ENABLE_OPENMP)
    return "OpenMP";
#else
    return "Serial";
#endif
}

#else // !ENABLE_GPU

// ============================================================================
// Stub definitions when GPU is disabled
// ============================================================================

inline bool isGPUEnabled() { return false; }
inline const char* getBackendName() { return "None (CPU only)"; }
inline void gpuFence() { /* no-op */ }

#endif // ENABLE_GPU

#endif // VFSWIND_GPU_CONFIG_HPP
