/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Smoke Tests for VFS-Wind                                   *
 ******************************************************************/

#include <gtest/gtest.h>

#ifdef ENABLE_GPU
#include "gpu_config.hpp"
#include <Kokkos_Core.hpp>
#endif

#include "kokkos_init.h"

class GPUSmokeTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Kokkos should already be initialized by main()
    }

    static void TearDownTestSuite() {
        // Kokkos will be finalized by main()
    }
};

// ============================================================================
// Basic GPU Tests
// ============================================================================

TEST_F(GPUSmokeTest, GPUEnabled) {
    // This test verifies the GPU backend is properly configured
    // Note: VFSWind_GPU_IsEnabled returns true only for CUDA/HIP/SYCL,
    // not for OpenMP (which is a CPU threading backend)
    int enabled = VFSWind_GPU_IsEnabled();

#ifdef ENABLE_GPU
    #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
        EXPECT_EQ(enabled, 1) << "GPU should be enabled for CUDA/HIP/SYCL backends";
    #else
        // OpenMP/Serial backends are CPU-based
        EXPECT_EQ(enabled, 0) << "GPU flag is false for OpenMP/Serial backends (correct)";
    #endif
#else
    EXPECT_EQ(enabled, 0) << "GPU should be disabled when compiled without ENABLE_GPU";
#endif
}

TEST_F(GPUSmokeTest, BackendName) {
    const char* backend = VFSWind_GPU_GetBackendName();
    ASSERT_NE(backend, nullptr);

#ifdef ENABLE_GPU
    // Should be one of the known backends
    std::string name(backend);
    EXPECT_TRUE(name == "CUDA" || name == "HIP" || name == "SYCL" ||
                name == "OpenMP" || name == "Serial")
        << "Unknown backend: " << name;
#else
    EXPECT_STREQ(backend, "None (CPU only)");
#endif
}

TEST_F(GPUSmokeTest, BuiltInSmokeTest) {
    int result = VFSWind_GPU_SmokeTest();
    EXPECT_EQ(result, 0) << "Built-in GPU smoke test failed";
}

#ifdef ENABLE_GPU

// ============================================================================
// Kokkos-specific tests (only compiled when GPU is enabled)
// ============================================================================

TEST_F(GPUSmokeTest, KokkosInitialized) {
    EXPECT_TRUE(Kokkos::is_initialized()) << "Kokkos should be initialized";
}

TEST_F(GPUSmokeTest, ParallelFor1D) {
    const int N = 10000;

    // Create device view
    Kokkos::View<double*, DeviceMemorySpace> d_result("result", N);

    // Run parallel_for
    Kokkos::parallel_for("test_1d",
        Kokkos::RangePolicy<DefaultExecutionSpace>(0, N),
        KOKKOS_LAMBDA(int i) {
            d_result(i) = i * 2.0;
        }
    );

    // Copy to host and verify
    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_result);

    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(h_result(i), i * 2.0) << "Mismatch at i=" << i;
    }
}

TEST_F(GPUSmokeTest, ParallelReduce) {
    const int N = 100000;

    // Create device view
    Kokkos::View<double*, DeviceMemorySpace> d_array("array", N);

    // Initialize
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<DefaultExecutionSpace>(0, N),
        KOKKOS_LAMBDA(int i) {
            d_array(i) = 1.0;  // All ones
        }
    );

    // Sum reduction
    double sum = 0.0;
    Kokkos::parallel_reduce("sum",
        Kokkos::RangePolicy<DefaultExecutionSpace>(0, N),
        KOKKOS_LAMBDA(int i, double& local_sum) {
            local_sum += d_array(i);
        },
        sum
    );

    EXPECT_DOUBLE_EQ(sum, static_cast<double>(N)) << "Sum should equal N";
}

TEST_F(GPUSmokeTest, ParallelFor3D) {
    const int NX = 32, NY = 32, NZ = 32;

    // Create 3D device view
    Kokkos::View<double***, DeviceMemorySpace> d_field("field", NZ, NY, NX);

    // Run 3D parallel_for
    Kokkos::parallel_for("test_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            d_field(k, j, i) = k * 100.0 + j * 10.0 + i;
        }
    );

    // Copy to host and verify
    auto h_field = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_field);

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                double expected = k * 100.0 + j * 10.0 + i;
                EXPECT_DOUBLE_EQ(h_field(k, j, i), expected)
                    << "Mismatch at (" << k << "," << j << "," << i << ")";
            }
        }
    }
}

TEST_F(GPUSmokeTest, MemoryAllocation) {
    // Test large allocation (100 MB)
    const size_t N = 100 * 1024 * 1024 / sizeof(double);

    // Note: Using a lambda to avoid macro expansion issues with template commas
    bool alloc_succeeded = true;
    try {
        using LargeView = Kokkos::View<double*, DeviceMemorySpace>;
        LargeView d_large("large", N);
        Kokkos::fence();
    } catch (...) {
        alloc_succeeded = false;
    }
    EXPECT_TRUE(alloc_succeeded) << "Failed to allocate 100 MB on GPU";
}

TEST_F(GPUSmokeTest, VectorFieldView) {
    // Test the VectorFieldView type from gpu_config.hpp
    const int NX = 16, NY = 16, NZ = 16;

    VectorFieldView<DeviceMemorySpace> d_velocity("velocity", NZ, NY, NX, 3);

    // Initialize velocity components
    Kokkos::parallel_for("init_velocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            d_velocity(k, j, i, 0) = 1.0;  // u
            d_velocity(k, j, i, 1) = 2.0;  // v
            d_velocity(k, j, i, 2) = 3.0;  // w
        }
    );

    // Verify
    auto h_velocity = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_velocity);

    for (int k = 0; k < NZ; ++k) {
        for (int j = 0; j < NY; ++j) {
            for (int i = 0; i < NX; ++i) {
                EXPECT_DOUBLE_EQ(h_velocity(k, j, i, 0), 1.0);
                EXPECT_DOUBLE_EQ(h_velocity(k, j, i, 1), 2.0);
                EXPECT_DOUBLE_EQ(h_velocity(k, j, i, 2), 3.0);
            }
        }
    }
}

TEST_F(GPUSmokeTest, Stencil3Point) {
    // Test a simple 3-point stencil (like what we'd use in convection)
    const int N = 100;

    Kokkos::View<double*, DeviceMemorySpace> d_in("in", N);
    Kokkos::View<double*, DeviceMemorySpace> d_out("out", N);

    // Initialize input: f(x) = x^2
    Kokkos::parallel_for("init",
        Kokkos::RangePolicy<DefaultExecutionSpace>(0, N),
        KOKKOS_LAMBDA(int i) {
            d_in(i) = static_cast<double>(i * i);
        }
    );

    // Apply stencil: out[i] = (in[i-1] + in[i] + in[i+1]) / 3
    Kokkos::parallel_for("stencil",
        Kokkos::RangePolicy<DefaultExecutionSpace>(1, N-1),
        KOKKOS_LAMBDA(int i) {
            d_out(i) = (d_in(i-1) + d_in(i) + d_in(i+1)) / 3.0;
        }
    );

    // Verify a few points
    auto h_in = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_in);
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out);

    for (int i = 1; i < N-1; ++i) {
        double expected = (h_in(i-1) + h_in(i) + h_in(i+1)) / 3.0;
        EXPECT_NEAR(h_out(i), expected, 1e-10) << "Stencil error at i=" << i;
    }
}

#endif // ENABLE_GPU

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize GPU/Kokkos
    VFSWind_GPU_Initialize(&argc, &argv);

    int result = RUN_ALL_TESTS();

    // Finalize GPU/Kokkos
    VFSWind_GPU_Finalize();

    return result;
}
