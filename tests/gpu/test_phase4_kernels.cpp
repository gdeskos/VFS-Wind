/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * VFS-Wind Phase 4 GPU Kernel Tests                              *
 * Level-Set and Turbulence Models                                *
 ******************************************************************/

#include <gtest/gtest.h>
#include <cmath>

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#include "gpu_config.hpp"
#include "kernels/turbulence_kernels.hpp"
#endif

namespace {

class Phase4KernelTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef ENABLE_GPU
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
#endif
    }
};

// ============================================================================
// WENO Scheme Tests
// ============================================================================

TEST_F(Phase4KernelTest, WENO3_SmoothFunction) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::levelset;

    // Test WENO3 on smooth data (should give accurate result)
    // Using f(x) = sin(x) at x = -0.5, 0.0, 0.5, 1.0
    double f0 = std::sin(-0.5);
    double f1 = std::sin(0.0);
    double f2 = std::sin(0.5);
    double f3 = std::sin(1.0);

    // Positive wave speed (upwind from left)
    double result_pos = WENOSchemes::weno3(f0, f1, f2, f3, 1.0);

    // Result should be bounded between f1 and f2
    EXPECT_GE(result_pos, f1);
    EXPECT_LE(result_pos, f2);

    // Negative wave speed (upwind from right) - stencil is mirrored
    double result_neg = WENOSchemes::weno3(f0, f1, f2, f3, -1.0);

    // Result should be bounded between f1 and f2
    EXPECT_GE(result_neg, f1);
    EXPECT_LE(result_neg, f2);

    // WENO should give smooth, monotonic results
    EXPECT_GT(result_pos, 0.0);
    EXPECT_GT(result_neg, 0.0);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, WENO5_SmoothFunction) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::levelset;

    // Test WENO5 on smooth quadratic data
    // f(x) = x^2 at x = -2, -1, 0, 1, 2, 3
    double f0 = 4.0;   // (-2)^2
    double f1 = 1.0;   // (-1)^2
    double f2 = 0.0;   // 0^2
    double f3 = 1.0;   // 1^2
    double f4 = 4.0;   // 2^2
    double f5 = 9.0;   // 3^2

    // Positive wave speed
    double result_pos = WENOSchemes::weno5(f0, f1, f2, f3, f4, f5, 1.0);

    // WENO5 should be very accurate for smooth polynomial data
    // Expect value near f(0.5) = 0.25
    EXPECT_NEAR(result_pos, 0.25, 0.1);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, WENO3_Discontinuity) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::levelset;

    // Test WENO3 with discontinuous data (step function)
    // f = 0 for x < 0, f = 1 for x >= 0
    double f0 = 0.0;
    double f1 = 0.0;
    double f2 = 1.0;
    double f3 = 1.0;

    // Should be non-oscillatory
    double result = WENOSchemes::weno3(f0, f1, f2, f3, 1.0);

    // Result should be bounded between 0 and 1
    EXPECT_GE(result, 0.0);
    EXPECT_LE(result, 1.0);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

// ============================================================================
// Level-Set Property Tests
// ============================================================================

TEST_F(Phase4KernelTest, HeavisideFunction) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::levelset;

    double epsilon = 0.1;

    // Test Heaviside at various locations
    // Far negative: H should be 0
    EXPECT_NEAR(LevelsetPropertiesKernel::heaviside(-1.0, epsilon), 0.0, 1e-10);

    // Far positive: H should be 1
    EXPECT_NEAR(LevelsetPropertiesKernel::heaviside(1.0, epsilon), 1.0, 1e-10);

    // At interface: H should be 0.5
    EXPECT_NEAR(LevelsetPropertiesKernel::heaviside(0.0, epsilon), 0.5, 1e-10);

    // Smoothly varying in transition zone
    double H_minus = LevelsetPropertiesKernel::heaviside(-0.05, epsilon);
    double H_plus = LevelsetPropertiesKernel::heaviside(0.05, epsilon);
    EXPECT_LT(H_minus, 0.5);
    EXPECT_GT(H_plus, 0.5);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, DiracDelta) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::levelset;

    double epsilon = 0.1;

    // Delta should be zero outside interface region
    EXPECT_NEAR(LevelsetPropertiesKernel::diracDelta(-0.2, epsilon), 0.0, 1e-10);
    EXPECT_NEAR(LevelsetPropertiesKernel::diracDelta(0.2, epsilon), 0.0, 1e-10);

    // Delta should be maximum at interface
    double delta_0 = LevelsetPropertiesKernel::diracDelta(0.0, epsilon);
    double delta_half = LevelsetPropertiesKernel::diracDelta(0.05, epsilon);
    EXPECT_GT(delta_0, delta_half);
    EXPECT_GT(delta_0, 0.0);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, DensityComputation) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::levelset;

    const int N = 10;
    ScalarView3D<DeviceMemorySpace> phi("phi", N, N, N);
    ScalarView3D<DeviceMemorySpace> rho("rho", N, N, N);

    double rho1 = 1.0;    // Fluid 1 density
    double rho2 = 1000.0; // Fluid 2 density
    double epsilon = 0.5;

    // Initialize level-set: positive in half, negative in other half
    Kokkos::parallel_for("InitPhi",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            phi(k, j, i) = (double)j - N/2.0;  // Interface at j = N/2
        }
    );
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;
    domain.lxs = 1; domain.lxe = N-1;
    domain.lys = 1; domain.lye = N-1;
    domain.lzs = 1; domain.lze = N-1;

    LevelsetPropertiesKernel::computeDensity(phi, rho, rho1, rho2, epsilon, domain);

    // Copy result to host and verify
    auto rho_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);

    // Check regions
    EXPECT_NEAR(rho_h(1, 1, 1), rho1, 10.0);      // Should be near rho1 (phi < 0)
    EXPECT_NEAR(rho_h(1, N-2, 1), rho2, 10.0);   // Should be near rho2 (phi > 0)
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

// ============================================================================
// LES Tests
// ============================================================================

TEST_F(Phase4KernelTest, StrainRateMagnitude) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::les;

    // Simple shear flow: u = y, v = 0, w = 0
    // du/dy = 1, all other gradients = 0
    // S_xy = 0.5 * (du/dy + dv/dx) = 0.5
    // S_ij S_ij = 2 * S_xy^2 = 2 * 0.25 = 0.5
    // |S| = sqrt(2 * S_ij S_ij) = sqrt(2 * 0.5) = 1.0

    double Sabs = StrainRateTensor::computeMagnitude(
        0.0, 1.0, 0.0,  // du/dx, du/dy, du/dz
        0.0, 0.0, 0.0,  // dv/dx, dv/dy, dv/dz
        0.0, 0.0, 0.0   // dw/dx, dw/dy, dw/dz
    );

    EXPECT_NEAR(Sabs, 1.0, 1e-10);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, SmagorinskyConstantCs) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::les;

    const int N = 8;
    VectorView3D<DeviceMemorySpace> ucat("ucat", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);
    VectorView3D<DeviceMemorySpace> csi("csi", N, N, N);
    VectorView3D<DeviceMemorySpace> eta("eta", N, N, N);
    VectorView3D<DeviceMemorySpace> zet("zet", N, N, N);
    ScalarView3D<DeviceMemorySpace> aj("aj", N, N, N);
    ScalarView3D<DeviceMemorySpace> nu_t("nu_t", N, N, N);

    // Initialize simple uniform grid (Cartesian)
    Kokkos::parallel_for("InitGrid",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Simple shear flow: u = j
            ucat(k, j, i).x = (double)j;
            ucat(k, j, i).y = 0.0;
            ucat(k, j, i).z = 0.0;

            nvert(k, j, i) = 0.0;  // No solid cells

            // Cartesian grid metrics
            csi(k, j, i).x = 1.0; csi(k, j, i).y = 0.0; csi(k, j, i).z = 0.0;
            eta(k, j, i).x = 0.0; eta(k, j, i).y = 1.0; eta(k, j, i).z = 0.0;
            zet(k, j, i).x = 0.0; zet(k, j, i).y = 0.0; zet(k, j, i).z = 1.0;

            aj(k, j, i) = 1.0;  // Unit Jacobian
        }
    );
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;
    domain.lxs = 2; domain.lxe = N-2;
    domain.lys = 2; domain.lye = N-2;
    domain.lzs = 2; domain.lze = N-2;

    double Cs = 0.1;  // Smagorinsky constant
    SmagorinskyKernel::computeEddyViscosityConstantCs(
        ucat, nvert, csi, eta, zet, aj, Cs, nu_t, domain
    );

    // Copy result to host
    auto nu_t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nu_t);

    // Check that eddy viscosity is positive in interior
    for (int k = 2; k < N-2; ++k) {
        for (int j = 2; j < N-2; ++j) {
            for (int i = 2; i < N-2; ++i) {
                EXPECT_GE(nu_t_h(k, j, i), 0.0) << "at (" << k << "," << j << "," << i << ")";
            }
        }
    }

    // Expected: nu_t = Cs * filter^2 * |S|
    // filter = 1.0 (unit Jacobian), |S| = 1.0 (for du/dy = 1), Cs = 0.1
    // nu_t ≈ 0.1 * 1.0 * 1.0 = 0.1
    double expected_nu_t = Cs * 1.0 * 1.0;
    EXPECT_NEAR(nu_t_h(3, 3, 3), expected_nu_t, 0.02);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, VanDriestDamping) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::les;

    // van Driest damping: D = 1 - exp(-y+/A+)
    // A+ = 26

    // At y+ = 0: D = 0
    EXPECT_NEAR(WallDampingKernel::vanDriestDamping(0.0), 0.0, 1e-10);

    // At y+ = 26: D ≈ 1 - e^-1 ≈ 0.632
    EXPECT_NEAR(WallDampingKernel::vanDriestDamping(26.0), 1.0 - std::exp(-1.0), 0.01);

    // At large y+: D → 1
    EXPECT_NEAR(WallDampingKernel::vanDriestDamping(1000.0), 1.0, 0.001);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

// ============================================================================
// RANS Tests
// ============================================================================

TEST_F(Phase4KernelTest, KOmegaConstants) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::rans;

    // Verify standard k-omega constants
    EXPECT_NEAR(KOmegaConstants::BETA_STAR, 0.09, 1e-10);
    EXPECT_NEAR(KOmegaConstants::BETA1, 0.075, 1e-10);
    EXPECT_NEAR(KOmegaConstants::BETA2, 0.0828, 1e-10);
    EXPECT_NEAR(KOmegaConstants::A1, 0.31, 1e-10);
    EXPECT_NEAR(KOmegaConstants::ALPHA1, 5.0/9.0, 1e-10);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, WallOmegaBC) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::rans;

    // omega_wall = 6 / (beta1 * Re * y^2)
    double Re = 1000.0;
    double y = 0.01;

    double omega_wall = KOmegaConstants::wallOmega(Re, y);

    // Expected: 6 / (0.075 * 1000 * 0.0001) = 6 / 7.5 = 0.8
    double expected = 6.0 / (KOmegaConstants::BETA1 * Re * y * y);
    EXPECT_NEAR(omega_wall, expected, 1e-6);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, AlphaBetaStarLowRe) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::rans;

    double Re_nu = 1000.0;
    double K = 0.01;
    double O = 10.0;
    double alpha, alpha_star, beta_star;

    // Low-Re model
    KOmegaKernel::getAlphaBetaStar(Re_nu, K, O, RANSModel::WILCOX_LOW_RE,
                                    alpha, alpha_star, beta_star);

    // Should return reasonable values
    EXPECT_GT(alpha, 0.0);
    EXPECT_GT(alpha_star, 0.0);
    EXPECT_GT(beta_star, 0.0);
    EXPECT_LE(alpha_star, 1.0);
    EXPECT_LE(beta_star, 0.1);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, AlphaBetaStarHighRe) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::rans;

    double Re_nu = 1000.0;
    double K = 0.01;
    double O = 10.0;
    double alpha, alpha_star, beta_star;

    // High-Re model (should return standard values)
    KOmegaKernel::getAlphaBetaStar(Re_nu, K, O, RANSModel::WILCOX_HIGH_RE,
                                    alpha, alpha_star, beta_star);

    EXPECT_NEAR(alpha, KOmegaConstants::ALPHA1, 1e-10);
    EXPECT_NEAR(alpha_star, 1.0, 1e-10);
    EXPECT_NEAR(beta_star, 0.09, 1e-10);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, TurbulentViscosityKOmega) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::rans;

    const int N = 6;
    ScalarView3D<DeviceMemorySpace> k("k", N, N, N);
    ScalarView3D<DeviceMemorySpace> omega("omega", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);
    ScalarView3D<DeviceMemorySpace> F1("F1", N, N, N);
    ScalarView3D<DeviceMemorySpace> nu_t("nu_t", N, N, N);

    // Initialize with uniform turbulence field
    double k_val = 0.1;
    double omega_val = 100.0;

    Kokkos::parallel_for("InitKOmega",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(int kk, int j, int i) {
            k(kk, j, i) = k_val;
            omega(kk, j, i) = omega_val;
            nvert(kk, j, i) = 0.0;
            F1(kk, j, i) = 1.0;
        }
    );
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;
    domain.lxs = 1; domain.lxe = N-1;
    domain.lys = 1; domain.lye = N-1;
    domain.lzs = 1; domain.lze = N-1;

    double Re_nu = 1000.0;

    KOmegaKernel::computeTurbulentViscosity(
        k, omega, nvert, F1, nu_t, Re_nu, RANSModel::WILCOX_HIGH_RE, domain
    );

    // Copy result
    auto nu_t_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nu_t);

    // Expected: nu_t = alpha_star * k / omega = 1.0 * 0.1 / 100 = 0.001
    double expected = 1.0 * k_val / omega_val;
    EXPECT_NEAR(nu_t_h(2, 2, 2), expected, 1e-6);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

// ============================================================================
// Unified Interface Tests
// ============================================================================

TEST_F(Phase4KernelTest, TurbulenceModelEnum) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;

    // Verify enum values
    EXPECT_EQ(static_cast<int>(TurbulenceModel::NONE), 0);
    EXPECT_EQ(static_cast<int>(TurbulenceModel::LES_STATIC_SMAGORINSKY), 1);
    EXPECT_EQ(static_cast<int>(TurbulenceModel::LES_DYNAMIC_SMAGORINSKY), 2);
    EXPECT_EQ(static_cast<int>(TurbulenceModel::RANS_KOMEGA_WILCOX_LOW_RE), 3);
    EXPECT_EQ(static_cast<int>(TurbulenceModel::RANS_KOMEGA_WILCOX_HIGH_RE), 4);
    EXPECT_EQ(static_cast<int>(TurbulenceModel::RANS_KOMEGA_SST), 5);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

TEST_F(Phase4KernelTest, PrintPhase4Info) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;

    // Should not crash
    testing::internal::CaptureStdout();
    printPhase4Info();
    std::string output = testing::internal::GetCapturedStdout();

    // Check for expected content
    EXPECT_NE(output.find("Phase 4"), std::string::npos);
    EXPECT_NE(output.find("Level-Set"), std::string::npos);
    EXPECT_NE(output.find("LES"), std::string::npos);
    EXPECT_NE(output.find("RANS"), std::string::npos);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(Phase4KernelTest, LevelsetAdvectionPerformance) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::levelset;

    // Moderate grid size for performance test
    const int N = 32;
    ScalarView3D<DeviceMemorySpace> phi("phi", N, N, N);
    VectorView3D<DeviceMemorySpace> ucont("ucont", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);
    ScalarView3D<DeviceMemorySpace> aj("aj", N, N, N);
    ScalarView3D<DeviceMemorySpace> rhs("rhs", N, N, N);

    // Initialize
    Kokkos::parallel_for("Init",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            phi(k, j, i) = std::sin(0.1 * i) * std::cos(0.1 * j);
            ucont(k, j, i).x = 1.0;
            ucont(k, j, i).y = 0.5;
            ucont(k, j, i).z = 0.25;
            nvert(k, j, i) = 0.0;
            aj(k, j, i) = 1.0;
        }
    );
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;
    domain.lxs = 3; domain.lxe = N-3;
    domain.lys = 3; domain.lye = N-3;
    domain.lzs = 3; domain.lze = N-3;

    // Warm up
    LevelsetAdvectionKernel::computeAdvectionRHS(
        phi, ucont, nvert, aj, rhs, domain,
        LevelsetAdvectionKernel::Scheme::WENO3
    );

    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 100;
    for (int iter = 0; iter < iterations; ++iter) {
        LevelsetAdvectionKernel::computeAdvectionRHS(
            phi, ucont, nvert, aj, rhs, domain,
            LevelsetAdvectionKernel::Scheme::WENO3
        );
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_us = duration.count() / static_cast<double>(iterations);

    std::cout << "Level-set advection (" << N << "^3 grid): "
              << avg_time_us << " us/iteration" << std::endl;

    // Just ensure it completes without error
    EXPECT_GT(avg_time_us, 0.0);
#else
    GTEST_SKIP() << "GPU support not enabled";
#endif
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

#ifdef ENABLE_GPU
    Kokkos::initialize(argc, argv);
#endif

    int result = RUN_ALL_TESTS();

#ifdef ENABLE_GPU
    Kokkos::finalize();
#endif

    return result;
}
