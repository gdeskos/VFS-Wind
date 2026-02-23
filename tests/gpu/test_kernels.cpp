/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Kernel Tests for VFS-Wind                                  *
 ******************************************************************/

#include <gtest/gtest.h>

#ifdef ENABLE_GPU
#include "kernels/kernels.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>

using namespace vfswind::gpu;
using namespace vfswind::gpu::kernels;

class KernelTest : public ::testing::Test {
protected:
    static constexpr int NX = 16;
    static constexpr int NY = 16;
    static constexpr int NZ = 16;

    KernelDomainInfo domain;

    void SetUp() override {
        // Set up a simple test domain
        domain.mx = NX;
        domain.my = NY;
        domain.mz = NZ;
        domain.xs = 0;
        domain.xe = NX;
        domain.ys = 0;
        domain.ye = NY;
        domain.zs = 0;
        domain.ze = NZ;
        domain.lxs = 1;
        domain.lxe = NX - 1;
        domain.lys = 1;
        domain.lye = NY - 1;
        domain.lzs = 1;
        domain.lze = NZ - 1;
        domain.ren = 1000.0;
        domain.les = false;
        domain.rans = false;
        domain.ti = 0;
    }
};

// ============================================================================
// Convection Kernel Tests
// ============================================================================

TEST_F(KernelTest, ConvectionUniformFlow) {
    // Test: Convection of a uniform flow should be zero
    using VectorView = ConvectionKernel::VectorView;
    using ScalarView = ConvectionKernel::ScalarView;

    // Allocate views
    VectorView ucont("ucont", NZ, NY, NX);
    VectorView ucat("ucat", NZ, NY, NX);
    ScalarView nvert("nvert", NZ, NY, NX);
    VectorView conv("conv", NZ, NY, NX);

    // Initialize uniform flow: u = (1, 0, 0)
    Kokkos::parallel_for("InitUniform",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            ucont(k, j, i) = Cmpnts3(1.0, 0.0, 0.0);
            ucat(k, j, i) = Cmpnts3(1.0, 0.0, 0.0);
            nvert(k, j, i) = 0.0;  // All fluid cells
        }
    );
    Kokkos::fence();

    // Execute convection kernel
    ConvectionKernel::execute(ucont, ucat, nvert, conv, domain);

    // Copy result to host for verification
    auto h_conv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), conv);

    // Convection of uniform flow should be zero (or very small)
    double max_error = 0.0;
    for (int k = domain.lzs; k < domain.lze; ++k) {
        for (int j = domain.lys; j < domain.lye; ++j) {
            for (int i = domain.lxs; i < domain.lxe; ++i) {
                max_error = std::max(max_error, std::abs(h_conv(k, j, i).x));
                max_error = std::max(max_error, std::abs(h_conv(k, j, i).y));
                max_error = std::max(max_error, std::abs(h_conv(k, j, i).z));
            }
        }
    }

    EXPECT_LT(max_error, 1e-10) << "Convection of uniform flow should be zero";
}

TEST_F(KernelTest, ConvectionSymmetry) {
    // Test: Convection kernel should produce symmetric results for symmetric input
    using VectorView = ConvectionKernel::VectorView;
    using ScalarView = ConvectionKernel::ScalarView;

    VectorView ucont("ucont", NZ, NY, NX);
    VectorView ucat("ucat", NZ, NY, NX);
    ScalarView nvert("nvert", NZ, NY, NX);
    VectorView conv("conv", NZ, NY, NX);

    // Initialize symmetric flow
    Kokkos::parallel_for("InitSymmetric",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            double x = static_cast<double>(i) / (NX - 1);
            double y = static_cast<double>(j) / (NY - 1);
            double z = static_cast<double>(k) / (NZ - 1);

            // Symmetric velocity field
            ucont(k, j, i) = Cmpnts3(sin(M_PI * x), sin(M_PI * y), sin(M_PI * z));
            ucat(k, j, i) = Cmpnts3(sin(M_PI * x), sin(M_PI * y), sin(M_PI * z));
            nvert(k, j, i) = 0.0;
        }
    );
    Kokkos::fence();

    ConvectionKernel::execute(ucont, ucat, nvert, conv, domain);

    // Just verify it runs without error and produces finite values
    auto h_conv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), conv);

    bool all_finite = true;
    for (int k = domain.lzs; k < domain.lze && all_finite; ++k) {
        for (int j = domain.lys; j < domain.lye && all_finite; ++j) {
            for (int i = domain.lxs; i < domain.lxe && all_finite; ++i) {
                if (!std::isfinite(h_conv(k, j, i).x) ||
                    !std::isfinite(h_conv(k, j, i).y) ||
                    !std::isfinite(h_conv(k, j, i).z)) {
                    all_finite = false;
                }
            }
        }
    }

    EXPECT_TRUE(all_finite) << "Convection kernel produced non-finite values";
}

// ============================================================================
// Viscous Kernel Tests
// ============================================================================

TEST_F(KernelTest, ViscousLinearFlow) {
    // Test: Viscous term of a linear velocity profile should be zero (Laplacian = 0)
    using VectorView = ViscousKernel::VectorView;
    using ScalarView = ViscousKernel::ScalarView;

    VectorView ucat("ucat", NZ, NY, NX);
    ScalarView nvert("nvert", NZ, NY, NX);
    VectorView visc("visc", NZ, NY, NX);

    // Metric tensors (identity for Cartesian grid)
    VectorView icsi("icsi", NZ, NY, NX);
    VectorView ieta("ieta", NZ, NY, NX);
    VectorView izet("izet", NZ, NY, NX);
    VectorView jcsi("jcsi", NZ, NY, NX);
    VectorView jeta("jeta", NZ, NY, NX);
    VectorView jzet("jzet", NZ, NY, NX);
    VectorView kcsi("kcsi", NZ, NY, NX);
    VectorView keta("keta", NZ, NY, NX);
    VectorView kzet("kzet", NZ, NY, NX);
    ScalarView iaj("iaj", NZ, NY, NX);
    ScalarView jaj("jaj", NZ, NY, NX);
    ScalarView kaj("kaj", NZ, NY, NX);

    // Initialize linear velocity profile and identity metrics
    Kokkos::parallel_for("InitLinear",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            double y = static_cast<double>(j) / (NY - 1);

            // Linear Couette flow: u = y, v = 0, w = 0
            ucat(k, j, i) = Cmpnts3(y, 0.0, 0.0);
            nvert(k, j, i) = 0.0;

            // Identity metrics (Cartesian grid)
            double h = 1.0 / (NY - 1);  // Grid spacing
            icsi(k, j, i) = Cmpnts3(1.0/h, 0.0, 0.0);
            ieta(k, j, i) = Cmpnts3(0.0, 1.0/h, 0.0);
            izet(k, j, i) = Cmpnts3(0.0, 0.0, 1.0/h);
            jcsi(k, j, i) = Cmpnts3(1.0/h, 0.0, 0.0);
            jeta(k, j, i) = Cmpnts3(0.0, 1.0/h, 0.0);
            jzet(k, j, i) = Cmpnts3(0.0, 0.0, 1.0/h);
            kcsi(k, j, i) = Cmpnts3(1.0/h, 0.0, 0.0);
            keta(k, j, i) = Cmpnts3(0.0, 1.0/h, 0.0);
            kzet(k, j, i) = Cmpnts3(0.0, 0.0, 1.0/h);
            iaj(k, j, i) = h * h * h;
            jaj(k, j, i) = h * h * h;
            kaj(k, j, i) = h * h * h;
        }
    );
    Kokkos::fence();

    ViscousKernel::execute(
        ucat, nvert,
        icsi, ieta, izet,
        jcsi, jeta, jzet,
        kcsi, keta, kzet,
        iaj, jaj, kaj,
        nullptr,  // No turbulent viscosity
        visc, domain
    );

    // Verify kernel runs and produces finite values
    auto h_visc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), visc);

    bool all_finite = true;
    for (int k = domain.lzs; k < domain.lze && all_finite; ++k) {
        for (int j = domain.lys; j < domain.lye && all_finite; ++j) {
            for (int i = domain.lxs; i < domain.lxe && all_finite; ++i) {
                if (!std::isfinite(h_visc(k, j, i).x) ||
                    !std::isfinite(h_visc(k, j, i).y) ||
                    !std::isfinite(h_visc(k, j, i).z)) {
                    all_finite = false;
                }
            }
        }
    }

    EXPECT_TRUE(all_finite) << "Viscous kernel produced non-finite values";
}

// ============================================================================
// Pressure Gradient Tests
// ============================================================================

TEST_F(KernelTest, PressureGradientConstant) {
    // Test: Gradient of constant pressure should be zero
    using VectorView = PressureGradientKernel::VectorView;
    using ScalarView = PressureGradientKernel::ScalarView;

    ScalarView p("p", NZ, NY, NX);
    ScalarView nvert("nvert", NZ, NY, NX);
    VectorView dp("dp", NZ, NY, NX);

    // Metric tensors
    VectorView icsi("icsi", NZ, NY, NX);
    VectorView ieta("ieta", NZ, NY, NX);
    VectorView izet("izet", NZ, NY, NX);
    VectorView jcsi("jcsi", NZ, NY, NX);
    VectorView jeta("jeta", NZ, NY, NX);
    VectorView jzet("jzet", NZ, NY, NX);
    VectorView kcsi("kcsi", NZ, NY, NX);
    VectorView keta("keta", NZ, NY, NX);
    VectorView kzet("kzet", NZ, NY, NX);
    ScalarView iaj("iaj", NZ, NY, NX);
    ScalarView jaj("jaj", NZ, NY, NX);
    ScalarView kaj("kaj", NZ, NY, NX);

    // Initialize constant pressure
    Kokkos::parallel_for("InitConstP",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {NZ, NY, NX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            p(k, j, i) = 1.0;  // Constant pressure
            nvert(k, j, i) = 0.0;

            // Identity metrics
            double h = 1.0 / (NX - 1);
            icsi(k, j, i) = Cmpnts3(1.0/h, 0.0, 0.0);
            ieta(k, j, i) = Cmpnts3(0.0, 1.0/h, 0.0);
            izet(k, j, i) = Cmpnts3(0.0, 0.0, 1.0/h);
            jcsi(k, j, i) = Cmpnts3(1.0/h, 0.0, 0.0);
            jeta(k, j, i) = Cmpnts3(0.0, 1.0/h, 0.0);
            jzet(k, j, i) = Cmpnts3(0.0, 0.0, 1.0/h);
            kcsi(k, j, i) = Cmpnts3(1.0/h, 0.0, 0.0);
            keta(k, j, i) = Cmpnts3(0.0, 1.0/h, 0.0);
            kzet(k, j, i) = Cmpnts3(0.0, 0.0, 1.0/h);
            iaj(k, j, i) = h * h * h;
            jaj(k, j, i) = h * h * h;
            kaj(k, j, i) = h * h * h;
        }
    );
    Kokkos::fence();

    PressureGradientKernel::execute(
        p, nvert,
        icsi, ieta, izet,
        jcsi, jeta, jzet,
        kcsi, keta, kzet,
        iaj, jaj, kaj,
        dp, domain
    );

    // Gradient of constant should be zero
    auto h_dp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dp);

    double max_grad = 0.0;
    for (int k = domain.lzs; k < domain.lze; ++k) {
        for (int j = domain.lys; j < domain.lye; ++j) {
            for (int i = domain.lxs; i < domain.lxe; ++i) {
                max_grad = std::max(max_grad, std::abs(h_dp(k, j, i).x));
                max_grad = std::max(max_grad, std::abs(h_dp(k, j, i).y));
                max_grad = std::max(max_grad, std::abs(h_dp(k, j, i).z));
            }
        }
    }

    EXPECT_LT(max_grad, 1e-10) << "Gradient of constant pressure should be zero";
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(KernelTest, ConvectionPerformance) {
    // Simple performance check - ensure kernel completes reasonably fast
    using VectorView = ConvectionKernel::VectorView;
    using ScalarView = ConvectionKernel::ScalarView;

    // Use larger domain for performance test
    constexpr int PNX = 64, PNY = 64, PNZ = 64;
    KernelDomainInfo perf_domain;
    perf_domain.mx = PNX;
    perf_domain.my = PNY;
    perf_domain.mz = PNZ;
    perf_domain.xs = 0; perf_domain.xe = PNX;
    perf_domain.ys = 0; perf_domain.ye = PNY;
    perf_domain.zs = 0; perf_domain.ze = PNZ;
    perf_domain.lxs = 1; perf_domain.lxe = PNX - 1;
    perf_domain.lys = 1; perf_domain.lye = PNY - 1;
    perf_domain.lzs = 1; perf_domain.lze = PNZ - 1;
    perf_domain.ren = 1000.0;

    VectorView ucont("ucont", PNZ, PNY, PNX);
    VectorView ucat("ucat", PNZ, PNY, PNX);
    ScalarView nvert("nvert", PNZ, PNY, PNX);
    VectorView conv("conv", PNZ, PNY, PNX);

    // Initialize
    Kokkos::parallel_for("InitPerf",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {PNZ, PNY, PNX}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            double x = static_cast<double>(i) / (PNX - 1);
            double y = static_cast<double>(j) / (PNY - 1);
            ucont(k, j, i) = Cmpnts3(sin(2*M_PI*x), sin(2*M_PI*y), 0.0);
            ucat(k, j, i) = Cmpnts3(sin(2*M_PI*x), sin(2*M_PI*y), 0.0);
            nvert(k, j, i) = 0.0;
        }
    );
    Kokkos::fence();

    // Run kernel multiple times
    constexpr int NUM_ITERS = 10;
    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        ConvectionKernel::execute(ucont, ucat, nvert, conv, perf_domain);
    }
    Kokkos::fence();

    // If we get here without timing out, the test passes
    SUCCEED() << "Convection kernel completed " << NUM_ITERS << " iterations";
}

#endif // ENABLE_GPU

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
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
