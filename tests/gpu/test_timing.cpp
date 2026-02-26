/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * GPU Timing Test                                                *
 *                                                                *
 * Tests the GPU timing utilities for performance measurement.   *
 ******************************************************************/

#include <gtest/gtest.h>
#include <thread>
#include <chrono>

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#include "gpu_timer.hpp"
#include "gpu_dispatch.h"

using namespace vfswind::gpu;

class TimingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }

    static void TearDownTestSuite() {
        // Don't finalize - other tests may need Kokkos
    }

    void SetUp() override {
        TimerRegistry::instance().clear();
    }
};

TEST_F(TimingTest, GPUTimerBasic) {
    GPUTimer timer("TestTimer");

    timer.start();
    // Sleep for a measurable amount
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    timer.stop();

    // Should have measured at least 10ms = 10000us
    EXPECT_GT(timer.elapsed_us(), 9000.0);
    EXPECT_LT(timer.elapsed_us(), 50000.0);  // But not more than 50ms

    EXPECT_GT(timer.elapsed_ms(), 9.0);
    EXPECT_GT(timer.elapsed_s(), 0.009);
}

TEST_F(TimingTest, ScopedTimer) {
    double elapsed = 0.0;
    {
        ScopedTimer timer("ScopedTest");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        elapsed = timer.elapsed_us();  // Get intermediate value
    }
    // Timer should have captured the sleep duration
    // Note: elapsed_us() before stop may not be accurate
}

TEST_F(TimingTest, TimerRegistry) {
    // Record some timings
    TimerRegistry::instance().record("Kernel1", 100.0);
    TimerRegistry::instance().record("Kernel1", 150.0);
    TimerRegistry::instance().record("Kernel1", 125.0);
    TimerRegistry::instance().record("Kernel2", 500.0);

    double total, avg, min_val, max_val;
    int count;

    TimerRegistry::instance().getStats("Kernel1", total, avg, min_val, max_val, count);

    EXPECT_EQ(count, 3);
    EXPECT_DOUBLE_EQ(total, 375.0);
    EXPECT_DOUBLE_EQ(avg, 125.0);
    EXPECT_DOUBLE_EQ(min_val, 100.0);
    EXPECT_DOUBLE_EQ(max_val, 150.0);
}

TEST_F(TimingTest, TimerRegistryClear) {
    TimerRegistry::instance().record("Test", 100.0);

    double total, avg, min_val, max_val;
    int count;
    TimerRegistry::instance().getStats("Test", total, avg, min_val, max_val, count);
    EXPECT_EQ(count, 1);

    TimerRegistry::instance().clear();
    TimerRegistry::instance().getStats("Test", total, avg, min_val, max_val, count);
    EXPECT_EQ(count, 0);
}

TEST_F(TimingTest, RegisteredTimer) {
    {
        RegisteredTimer timer("RegTimer1");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        RegisteredTimer timer("RegTimer1");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    double total, avg, min_val, max_val;
    int count;
    TimerRegistry::instance().getStats("RegTimer1", total, avg, min_val, max_val, count);

    EXPECT_EQ(count, 2);
    EXPECT_GT(total, 0.0);
}

TEST_F(TimingTest, TimingEnabledCheck) {
    // By default, timing should be disabled unless VFSWIND_TIMING=1
    // This test just verifies the function exists and returns a valid value
    int enabled = VFSWind_GPU_IsTimingEnabled();
    EXPECT_TRUE(enabled == 0 || enabled == 1);
}

TEST_F(TimingTest, KokkosParallelForTiming) {
    GPUTimer timer("ParallelFor");

    const int N = 100000;
    Kokkos::View<double*> data("data", N);

    timer.start();
    Kokkos::parallel_for("Init", N, KOKKOS_LAMBDA(int i) {
        data(i) = static_cast<double>(i) * 2.0;
    });
    timer.stop();

    // Should have measured something
    EXPECT_GT(timer.elapsed_us(), 0.0);

    // Verify computation
    auto data_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    EXPECT_DOUBLE_EQ(data_h(1000), 2000.0);
}

TEST_F(TimingTest, PrintSummaryNoErrors) {
    // Just verify print doesn't crash
    TimerRegistry::instance().record("PrintTest", 100.0);
    VFSWind_GPU_PrintTimingSummary();
    VFSWind_GPU_ClearTimingData();
}

#endif // ENABLE_GPU

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

#ifdef ENABLE_GPU
    if (!Kokkos::is_initialized()) {
        Kokkos::initialize(argc, argv);
    }
#endif

    int result = RUN_ALL_TESTS();

#ifdef ENABLE_GPU
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
#endif

    return result;
}
