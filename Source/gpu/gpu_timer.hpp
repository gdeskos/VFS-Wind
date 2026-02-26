/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * GPU Timer Utilities                                            *
 *                                                                *
 * PURPOSE:                                                       *
 * Provides wallclock timing for GPU kernel performance analysis. *
 * Timing can be enabled via environment variable VFSWIND_TIMING. *
 *                                                                *
 * USAGE:                                                         *
 *   GPUTimer timer("KernelName");                                *
 *   timer.start();                                               *
 *   // ... kernel execution ...                                  *
 *   timer.stop();                                                *
 *   timer.print();  // Prints if VFSWIND_TIMING=1                *
 *                                                                *
 * Or use the RAII helper:                                        *
 *   {                                                            *
 *       ScopedTimer t("KernelName");                             *
 *       // ... kernel execution ...                              *
 *   } // Automatically prints on scope exit                      *
 *                                                                *
 ******************************************************************/

#ifndef VFSWIND_GPU_TIMER_HPP
#define VFSWIND_GPU_TIMER_HPP

#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#endif

namespace vfswind {
namespace gpu {

/**
 * @brief Check if timing is enabled via environment variable
 *
 * Set VFSWIND_TIMING=1 to enable timing output
 */
inline bool isTimingEnabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = std::getenv("VFSWIND_TIMING");
        enabled = (env != nullptr && std::string(env) == "1") ? 1 : 0;
    }
    return enabled == 1;
}

/**
 * @brief Simple wallclock timer for GPU kernels
 *
 * Uses std::chrono for portable high-resolution timing.
 * Includes Kokkos::fence() to ensure GPU work is complete.
 */
class GPUTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double, std::micro>;  // microseconds

    GPUTimer(const std::string& name = "Unnamed")
        : name_(name), elapsed_us_(0.0), running_(false) {}

    /**
     * @brief Start the timer
     *
     * Calls Kokkos::fence() to ensure any previous GPU work is complete.
     */
    void start() {
#ifdef ENABLE_GPU
        Kokkos::fence();  // Ensure GPU is idle before starting
#endif
        start_time_ = Clock::now();
        running_ = true;
    }

    /**
     * @brief Stop the timer
     *
     * Calls Kokkos::fence() to ensure GPU work is complete before measuring.
     */
    void stop() {
        if (!running_) return;
#ifdef ENABLE_GPU
        Kokkos::fence();  // Ensure GPU work is complete
#endif
        auto end_time = Clock::now();
        elapsed_us_ = std::chrono::duration_cast<Duration>(end_time - start_time_).count();
        running_ = false;
    }

    /**
     * @brief Get elapsed time in microseconds
     */
    double elapsed_us() const { return elapsed_us_; }

    /**
     * @brief Get elapsed time in milliseconds
     */
    double elapsed_ms() const { return elapsed_us_ / 1000.0; }

    /**
     * @brief Get elapsed time in seconds
     */
    double elapsed_s() const { return elapsed_us_ / 1000000.0; }

    /**
     * @brief Print timing result (if VFSWIND_TIMING=1)
     */
    void print() const {
        if (!isTimingEnabled()) return;

        if (elapsed_us_ < 1000.0) {
            printf("[TIMING] %s: %.2f us\n", name_.c_str(), elapsed_us_);
        } else if (elapsed_us_ < 1000000.0) {
            printf("[TIMING] %s: %.2f ms\n", name_.c_str(), elapsed_ms());
        } else {
            printf("[TIMING] %s: %.3f s\n", name_.c_str(), elapsed_s());
        }
    }

    /**
     * @brief Print timing result unconditionally
     */
    void printAlways() const {
        if (elapsed_us_ < 1000.0) {
            printf("[TIMING] %s: %.2f us\n", name_.c_str(), elapsed_us_);
        } else if (elapsed_us_ < 1000000.0) {
            printf("[TIMING] %s: %.2f ms\n", name_.c_str(), elapsed_ms());
        } else {
            printf("[TIMING] %s: %.3f s\n", name_.c_str(), elapsed_s());
        }
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    TimePoint start_time_;
    double elapsed_us_;
    bool running_;
};

/**
 * @brief RAII-style scoped timer
 *
 * Automatically starts on construction and prints on destruction.
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name) : timer_(name) {
        timer_.start();
    }

    ~ScopedTimer() {
        timer_.stop();
        timer_.print();
    }

    double elapsed_us() const { return timer_.elapsed_us(); }

private:
    GPUTimer timer_;
};

/**
 * @brief Timer registry for collecting statistics across multiple calls
 *
 * Useful for averaging kernel times over many timesteps.
 */
class TimerRegistry {
public:
    static TimerRegistry& instance() {
        static TimerRegistry registry;
        return registry;
    }

    /**
     * @brief Record a timing measurement
     */
    void record(const std::string& name, double elapsed_us) {
        timings_[name].push_back(elapsed_us);
    }

    /**
     * @brief Get statistics for a timer
     */
    void getStats(const std::string& name,
                  double& total_us, double& avg_us, double& min_us, double& max_us,
                  int& count) const {
        auto it = timings_.find(name);
        if (it == timings_.end() || it->second.empty()) {
            total_us = avg_us = min_us = max_us = 0.0;
            count = 0;
            return;
        }

        const auto& times = it->second;
        count = static_cast<int>(times.size());
        total_us = std::accumulate(times.begin(), times.end(), 0.0);
        avg_us = total_us / count;
        min_us = *std::min_element(times.begin(), times.end());
        max_us = *std::max_element(times.begin(), times.end());
    }

    /**
     * @brief Print summary of all recorded timings
     */
    void printSummary() const {
        if (timings_.empty()) return;

        printf("\n");
        printf("================================================================================\n");
        printf("                          GPU TIMING SUMMARY                                    \n");
        printf("================================================================================\n");
        printf("%-35s %8s %12s %12s %12s %12s\n",
               "Kernel", "Calls", "Total (ms)", "Avg (us)", "Min (us)", "Max (us)");
        printf("--------------------------------------------------------------------------------\n");

        double grand_total = 0.0;

        // Collect and sort by total time
        std::vector<std::pair<std::string, double>> sorted;
        for (const auto& kv : timings_) {
            double total = std::accumulate(kv.second.begin(), kv.second.end(), 0.0);
            sorted.emplace_back(kv.first, total);
            grand_total += total;
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& kv : sorted) {
            double total_us, avg_us, min_us, max_us;
            int count;
            getStats(kv.first, total_us, avg_us, min_us, max_us, count);

            printf("%-35s %8d %12.2f %12.2f %12.2f %12.2f\n",
                   kv.first.c_str(), count, total_us / 1000.0, avg_us, min_us, max_us);
        }

        printf("--------------------------------------------------------------------------------\n");
        printf("%-35s %8s %12.2f\n", "TOTAL", "", grand_total / 1000.0);
        printf("================================================================================\n\n");
    }

    /**
     * @brief Clear all recorded timings
     */
    void clear() {
        timings_.clear();
    }

private:
    TimerRegistry() = default;
    std::map<std::string, std::vector<double>> timings_;
};

/**
 * @brief Timer that records to the global registry
 */
class RegisteredTimer {
public:
    RegisteredTimer(const std::string& name) : timer_(name) {
        timer_.start();
    }

    ~RegisteredTimer() {
        timer_.stop();
        TimerRegistry::instance().record(timer_.name(), timer_.elapsed_us());
        timer_.print();
    }

private:
    GPUTimer timer_;
};

/**
 * @brief Convenience macros for timing
 *
 * VFSWIND_TIME_KERNEL("name") - Times the current scope, prints if enabled
 * VFSWIND_TIME_KERNEL_ALWAYS("name") - Times and always prints
 * VFSWIND_TIME_REGISTERED("name") - Times and records to registry
 */
#define VFSWIND_TIME_KERNEL(name) \
    vfswind::gpu::ScopedTimer _vfswind_timer_##__LINE__(name)

#define VFSWIND_TIME_REGISTERED(name) \
    vfswind::gpu::RegisteredTimer _vfswind_reg_timer_##__LINE__(name)

} // namespace gpu
} // namespace vfswind

#endif // VFSWIND_GPU_TIMER_HPP
