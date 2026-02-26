/*
 * VFS-Wind GPU Kernel Benchmark
 *
 * Measures performance of individual GPU kernels at various grid sizes.
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#include "gpu_timer.hpp"
#include "gpu_config.hpp"
#include "kernels/kernels.hpp"
#include "kernels/turbulence_kernels.hpp"

using namespace vfswind::gpu;

// Benchmark configuration
struct BenchConfig {
    int grid_size = 64;
    int iterations = 100;
    std::string test = "all";
    int num_threads = 0;
};

BenchConfig parseArgs(int argc, char** argv) {
    BenchConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--grid_size=") == 0) {
            config.grid_size = std::stoi(arg.substr(12));
        } else if (arg.find("--iterations=") == 0) {
            config.iterations = std::stoi(arg.substr(13));
        } else if (arg.find("--test=") == 0) {
            config.test = arg.substr(7);
        } else if (arg.find("--threads=") == 0) {
            config.num_threads = std::stoi(arg.substr(10));
        }
    }
    return config;
}

void printHeader(const std::string& name, int N, int iters) {
    std::cout << "\n----------------------------------------\n";
    std::cout << "Benchmark: " << name << "\n";
    std::cout << "Grid: " << N << "³ = " << (N*N*N) << " cells\n";
    std::cout << "Iterations: " << iters << "\n";
    std::cout << "----------------------------------------\n";
}

void printResult(const std::string& name, double total_us, int iters, long cells) {
    double avg_us = total_us / iters;
    double throughput = (cells / 1e6) / (avg_us / 1e6);  // MCells/sec

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::setw(25) << std::left << name
              << " Total: " << std::setw(10) << total_us/1000.0 << " ms"
              << "  Avg: " << std::setw(10) << avg_us << " us"
              << "  Throughput: " << std::setw(8) << throughput << " MCells/s\n";
}

// Benchmark: Convection kernel
void benchmarkConvection(int N, int iterations) {
    printHeader("Convection (QUICK scheme)", N, iterations);

    VectorView3D<DeviceMemorySpace> ucont("ucont", N, N, N);
    VectorView3D<DeviceMemorySpace> ucat("ucat", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);
    VectorView3D<DeviceMemorySpace> conv("conv", N, N, N);

    // Initialize with test data
    Kokkos::parallel_for("init_conv", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            ucont(k,j,i).x = 1.0;
            ucont(k,j,i).y = 0.0;
            ucont(k,j,i).z = 0.0;
            ucat(k,j,i).x = 1.0;
            ucat(k,j,i).y = 0.0;
            ucat(k,j,i).z = 0.0;
            nvert(k,j,i) = 0.0;
        });
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = domain.my = domain.mz = N;
    domain.lxs = domain.lys = domain.lzs = 1;
    domain.lxe = domain.lye = domain.lze = N - 1;
    domain.ren = 1000.0;

    GPUTimer timer("Convection");
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        kernels::ConvectionKernel::execute(ucont, ucat, nvert, conv, domain);
    }
    timer.stop();

    printResult("Convection", timer.elapsed_us(), iterations, (long)N*N*N);
    TimerRegistry::instance().record("Convection", timer.elapsed_us() / iterations);
}

// Benchmark: LES Smagorinsky
void benchmarkLES(int N, int iterations) {
    printHeader("LES Smagorinsky", N, iterations);

    VectorView3D<DeviceMemorySpace> ucat("ucat", N, N, N);
    VectorView3D<DeviceMemorySpace> csi("csi", N, N, N);
    VectorView3D<DeviceMemorySpace> eta("eta", N, N, N);
    VectorView3D<DeviceMemorySpace> zet("zet", N, N, N);
    ScalarView3D<DeviceMemorySpace> aj("aj", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);
    ScalarView3D<DeviceMemorySpace> cs("cs", N, N, N);
    ScalarView3D<DeviceMemorySpace> nu_t("nu_t", N, N, N);

    double dx = 1.0 / N;
    Kokkos::parallel_for("init_les", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Simple shear flow
            ucat(k,j,i).x = (double)j / N;
            ucat(k,j,i).y = 0.0;
            ucat(k,j,i).z = 0.0;
            // Unit metrics (Cartesian grid)
            csi(k,j,i).x = 1.0/dx; csi(k,j,i).y = 0.0; csi(k,j,i).z = 0.0;
            eta(k,j,i).x = 0.0; eta(k,j,i).y = 1.0/dx; eta(k,j,i).z = 0.0;
            zet(k,j,i).x = 0.0; zet(k,j,i).y = 0.0; zet(k,j,i).z = 1.0/dx;
            aj(k,j,i) = 1.0 / (dx * dx * dx);
            nvert(k,j,i) = 0.0;
            cs(k,j,i) = 0.1;  // Smagorinsky constant
        });
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = domain.my = domain.mz = N;
    domain.lxs = domain.lys = domain.lzs = 1;
    domain.lxe = domain.lye = domain.lze = N - 1;
    domain.ren = 1000.0;

    GPUTimer timer("LES_Smagorinsky");
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        les::SmagorinskyKernel::computeEddyViscosity(
            ucat, nvert, csi, eta, zet, aj, cs, nu_t, domain);
    }
    timer.stop();

    printResult("LES Smagorinsky", timer.elapsed_us(), iterations, (long)N*N*N);
    TimerRegistry::instance().record("LES_Smagorinsky", timer.elapsed_us() / iterations);
}

// Benchmark: Level-set advection (WENO3)
void benchmarkLevelset(int N, int iterations) {
    printHeader("Level-set Advection (WENO3)", N, iterations);

    ScalarView3D<DeviceMemorySpace> phi("phi", N, N, N);
    VectorView3D<DeviceMemorySpace> ucont("ucont", N, N, N);
    ScalarView3D<DeviceMemorySpace> aj("aj", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);

    double dx = 1.0 / N;
    Kokkos::parallel_for("init_ls", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            // Sphere level-set
            double x = (i - N/2.0) * dx;
            double y = (j - N/2.0) * dx;
            double z = (k - N/2.0) * dx;
            phi(k,j,i) = sqrt(x*x + y*y + z*z) - 0.25;
            // Uniform velocity
            ucont(k,j,i).x = 1.0;
            ucont(k,j,i).y = 0.0;
            ucont(k,j,i).z = 0.0;
            aj(k,j,i) = 1.0 / (dx * dx * dx);
            nvert(k,j,i) = 0.0;
        });
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = domain.my = domain.mz = N;
    domain.lxs = domain.lys = domain.lzs = 2;
    domain.lxe = domain.lye = domain.lze = N - 2;
    domain.ren = 1000.0;

    double dt = 0.001;

    GPUTimer timer("Levelset_WENO3");
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        LevelsetKernel::advect(phi, ucont, nvert, aj, dt, domain,
                              levelset::LevelsetAdvectionKernel::Scheme::WENO3);
    }
    timer.stop();

    printResult("Levelset WENO3", timer.elapsed_us(), iterations, (long)N*N*N);
    TimerRegistry::instance().record("Levelset_WENO3", timer.elapsed_us() / iterations);
}

// Benchmark: Two-phase properties
void benchmarkTwoPhase(int N, int iterations) {
    printHeader("Two-Phase Properties", N, iterations);

    ScalarView3D<DeviceMemorySpace> phi("phi", N, N, N);
    ScalarView3D<DeviceMemorySpace> rho("rho", N, N, N);
    ScalarView3D<DeviceMemorySpace> mu("mu", N, N, N);

    double dx = 1.0 / N;
    Kokkos::parallel_for("init_2ph", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            double y = (double)j / N - 0.5;
            phi(k,j,i) = y;  // Horizontal interface
        });
    Kokkos::fence();

    KernelDomainInfo domain;
    domain.mx = domain.my = domain.mz = N;
    domain.lxs = domain.lys = domain.lzs = 0;
    domain.lxe = domain.lye = domain.lze = N;

    double rho1 = 1000.0, rho2 = 1.0;
    double mu1 = 1e-3, mu2 = 1.8e-5;
    double epsilon = 3.0 * dx;

    GPUTimer timer("TwoPhase_Props");
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        LevelsetKernel::computeProperties(phi, rho, mu, rho1, rho2, mu1, mu2, epsilon, domain);
    }
    timer.stop();

    printResult("Two-Phase Props", timer.elapsed_us(), iterations, (long)N*N*N);
    TimerRegistry::instance().record("TwoPhase_Props", timer.elapsed_us() / iterations);
}

// Benchmark: Memory bandwidth (copy kernel)
void benchmarkMemory(int N, int iterations) {
    printHeader("Memory Bandwidth (Copy)", N, iterations);

    ScalarView3D<DeviceMemorySpace> src("src", N, N, N);
    ScalarView3D<DeviceMemorySpace> dst("dst", N, N, N);

    Kokkos::deep_copy(src, 1.0);
    Kokkos::fence();

    GPUTimer timer("Memory_Copy");
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("copy", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                dst(k,j,i) = src(k,j,i);
            });
    }
    Kokkos::fence();
    timer.stop();

    double bytes = 2.0 * N * N * N * sizeof(double) * iterations;  // Read + Write
    double bandwidth_gb = bytes / (timer.elapsed_us() * 1e-6) / 1e9;

    printResult("Memory Copy", timer.elapsed_us(), iterations, (long)N*N*N);
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2)
              << bandwidth_gb << " GB/s\n";
    TimerRegistry::instance().record("Memory_Copy", timer.elapsed_us() / iterations);
}

// Benchmark: Stencil operation (7-point Laplacian)
void benchmarkStencil(int N, int iterations) {
    printHeader("Stencil (7-point Laplacian)", N, iterations);

    ScalarView3D<DeviceMemorySpace> u("u", N, N, N);
    ScalarView3D<DeviceMemorySpace> lap("lap", N, N, N);

    Kokkos::parallel_for("init_stencil", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {N,N,N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            u(k,j,i) = sin(2.0 * M_PI * i / N) * sin(2.0 * M_PI * j / N);
        });
    Kokkos::fence();

    GPUTimer timer("Stencil_Laplacian");
    timer.start();
    for (int iter = 0; iter < iterations; ++iter) {
        Kokkos::parallel_for("laplacian",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({1,1,1}, {N-1,N-1,N-1}),
            KOKKOS_LAMBDA(int k, int j, int i) {
                lap(k,j,i) = u(k,j,i+1) + u(k,j,i-1)
                           + u(k,j+1,i) + u(k,j-1,i)
                           + u(k+1,j,i) + u(k-1,j,i)
                           - 6.0 * u(k,j,i);
            });
    }
    Kokkos::fence();
    timer.stop();

    printResult("7-pt Laplacian", timer.elapsed_us(), iterations, (long)N*N*N);
    TimerRegistry::instance().record("Stencil_Laplacian", timer.elapsed_us() / iterations);
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        BenchConfig config = parseArgs(argc, argv);

        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "VFS-Wind GPU Kernel Benchmarks\n";
        std::cout << "========================================\n";
        std::cout << "Backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";
        std::cout << "Grid Size: " << config.grid_size << "³\n";
        std::cout << "Iterations: " << config.iterations << "\n";

        const char* omp_threads = std::getenv("OMP_NUM_THREADS");
        if (omp_threads) {
            std::cout << "OMP_NUM_THREADS: " << omp_threads << "\n";
        }
        std::cout << "\n";

        int N = config.grid_size;
        int iters = config.iterations;

        if (config.test == "all" || config.test == "memory") {
            benchmarkMemory(N, iters);
        }
        if (config.test == "all" || config.test == "stencil") {
            benchmarkStencil(N, iters);
        }
        if (config.test == "all" || config.test == "convection") {
            benchmarkConvection(N, iters);
        }
        if (config.test == "all" || config.test == "les") {
            benchmarkLES(N, iters);
        }
        if (config.test == "all" || config.test == "levelset") {
            benchmarkLevelset(N, iters);
        }
        if (config.test == "all" || config.test == "twophase") {
            benchmarkTwoPhase(N, iters);
        }

        std::cout << "\n";
        TimerRegistry::instance().printSummary();
    }
    Kokkos::finalize();
    return 0;
}

#else

int main() {
    std::cerr << "GPU support not enabled. Rebuild with -DENABLE_GPU=ON\n";
    return 1;
}

#endif
