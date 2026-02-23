/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * IBM GPU Kernel Tests                                           *
 * Phase 3: Immersed Boundary Method GPU Acceleration             *
 ******************************************************************/

#include <gtest/gtest.h>
#include <cmath>

#ifdef ENABLE_GPU
#include <Kokkos_Core.hpp>
#include "gpu_config.hpp"
#include "kernels/ibm_kernels.hpp"
#endif

class IBMKernelTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
#ifdef ENABLE_GPU
        Kokkos::initialize();
#endif
    }

    static void TearDownTestSuite() {
#ifdef ENABLE_GPU
        Kokkos::finalize();
#endif
    }
};

// Test: IBM surface mesh allocation
TEST_F(IBMKernelTest, SurfaceMeshAllocation) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::ibm;

    IBMSurfaceMesh<DeviceMemorySpace> mesh;

    // Allocate for a simple triangle mesh (4 vertices, 2 triangles)
    mesh.allocate(4, 2);

    EXPECT_EQ(mesh.n_vertices, 4);
    EXPECT_EQ(mesh.n_elements, 2);

    // Verify views are allocated
    EXPECT_EQ(mesh.x_bp.extent(0), 4);
    EXPECT_EQ(mesh.y_bp.extent(0), 4);
    EXPECT_EQ(mesh.z_bp.extent(0), 4);
    EXPECT_EQ(mesh.nv1.extent(0), 2);
    EXPECT_EQ(mesh.nv2.extent(0), 2);
    EXPECT_EQ(mesh.nv3.extent(0), 2);
    EXPECT_EQ(mesh.nf_x.extent(0), 2);
    EXPECT_EQ(mesh.dA.extent(0), 2);
#else
    SUCCEED();
#endif
}

// Test: IBM interpolation data allocation
TEST_F(IBMKernelTest, InterpDataAllocation) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::ibm;

    IBMInterpData<DeviceMemorySpace> interp;

    // Allocate for 100 interpolation points
    interp.allocate(100);

    EXPECT_EQ(interp.n_points, 100);

    // Verify views are allocated
    EXPECT_EQ(interp.cell_i.extent(0), 100);
    EXPECT_EQ(interp.i1.extent(0), 100);
    EXPECT_EQ(interp.cr1.extent(0), 100);
    EXPECT_EQ(interp.d_surface.extent(0), 100);
    EXPECT_EQ(interp.mode.extent(0), 100);
#else
    SUCCEED();
#endif
}

// Test: IBM force data allocation
TEST_F(IBMKernelTest, ForceDataAllocation) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::ibm;

    IBMForceData<DeviceMemorySpace> forces;

    // Allocate for 50 force points
    forces.allocate(50);

    EXPECT_EQ(forces.n_points, 50);

    // Verify views are allocated
    EXPECT_EQ(forces.F_x.extent(0), 50);
    EXPECT_EQ(forces.F_y.extent(0), 50);
    EXPECT_EQ(forces.F_z.extent(0), 50);
    EXPECT_EQ(forces.U_x.extent(0), 50);

    // Test zero forces
    forces.zeroForces();
    Kokkos::fence();

    // Verify forces are zero
    auto h_Fx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.F_x);
    for (int i = 0; i < 50; i++) {
        EXPECT_DOUBLE_EQ(h_Fx(i), 0.0);
    }
#else
    SUCCEED();
#endif
}

// Test: Delta function values
TEST_F(IBMKernelTest, DeltaFunction) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::ibm;

    double h = 1.0;

    // At r=0, delta should be 2/3h
    double d0 = IBMForceKernel::deltaFunction(0.0, h);
    EXPECT_NEAR(d0, 2.0/3.0, 1e-10);

    // At r=h, delta should be 1/(6h) * (0.5)^2 = 1/24h
    double d1 = IBMForceKernel::deltaFunction(h, h);
    EXPECT_NEAR(d1, 1.0/24.0, 1e-10);

    // At r=1.5h, delta should be 0
    double d15 = IBMForceKernel::deltaFunction(1.5 * h, h);
    EXPECT_NEAR(d15, 0.0, 1e-10);

    // At r=2h, delta should be 0
    double d2 = IBMForceKernel::deltaFunction(2.0 * h, h);
    EXPECT_DOUBLE_EQ(d2, 0.0);

    // Symmetry: delta(-r) = delta(r)
    double dn = IBMForceKernel::deltaFunction(-0.3 * h, h);
    double dp = IBMForceKernel::deltaFunction(0.3 * h, h);
    EXPECT_DOUBLE_EQ(dn, dp);
#else
    SUCCEED();
#endif
}

// Test: Simple velocity interpolation
TEST_F(IBMKernelTest, VelocityInterpolation) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu;
    using namespace vfswind::gpu::ibm;

    // Create a small 8x8x8 velocity field
    const int N = 8;
    VectorView3D<DeviceMemorySpace> ucat("ucat", N, N, N);

    // Fill with uniform velocity (1, 0, 0)
    Kokkos::parallel_for("FillVelocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            ucat(k, j, i).x = 1.0;
            ucat(k, j, i).y = 0.0;
            ucat(k, j, i).z = 0.0;
        }
    );
    Kokkos::fence();

    // Create single interpolation point
    IBMInterpData<DeviceMemorySpace> interp;
    interp.allocate(1);

    // Set up interpolation to use cell (4,4,4) and neighbors
    auto h_i1 = Kokkos::create_mirror_view(interp.i1);
    auto h_j1 = Kokkos::create_mirror_view(interp.j1);
    auto h_k1 = Kokkos::create_mirror_view(interp.k1);
    auto h_i2 = Kokkos::create_mirror_view(interp.i2);
    auto h_j2 = Kokkos::create_mirror_view(interp.j2);
    auto h_k2 = Kokkos::create_mirror_view(interp.k2);
    auto h_i3 = Kokkos::create_mirror_view(interp.i3);
    auto h_j3 = Kokkos::create_mirror_view(interp.j3);
    auto h_k3 = Kokkos::create_mirror_view(interp.k3);
    auto h_cr1 = Kokkos::create_mirror_view(interp.cr1);
    auto h_cr2 = Kokkos::create_mirror_view(interp.cr2);
    auto h_cr3 = Kokkos::create_mirror_view(interp.cr3);
    auto h_mode = Kokkos::create_mirror_view(interp.mode);

    // Use three points with equal weights (1/3 each)
    h_i1(0) = 3; h_j1(0) = 4; h_k1(0) = 4;
    h_i2(0) = 4; h_j2(0) = 4; h_k2(0) = 4;
    h_i3(0) = 5; h_j3(0) = 4; h_k3(0) = 4;
    h_cr1(0) = 1.0/3.0;
    h_cr2(0) = 1.0/3.0;
    h_cr3(0) = 1.0/3.0;
    h_mode(0) = 1;  // Valid

    Kokkos::deep_copy(interp.i1, h_i1);
    Kokkos::deep_copy(interp.j1, h_j1);
    Kokkos::deep_copy(interp.k1, h_k1);
    Kokkos::deep_copy(interp.i2, h_i2);
    Kokkos::deep_copy(interp.j2, h_j2);
    Kokkos::deep_copy(interp.k2, h_k2);
    Kokkos::deep_copy(interp.i3, h_i3);
    Kokkos::deep_copy(interp.j3, h_j3);
    Kokkos::deep_copy(interp.k3, h_k3);
    Kokkos::deep_copy(interp.cr1, h_cr1);
    Kokkos::deep_copy(interp.cr2, h_cr2);
    Kokkos::deep_copy(interp.cr3, h_cr3);
    Kokkos::deep_copy(interp.mode, h_mode);

    // Create force data for output
    IBMForceData<DeviceMemorySpace> forces;
    forces.allocate(1);

    // Create domain info
    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;

    // Execute interpolation
    IBMInterpolationKernel::interpolateVelocity(ucat, interp, forces, domain);

    // Check result - should be (1, 0, 0) since uniform field
    auto h_Ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.U_x);
    auto h_Uy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.U_y);
    auto h_Uz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.U_z);

    EXPECT_NEAR(h_Ux(0), 1.0, 1e-10);
    EXPECT_NEAR(h_Uy(0), 0.0, 1e-10);
    EXPECT_NEAR(h_Uz(0), 0.0, 1e-10);
#else
    SUCCEED();
#endif
}

// Test: Direct forcing computation
TEST_F(IBMKernelTest, DirectForcing) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::ibm;
    using namespace vfswind::gpu;

    // Create surface mesh with one element
    IBMSurfaceMesh<DeviceMemorySpace> surface;
    surface.allocate(3, 1);

    // Set up a single triangle at z=0
    auto h_nv1 = Kokkos::create_mirror_view(surface.nv1);
    auto h_nv2 = Kokkos::create_mirror_view(surface.nv2);
    auto h_nv3 = Kokkos::create_mirror_view(surface.nv3);
    h_nv1(0) = 0; h_nv2(0) = 1; h_nv3(0) = 2;
    Kokkos::deep_copy(surface.nv1, h_nv1);
    Kokkos::deep_copy(surface.nv2, h_nv2);
    Kokkos::deep_copy(surface.nv3, h_nv3);

    // Surface velocity = 0 (stationary body)
    Kokkos::deep_copy(surface.u_x, 0.0);
    Kokkos::deep_copy(surface.u_y, 0.0);
    Kokkos::deep_copy(surface.u_z, 0.0);

    // Create interpolation data pointing to element 0
    IBMInterpData<DeviceMemorySpace> interp;
    interp.allocate(1);

    auto h_elem = Kokkos::create_mirror_view(interp.element_idx);
    auto h_cs1 = Kokkos::create_mirror_view(interp.cs1);
    auto h_cs2 = Kokkos::create_mirror_view(interp.cs2);
    auto h_cs3 = Kokkos::create_mirror_view(interp.cs3);
    auto h_mode = Kokkos::create_mirror_view(interp.mode);

    h_elem(0) = 0;
    h_cs1(0) = 1.0/3.0;
    h_cs2(0) = 1.0/3.0;
    h_cs3(0) = 1.0/3.0;
    h_mode(0) = 1;

    Kokkos::deep_copy(interp.element_idx, h_elem);
    Kokkos::deep_copy(interp.cs1, h_cs1);
    Kokkos::deep_copy(interp.cs2, h_cs2);
    Kokkos::deep_copy(interp.cs3, h_cs3);
    Kokkos::deep_copy(interp.mode, h_mode);

    // Create force data with interpolated velocity = (1, 0, 0)
    IBMForceData<DeviceMemorySpace> forces;
    forces.allocate(1);

    auto h_Ux = Kokkos::create_mirror_view(forces.U_x);
    auto h_Uy = Kokkos::create_mirror_view(forces.U_y);
    auto h_Uz = Kokkos::create_mirror_view(forces.U_z);
    h_Ux(0) = 1.0;
    h_Uy(0) = 0.0;
    h_Uz(0) = 0.0;
    Kokkos::deep_copy(forces.U_x, h_Ux);
    Kokkos::deep_copy(forces.U_y, h_Uy);
    Kokkos::deep_copy(forces.U_z, h_Uz);

    // Compute direct forcing with dt = 0.1
    double dt = 0.1;
    IBMForceKernel::computeDirectForcing(surface, interp, forces, dt);

    // Expected: F = (U_des - U_interp) / dt = (0 - 1) / 0.1 = -10
    auto h_Fx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.F_x);
    auto h_Fy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.F_y);
    auto h_Fz = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.F_z);

    EXPECT_NEAR(h_Fx(0), -10.0, 1e-10);
    EXPECT_NEAR(h_Fy(0), 0.0, 1e-10);
    EXPECT_NEAR(h_Fz(0), 0.0, 1e-10);
#else
    SUCCEED();
#endif
}

// Test: Force spreading to body force field
TEST_F(IBMKernelTest, ForceSpreading) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::ibm;
    using namespace vfswind::gpu;

    const int N = 8;

    // Create surface mesh with one element
    IBMSurfaceMesh<DeviceMemorySpace> surface;
    surface.allocate(3, 1);

    // Set element area = 1.0
    auto h_dA = Kokkos::create_mirror_view(surface.dA);
    h_dA(0) = 1.0;
    Kokkos::deep_copy(surface.dA, h_dA);

    // Create interpolation data - point at cell (4, 4, 4)
    IBMInterpData<DeviceMemorySpace> interp;
    interp.allocate(1);

    auto h_ci = Kokkos::create_mirror_view(interp.cell_i);
    auto h_cj = Kokkos::create_mirror_view(interp.cell_j);
    auto h_ck = Kokkos::create_mirror_view(interp.cell_k);
    auto h_elem = Kokkos::create_mirror_view(interp.element_idx);
    auto h_mode = Kokkos::create_mirror_view(interp.mode);

    h_ci(0) = 4; h_cj(0) = 4; h_ck(0) = 4;
    h_elem(0) = 0;
    h_mode(0) = 1;

    Kokkos::deep_copy(interp.cell_i, h_ci);
    Kokkos::deep_copy(interp.cell_j, h_cj);
    Kokkos::deep_copy(interp.cell_k, h_ck);
    Kokkos::deep_copy(interp.element_idx, h_elem);
    Kokkos::deep_copy(interp.mode, h_mode);

    // Create force data with F = (1, 2, 3)
    IBMForceData<DeviceMemorySpace> forces;
    forces.allocate(1);

    auto h_Fx = Kokkos::create_mirror_view(forces.F_x);
    auto h_Fy = Kokkos::create_mirror_view(forces.F_y);
    auto h_Fz = Kokkos::create_mirror_view(forces.F_z);
    h_Fx(0) = 1.0;
    h_Fy(0) = 2.0;
    h_Fz(0) = 3.0;
    Kokkos::deep_copy(forces.F_x, h_Fx);
    Kokkos::deep_copy(forces.F_y, h_Fy);
    Kokkos::deep_copy(forces.F_z, h_Fz);

    // Create body force field
    VectorView3D<DeviceMemorySpace> f_body("f_body", N, N, N);

    // Domain info
    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;

    // Spread forces
    double h = 1.0;
    IBMForceKernel::spreadForces(forces, surface, interp, f_body, h, domain);

    // Check that force appeared at cell (4, 4, 4)
    auto h_fbody = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_body);

    // Force = F * dA = (1, 2, 3) * 1.0 = (1, 2, 3)
    EXPECT_NEAR(h_fbody(4, 4, 4).x, 1.0, 1e-10);
    EXPECT_NEAR(h_fbody(4, 4, 4).y, 2.0, 1e-10);
    EXPECT_NEAR(h_fbody(4, 4, 4).z, 3.0, 1e-10);

    // Check that other cells are zero
    EXPECT_NEAR(h_fbody(3, 4, 4).x, 0.0, 1e-10);
    EXPECT_NEAR(h_fbody(4, 3, 4).x, 0.0, 1e-10);
#else
    SUCCEED();
#endif
}

// Test: Add body force to RHS
TEST_F(IBMKernelTest, AddBodyForceToRHS) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::ibm;
    using namespace vfswind::gpu;

    const int N = 8;

    // Create RHS and body force fields
    VectorView3D<DeviceMemorySpace> rhs("rhs", N, N, N);
    VectorView3D<DeviceMemorySpace> f_body("f_body", N, N, N);
    ScalarView3D<DeviceMemorySpace> nvert("nvert", N, N, N);

    // Initialize to zero
    Kokkos::deep_copy(rhs, Cmpnts3());
    Kokkos::deep_copy(f_body, Cmpnts3());
    Kokkos::deep_copy(nvert, 0.0);

    // Set a body force at cell (4, 4, 4)
    auto h_fbody = Kokkos::create_mirror_view(f_body);
    h_fbody(4, 4, 4).x = 5.0;
    h_fbody(4, 4, 4).y = 10.0;
    h_fbody(4, 4, 4).z = 15.0;
    Kokkos::deep_copy(f_body, h_fbody);

    // Domain info
    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;
    domain.lxs = 1; domain.lxe = N - 1;
    domain.lys = 1; domain.lye = N - 1;
    domain.lzs = 1; domain.lze = N - 1;

    // Add body force to RHS
    IBMForceKernel::addBodyForceToRHS(rhs, f_body, nvert, domain);

    // Check result
    auto h_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhs);

    EXPECT_NEAR(h_rhs(4, 4, 4).x, 5.0, 1e-10);
    EXPECT_NEAR(h_rhs(4, 4, 4).y, 10.0, 1e-10);
    EXPECT_NEAR(h_rhs(4, 4, 4).z, 15.0, 1e-10);
#else
    SUCCEED();
#endif
}

// Test: Performance test with larger mesh
TEST_F(IBMKernelTest, InterpolationPerformance) {
#ifdef ENABLE_GPU
    using namespace vfswind::gpu::ibm;
    using namespace vfswind::gpu;

    const int N = 64;
    const int n_ibm_points = 1000;

    // Create velocity field
    VectorView3D<DeviceMemorySpace> ucat("ucat", N, N, N);
    Kokkos::parallel_for("FillVelocity",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {N, N, N}),
        KOKKOS_LAMBDA(int k, int j, int i) {
            ucat(k, j, i).x = static_cast<double>(i);
            ucat(k, j, i).y = static_cast<double>(j);
            ucat(k, j, i).z = static_cast<double>(k);
        }
    );
    Kokkos::fence();

    // Create interpolation data
    IBMInterpData<DeviceMemorySpace> interp;
    interp.allocate(n_ibm_points);

    // Set up random-ish interpolation points
    auto h_i1 = Kokkos::create_mirror_view(interp.i1);
    auto h_j1 = Kokkos::create_mirror_view(interp.j1);
    auto h_k1 = Kokkos::create_mirror_view(interp.k1);
    auto h_cr1 = Kokkos::create_mirror_view(interp.cr1);
    auto h_cr2 = Kokkos::create_mirror_view(interp.cr2);
    auto h_cr3 = Kokkos::create_mirror_view(interp.cr3);
    auto h_mode = Kokkos::create_mirror_view(interp.mode);

    for (int p = 0; p < n_ibm_points; p++) {
        int base_i = 10 + (p * 7) % (N - 20);
        int base_j = 10 + (p * 11) % (N - 20);
        int base_k = 10 + (p * 13) % (N - 20);

        h_i1(p) = base_i;
        h_j1(p) = base_j;
        h_k1(p) = base_k;
        h_cr1(p) = 0.5;
        h_cr2(p) = 0.3;
        h_cr3(p) = 0.2;
        h_mode(p) = 1;
    }

    // Copy i2, j2, k2, i3, j3, k3 (offset from i1, j1, k1)
    auto h_i2 = Kokkos::create_mirror_view(interp.i2);
    auto h_j2 = Kokkos::create_mirror_view(interp.j2);
    auto h_k2 = Kokkos::create_mirror_view(interp.k2);
    auto h_i3 = Kokkos::create_mirror_view(interp.i3);
    auto h_j3 = Kokkos::create_mirror_view(interp.j3);
    auto h_k3 = Kokkos::create_mirror_view(interp.k3);

    for (int p = 0; p < n_ibm_points; p++) {
        h_i2(p) = h_i1(p) + 1;
        h_j2(p) = h_j1(p);
        h_k2(p) = h_k1(p);
        h_i3(p) = h_i1(p);
        h_j3(p) = h_j1(p) + 1;
        h_k3(p) = h_k1(p);
    }

    Kokkos::deep_copy(interp.i1, h_i1);
    Kokkos::deep_copy(interp.j1, h_j1);
    Kokkos::deep_copy(interp.k1, h_k1);
    Kokkos::deep_copy(interp.i2, h_i2);
    Kokkos::deep_copy(interp.j2, h_j2);
    Kokkos::deep_copy(interp.k2, h_k2);
    Kokkos::deep_copy(interp.i3, h_i3);
    Kokkos::deep_copy(interp.j3, h_j3);
    Kokkos::deep_copy(interp.k3, h_k3);
    Kokkos::deep_copy(interp.cr1, h_cr1);
    Kokkos::deep_copy(interp.cr2, h_cr2);
    Kokkos::deep_copy(interp.cr3, h_cr3);
    Kokkos::deep_copy(interp.mode, h_mode);

    // Create force data
    IBMForceData<DeviceMemorySpace> forces;
    forces.allocate(n_ibm_points);

    // Domain info
    KernelDomainInfo domain;
    domain.mx = N; domain.my = N; domain.mz = N;

    // Run interpolation multiple times
    for (int iter = 0; iter < 10; iter++) {
        IBMInterpolationKernel::interpolateVelocity(ucat, interp, forces, domain);
    }

    // Verify some results
    auto h_Ux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forces.U_x);

    // All interpolated values should be finite
    for (int p = 0; p < n_ibm_points; p++) {
        EXPECT_TRUE(std::isfinite(h_Ux(p)));
    }

    std::cout << "IBM interpolation performance test: "
              << n_ibm_points << " points on " << N << "^3 grid" << std::endl;
#else
    SUCCEED();
#endif
}
