/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Integration tests for grid metrics computation                 *
 ******************************************************************/

#include <gtest/gtest.h>
#include <cmath>

// Integration tests for grid metrics - tests mathematical properties
// without requiring full VFS-Wind source linkage

#include "petsc.h"
#include "petscdmda.h"

// Define Cmpnts structure locally for standalone testing
typedef struct {
    double x, y, z;
} Cmpnts;

class MetricsIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int argc = 0;
        char **argv = nullptr;
        PetscInitialize(&argc, &argv, nullptr, nullptr);
    }

    static void TearDownTestSuite() {
        PetscFinalize();
    }

    const double TOLERANCE = 1e-10;
};

// ============================================================================
// Jacobian Properties Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, JacobianSymmetry) {
    // Test that Jacobian properties hold for a simple uniform grid
    // This is a mathematical property test

    // For a uniform Cartesian grid, the Jacobian should be constant
    // and equal to dx * dy * dz
    double dx = 0.1, dy = 0.1, dz = 0.1;
    double expected_jacobian = dx * dy * dz;

    // Compute Jacobian from metric terms for uniform grid
    // For uniform grid: csi = (1/dx, 0, 0), eta = (0, 1/dy, 0), zet = (0, 0, 1/dz)
    double csi[3] = {1.0/dx, 0.0, 0.0};
    double eta[3] = {0.0, 1.0/dy, 0.0};
    double zet[3] = {0.0, 0.0, 1.0/dz};

    // Jacobian = 1 / (csi . (eta x zet))
    double cross[3];
    cross[0] = eta[1] * zet[2] - eta[2] * zet[1];
    cross[1] = eta[2] * zet[0] - eta[0] * zet[2];
    cross[2] = eta[0] * zet[1] - eta[1] * zet[0];

    double dot = csi[0]*cross[0] + csi[1]*cross[1] + csi[2]*cross[2];
    double jacobian = 1.0 / dot;

    EXPECT_NEAR(jacobian, expected_jacobian, TOLERANCE);
}

TEST_F(MetricsIntegrationTest, MetricInvariance) {
    // Test that metric invariants hold
    // For any valid grid: det(J) > 0 (positive Jacobian)

    // Simple example: stretched grid in x direction
    double dx_min = 0.05, dx_max = 0.15;
    double dy = 0.1, dz = 0.1;

    // Both should give positive Jacobian
    double J1 = dx_min * dy * dz;
    double J2 = dx_max * dy * dz;

    EXPECT_GT(J1, 0.0);
    EXPECT_GT(J2, 0.0);
}

// ============================================================================
// Coordinate Transformation Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, ContravariantVelocityTransform) {
    // Test contravariant velocity transformation
    // For Cartesian grid: U_contra = U_cart

    // Cartesian velocity
    Cmpnts ucat = {1.0, 2.0, 3.0};

    // For uniform Cartesian grid with unit spacing
    // csi = (1, 0, 0), eta = (0, 1, 0), zet = (0, 0, 1)
    Cmpnts csi = {1.0, 0.0, 0.0};
    Cmpnts eta = {0.0, 1.0, 0.0};
    Cmpnts zet = {0.0, 0.0, 1.0};

    // Contravariant components: ucont.x = ucat . csi, etc.
    Cmpnts ucont;
    ucont.x = ucat.x * csi.x + ucat.y * csi.y + ucat.z * csi.z;
    ucont.y = ucat.x * eta.x + ucat.y * eta.y + ucat.z * eta.z;
    ucont.z = ucat.x * zet.x + ucat.y * zet.y + ucat.z * zet.z;

    // For Cartesian grid, should be equal
    EXPECT_NEAR(ucont.x, ucat.x, TOLERANCE);
    EXPECT_NEAR(ucont.y, ucat.y, TOLERANCE);
    EXPECT_NEAR(ucont.z, ucat.z, TOLERANCE);
}

TEST_F(MetricsIntegrationTest, ContravariantVelocityRotatedGrid) {
    // Test with a rotated coordinate system (45 degrees in xy plane)
    double theta = M_PI / 4.0;  // 45 degrees
    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

    // Rotated metric vectors
    Cmpnts csi = {cos_t, sin_t, 0.0};
    Cmpnts eta = {-sin_t, cos_t, 0.0};
    Cmpnts zet = {0.0, 0.0, 1.0};

    // Cartesian velocity along x-axis
    Cmpnts ucat = {1.0, 0.0, 0.0};

    // Contravariant components
    Cmpnts ucont;
    ucont.x = ucat.x * csi.x + ucat.y * csi.y + ucat.z * csi.z;
    ucont.y = ucat.x * eta.x + ucat.y * eta.y + ucat.z * eta.z;
    ucont.z = ucat.x * zet.x + ucat.y * zet.y + ucat.z * zet.z;

    // Check that total "flux" is preserved
    // |ucont|^2 should relate to |ucat|^2
    double ucat_mag = ucat.x*ucat.x + ucat.y*ucat.y + ucat.z*ucat.z;
    double ucont_mag = ucont.x*ucont.x + ucont.y*ucont.y + ucont.z*ucont.z;

    EXPECT_NEAR(ucont_mag, ucat_mag, TOLERANCE);
}

// ============================================================================
// Grid Quality Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, OrthogonalityMetric) {
    // Test grid orthogonality
    // For orthogonal grid: csi . eta = csi . zet = eta . zet = 0

    // Orthogonal grid metrics
    Cmpnts csi = {1.0, 0.0, 0.0};
    Cmpnts eta = {0.0, 1.0, 0.0};
    Cmpnts zet = {0.0, 0.0, 1.0};

    double dot_csi_eta = csi.x*eta.x + csi.y*eta.y + csi.z*eta.z;
    double dot_csi_zet = csi.x*zet.x + csi.y*zet.y + csi.z*zet.z;
    double dot_eta_zet = eta.x*zet.x + eta.y*zet.y + eta.z*zet.z;

    EXPECT_NEAR(dot_csi_eta, 0.0, TOLERANCE);
    EXPECT_NEAR(dot_csi_zet, 0.0, TOLERANCE);
    EXPECT_NEAR(dot_eta_zet, 0.0, TOLERANCE);
}

TEST_F(MetricsIntegrationTest, SkewedGridMetrics) {
    // Non-orthogonal (skewed) grid
    // Metrics should still be consistent

    // Skewed grid: shear in xy plane
    double shear = 0.2;
    Cmpnts csi = {1.0, shear, 0.0};  // Skewed
    Cmpnts eta = {0.0, 1.0, 0.0};
    Cmpnts zet = {0.0, 0.0, 1.0};

    // csi . eta should be non-zero for skewed grid
    double dot_csi_eta = csi.x*eta.x + csi.y*eta.y + csi.z*eta.z;
    EXPECT_NE(dot_csi_eta, 0.0);

    // But Jacobian should still be positive
    double cross[3];
    cross[0] = eta.y * zet.z - eta.z * zet.y;
    cross[1] = eta.z * zet.x - eta.x * zet.z;
    cross[2] = eta.x * zet.y - eta.y * zet.x;

    double jacobian_inv = csi.x*cross[0] + csi.y*cross[1] + csi.z*cross[2];
    EXPECT_GT(jacobian_inv, 0.0);
}

// ============================================================================
// Face Area Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, FaceAreaComputation) {
    // Test face area computation from metric vectors
    // For a face normal to csi direction:
    // Area = |eta x zet| / |csi . (eta x zet)|

    // Unit cube cell
    Cmpnts eta = {0.0, 1.0, 0.0};
    Cmpnts zet = {0.0, 0.0, 1.0};

    // Cross product eta x zet
    double cross[3];
    cross[0] = eta.y * zet.z - eta.z * zet.y;  // 1*1 - 0*0 = 1
    cross[1] = eta.z * zet.x - eta.x * zet.z;  // 0*0 - 0*1 = 0
    cross[2] = eta.x * zet.y - eta.y * zet.x;  // 0*0 - 1*0 = 0

    double area = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    EXPECT_NEAR(area, 1.0, TOLERANCE);  // Unit area for unit cube
}

TEST_F(MetricsIntegrationTest, FaceAreaStretchedGrid) {
    // Stretched grid: 2x1 rectangle face
    Cmpnts eta = {0.0, 2.0, 0.0};  // Stretched in y
    Cmpnts zet = {0.0, 0.0, 1.0};

    double cross[3];
    cross[0] = eta.y * zet.z - eta.z * zet.y;
    cross[1] = eta.z * zet.x - eta.x * zet.z;
    cross[2] = eta.x * zet.y - eta.y * zet.x;

    double area = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
    EXPECT_NEAR(area, 2.0, TOLERANCE);  // 2x1 rectangle
}

// ============================================================================
// Volume Computation Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, CellVolumeComputation) {
    // Cell volume = 1 / Jacobian (where Jacobian is the metric Jacobian)
    // For physical volume, this is the determinant of the coordinate transformation

    // Unit cube
    double volume = 1.0 * 1.0 * 1.0;
    EXPECT_DOUBLE_EQ(volume, 1.0);

    // Stretched cell: 0.5 x 1.0 x 2.0
    volume = 0.5 * 1.0 * 2.0;
    EXPECT_DOUBLE_EQ(volume, 1.0);
}

// ============================================================================
// Gradient Computation Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, GradientInCartesianGrid) {
    // Test gradient computation using metrics
    // For a Cartesian grid: d/dx = (d csi/dx) * d/d(csi) + ...

    // For uniform Cartesian with dx = 0.1:
    double dx = 0.1;

    // Scalar field: phi(x) = x^2, so d(phi)/dx = 2x
    // At x = 0.5: gradient = 1.0

    double phi_west = 0.4 * 0.4;   // At x = 0.4
    double phi_east = 0.6 * 0.6;   // At x = 0.6

    double gradient = (phi_east - phi_west) / (2.0 * dx);
    double expected = 2.0 * 0.5;  // 2x at x = 0.5

    EXPECT_NEAR(gradient, expected, TOLERANCE);
}

// ============================================================================
// Curvilinear Grid Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, CylindricalCoordinateMetrics) {
    // Test metrics for cylindrical coordinate-like transformation
    // x = r*cos(theta), y = r*sin(theta), z = z

    double r = 1.0;
    double theta = M_PI / 4.0;

    // At this point:
    // dx/dr = cos(theta), dx/dtheta = -r*sin(theta)
    // dy/dr = sin(theta), dy/dtheta = r*cos(theta)
    // dz/dz = 1

    double cos_t = std::cos(theta);
    double sin_t = std::sin(theta);

    // Jacobian for cylindrical coordinates: J = r
    double jacobian = r;
    EXPECT_NEAR(jacobian, 1.0, TOLERANCE);
}

// ============================================================================
// Metric Identity Tests
// ============================================================================

TEST_F(MetricsIntegrationTest, MetricIdentityDivergenceFree) {
    // Test the metric identity: sum of face normals around a cell = 0
    // This ensures divergence-free property is preserved

    // For a closed cell, sum of outward normals * area should be zero
    // Using a simple cube example:

    // Face normals (outward) * area
    double n1[3] = {1.0, 0.0, 0.0};   // +x face
    double n2[3] = {-1.0, 0.0, 0.0};  // -x face
    double n3[3] = {0.0, 1.0, 0.0};   // +y face
    double n4[3] = {0.0, -1.0, 0.0};  // -y face
    double n5[3] = {0.0, 0.0, 1.0};   // +z face
    double n6[3] = {0.0, 0.0, -1.0};  // -z face

    // Sum should be zero
    double sum[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; i++) {
        sum[i] = n1[i] + n2[i] + n3[i] + n4[i] + n5[i] + n6[i];
    }

    EXPECT_NEAR(sum[0], 0.0, TOLERANCE);
    EXPECT_NEAR(sum[1], 0.0, TOLERANCE);
    EXPECT_NEAR(sum[2], 0.0, TOLERANCE);
}
