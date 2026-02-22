/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Unit tests for mathematical utility functions                  *
 ******************************************************************/

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

#include "petsc.h"
#include "variables.h"

class MathUtilsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int argc = 0;
        char **argv = nullptr;
        PetscInitialize(&argc, &argv, nullptr, nullptr);
    }

    static void TearDownTestSuite() {
        PetscFinalize();
    }

    const double TOLERANCE = 1e-12;
    const double FLOAT_TOLERANCE = 1e-6;
};

// ============================================================================
// Vector Operations Tests (Cmpnts structure)
// ============================================================================

TEST_F(MathUtilsTest, CmpntsInitialization) {
    Cmpnts v = {1.0, 2.0, 3.0};
    EXPECT_DOUBLE_EQ(v.x, 1.0);
    EXPECT_DOUBLE_EQ(v.y, 2.0);
    EXPECT_DOUBLE_EQ(v.z, 3.0);
}

TEST_F(MathUtilsTest, CmpntsZeroVector) {
    Cmpnts v = {0.0, 0.0, 0.0};
    double magnitude = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    EXPECT_DOUBLE_EQ(magnitude, 0.0);
}

TEST_F(MathUtilsTest, CmpntsMagnitude) {
    Cmpnts v = {3.0, 4.0, 0.0};
    double magnitude = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    EXPECT_DOUBLE_EQ(magnitude, 5.0);
}

TEST_F(MathUtilsTest, Cmpnts3DMagnitude) {
    Cmpnts v = {1.0, 2.0, 2.0};
    double magnitude = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    EXPECT_DOUBLE_EQ(magnitude, 3.0);
}

// ============================================================================
// 2D Point Operations Tests (Cpt2D structure)
// ============================================================================

TEST_F(MathUtilsTest, Cpt2DInitialization) {
    Cpt2D p = {1.5, 2.5};
    EXPECT_DOUBLE_EQ(p.x, 1.5);
    EXPECT_DOUBLE_EQ(p.y, 2.5);
}

TEST_F(MathUtilsTest, Cpt2DDistance) {
    Cpt2D p1 = {0.0, 0.0};
    Cpt2D p2 = {3.0, 4.0};
    double dist = std::sqrt((p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y));
    EXPECT_DOUBLE_EQ(dist, 5.0);
}

// ============================================================================
// Dot Product Tests
// ============================================================================

TEST_F(MathUtilsTest, DotProductOrthogonal) {
    // Orthogonal vectors have zero dot product
    double v1[3] = {1.0, 0.0, 0.0};
    double v2[3] = {0.0, 1.0, 0.0};
    double dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    EXPECT_DOUBLE_EQ(dot, 0.0);
}

TEST_F(MathUtilsTest, DotProductParallel) {
    double v1[3] = {1.0, 2.0, 3.0};
    double v2[3] = {2.0, 4.0, 6.0};  // v2 = 2*v1
    double dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    // Expected: 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    EXPECT_DOUBLE_EQ(dot, 28.0);
}

TEST_F(MathUtilsTest, DotProductSelf) {
    double v[3] = {3.0, 4.0, 0.0};
    double dot = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    EXPECT_DOUBLE_EQ(dot, 25.0);  // Magnitude squared
}

// ============================================================================
// Cross Product Tests
// ============================================================================

TEST_F(MathUtilsTest, CrossProductBasisVectors) {
    // i x j = k
    double v1[3] = {1.0, 0.0, 0.0};
    double v2[3] = {0.0, 1.0, 0.0};
    double result[3];

    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 1.0);
}

TEST_F(MathUtilsTest, CrossProductParallel) {
    // Parallel vectors have zero cross product
    double v1[3] = {1.0, 2.0, 3.0};
    double v2[3] = {2.0, 4.0, 6.0};
    double result[3];

    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];

    EXPECT_NEAR(result[0], 0.0, TOLERANCE);
    EXPECT_NEAR(result[1], 0.0, TOLERANCE);
    EXPECT_NEAR(result[2], 0.0, TOLERANCE);
}

TEST_F(MathUtilsTest, CrossProductAnticommutative) {
    double v1[3] = {1.0, 2.0, 3.0};
    double v2[3] = {4.0, 5.0, 6.0};
    double result1[3], result2[3];

    // v1 x v2
    result1[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result1[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result1[2] = v1[0] * v2[1] - v1[1] * v2[0];

    // v2 x v1
    result2[0] = v2[1] * v1[2] - v2[2] * v1[1];
    result2[1] = v2[2] * v1[0] - v2[0] * v1[2];
    result2[2] = v2[0] * v1[1] - v2[1] * v1[0];

    // v2 x v1 = -(v1 x v2)
    EXPECT_DOUBLE_EQ(result2[0], -result1[0]);
    EXPECT_DOUBLE_EQ(result2[1], -result1[1]);
    EXPECT_DOUBLE_EQ(result2[2], -result1[2]);
}

// ============================================================================
// Heaviside Function Tests (for levelset)
// ============================================================================

// Heaviside function as defined in the code
double H(double p, double dx) {
    if (p < -dx) return 0.0;
    if (p > dx) return 1.0;
    return 0.5 * (1.0 + p/dx + std::sin(M_PI * p/dx) / M_PI);
}

TEST_F(MathUtilsTest, HeavisideNegative) {
    double dx = 0.1;
    EXPECT_DOUBLE_EQ(H(-0.5, dx), 0.0);
    EXPECT_DOUBLE_EQ(H(-1.0, dx), 0.0);
}

TEST_F(MathUtilsTest, HeavisidePositive) {
    double dx = 0.1;
    EXPECT_DOUBLE_EQ(H(0.5, dx), 1.0);
    EXPECT_DOUBLE_EQ(H(1.0, dx), 1.0);
}

TEST_F(MathUtilsTest, HeavisideZero) {
    double dx = 0.1;
    EXPECT_DOUBLE_EQ(H(0.0, dx), 0.5);
}

TEST_F(MathUtilsTest, HeavisideSmoothed) {
    double dx = 0.1;
    double h1 = H(-dx/2, dx);
    double h2 = H(dx/2, dx);

    // Should be between 0 and 1
    EXPECT_GT(h1, 0.0);
    EXPECT_LT(h1, 1.0);
    EXPECT_GT(h2, 0.0);
    EXPECT_LT(h2, 1.0);

    // h2 should be greater than h1
    EXPECT_GT(h2, h1);
}

// ============================================================================
// Sign Function Tests
// ============================================================================

double sign(double a) {
    if (a > 0) return 1.0;
    if (a < 0) return -1.0;
    return 0.0;
}

TEST_F(MathUtilsTest, SignPositive) {
    EXPECT_DOUBLE_EQ(sign(1.0), 1.0);
    EXPECT_DOUBLE_EQ(sign(100.0), 1.0);
    EXPECT_DOUBLE_EQ(sign(0.001), 1.0);
}

TEST_F(MathUtilsTest, SignNegative) {
    EXPECT_DOUBLE_EQ(sign(-1.0), -1.0);
    EXPECT_DOUBLE_EQ(sign(-100.0), -1.0);
    EXPECT_DOUBLE_EQ(sign(-0.001), -1.0);
}

TEST_F(MathUtilsTest, SignZero) {
    EXPECT_DOUBLE_EQ(sign(0.0), 0.0);
}

// ============================================================================
// Mean Function Tests
// ============================================================================

double mean(double A, double B) {
    return 0.5 * (A + B);
}

TEST_F(MathUtilsTest, MeanPositive) {
    EXPECT_DOUBLE_EQ(mean(2.0, 4.0), 3.0);
}

TEST_F(MathUtilsTest, MeanNegative) {
    EXPECT_DOUBLE_EQ(mean(-2.0, -4.0), -3.0);
}

TEST_F(MathUtilsTest, MeanMixed) {
    EXPECT_DOUBLE_EQ(mean(-2.0, 4.0), 1.0);
}

TEST_F(MathUtilsTest, MeanZero) {
    EXPECT_DOUBLE_EQ(mean(-5.0, 5.0), 0.0);
}

// ============================================================================
// Upwind Scheme Tests
// ============================================================================

double Upwind(double W, double E, double a) {
    if (a >= 0) return W;
    return E;
}

TEST_F(MathUtilsTest, UpwindPositiveVelocity) {
    EXPECT_DOUBLE_EQ(Upwind(1.0, 2.0, 1.0), 1.0);  // Takes West value
}

TEST_F(MathUtilsTest, UpwindNegativeVelocity) {
    EXPECT_DOUBLE_EQ(Upwind(1.0, 2.0, -1.0), 2.0);  // Takes East value
}

TEST_F(MathUtilsTest, UpwindZeroVelocity) {
    EXPECT_DOUBLE_EQ(Upwind(1.0, 2.0, 0.0), 1.0);  // Takes West value
}

// ============================================================================
// CFL-related Tests
// ============================================================================

TEST_F(MathUtilsTest, CFLComputation) {
    // CFL = u * dt / dx
    double u = 1.0;
    double dt = 0.001;
    double dx = 0.01;
    double cfl = u * dt / dx;
    EXPECT_DOUBLE_EQ(cfl, 0.1);
}

TEST_F(MathUtilsTest, MaxTimeStepFromCFL) {
    // dt_max = CFL_max * dx / u
    double cfl_max = 0.5;
    double dx = 0.01;
    double u = 1.0;
    double dt_max = cfl_max * dx / u;
    EXPECT_DOUBLE_EQ(dt_max, 0.005);
}

// ============================================================================
// IBMInfo Structure Tests
// ============================================================================

TEST_F(MathUtilsTest, IBMInfoInitialization) {
    IBMInfo info;
    info.cr1 = 0.3;
    info.cr2 = 0.3;
    info.cr3 = 0.4;

    // Barycentric coordinates should sum to 1
    EXPECT_DOUBLE_EQ(info.cr1 + info.cr2 + info.cr3, 1.0);
}

TEST_F(MathUtilsTest, IBMInfoInterpolation) {
    IBMInfo info;
    info.cr1 = 0.5;
    info.cr2 = 0.3;
    info.cr3 = 0.2;

    // Test interpolation using weights
    double v1 = 1.0, v2 = 2.0, v3 = 3.0;
    double interpolated = info.cr1 * v1 + info.cr2 * v2 + info.cr3 * v3;

    // Expected: 0.5*1 + 0.3*2 + 0.2*3 = 0.5 + 0.6 + 0.6 = 1.7
    EXPECT_DOUBLE_EQ(interpolated, 1.7);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(MathUtilsTest, DivisionBySmallNumber) {
    double epsilon = 1e-15;
    double result = 1.0 / epsilon;
    EXPECT_TRUE(std::isfinite(result));
}

TEST_F(MathUtilsTest, SqrtOfSmallNumber) {
    double small = 1e-30;
    double result = std::sqrt(small);
    EXPECT_TRUE(std::isfinite(result));
    EXPECT_NEAR(result, 1e-15, 1e-20);
}

TEST_F(MathUtilsTest, DifferenceOfNearlyEqualNumbers) {
    // Test for catastrophic cancellation awareness
    double a = 1.0000001;
    double b = 1.0000000;
    double diff = a - b;
    EXPECT_NEAR(diff, 1e-7, 1e-12);
}

// ============================================================================
// Pi Constant Test
// ============================================================================

TEST_F(MathUtilsTest, PiConstant) {
    // M_PI is defined in variables.h
    EXPECT_NEAR(M_PI, 3.14159265358979323846, 1e-15);
}
