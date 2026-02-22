/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Unit tests for computational geometry functions (compgeom.c)   *
 ******************************************************************/

#include <gtest/gtest.h>
#include <cmath>

// Need to include PETSc first for proper type definitions
#include "petsc.h"
#include "variables.h"

// Functions from compgeom.c - already declared in variables.h
// tri_area and ISInsideTriangle2D are not declared in variables.h, so declare them here
double tri_area(double x1, double y1, double z1,
                double x2, double y2, double z2,
                double x3, double y3, double z3);

PetscInt ISInsideTriangle2D(Cpt2D p, Cpt2D pa, Cpt2D pb, Cpt2D pc);

class CompGeomTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize PETSc once for all tests
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
// Triangle Area Tests
// ============================================================================

TEST_F(CompGeomTest, TriArea_UnitTriangleInXYPlane) {
    // Right triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
    // Area should be 0.5
    double area = tri_area(0, 0, 0,  1, 0, 0,  0, 1, 0);
    EXPECT_NEAR(area, 0.5, TOLERANCE);
}

TEST_F(CompGeomTest, TriArea_EquilateralTriangle) {
    // Equilateral triangle with side length 2
    // Vertices: (0,0,0), (2,0,0), (1, sqrt(3), 0)
    // Area = (sqrt(3)/4) * s^2 = sqrt(3)
    double area = tri_area(0, 0, 0,  2, 0, 0,  1, std::sqrt(3.0), 0);
    EXPECT_NEAR(area, std::sqrt(3.0), TOLERANCE);
}

TEST_F(CompGeomTest, TriArea_TriangleIn3D) {
    // Triangle in 3D space
    // Vertices: (0,0,0), (1,0,0), (0,0,1)
    // Area should be 0.5 (right triangle in XZ plane)
    double area = tri_area(0, 0, 0,  1, 0, 0,  0, 0, 1);
    EXPECT_NEAR(area, 0.5, TOLERANCE);
}

TEST_F(CompGeomTest, TriArea_DegenerateTriangle) {
    // Collinear points - degenerate triangle with zero area
    double area = tri_area(0, 0, 0,  1, 0, 0,  2, 0, 0);
    EXPECT_NEAR(area, 0.0, TOLERANCE);
}

TEST_F(CompGeomTest, TriArea_ScaledTriangle) {
    // Scale triangle by factor of 3 - area should scale by 9
    double area1 = tri_area(0, 0, 0,  1, 0, 0,  0, 1, 0);
    double area2 = tri_area(0, 0, 0,  3, 0, 0,  0, 3, 0);
    EXPECT_NEAR(area2, 9.0 * area1, TOLERANCE);
}

// ============================================================================
// Point-to-Line Distance Tests
// ============================================================================

TEST_F(CompGeomTest, DisPLine_PointOnLine) {
    Cmpnts p = {0.5, 0.0, 0.0};
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts po;
    PetscReal d;

    Dis_P_Line(p, p1, p2, &po, &d);

    EXPECT_NEAR(d, 0.0, TOLERANCE);
    EXPECT_NEAR(po.x, 0.5, TOLERANCE);
    EXPECT_NEAR(po.y, 0.0, TOLERANCE);
    EXPECT_NEAR(po.z, 0.0, TOLERANCE);
}

TEST_F(CompGeomTest, DisPLine_PointPerpendicularToLine) {
    Cmpnts p = {0.5, 1.0, 0.0};  // Point above midpoint of line
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts po;
    PetscReal d;

    Dis_P_Line(p, p1, p2, &po, &d);

    EXPECT_NEAR(d, 1.0, TOLERANCE);
    EXPECT_NEAR(po.x, 0.5, TOLERANCE);
    EXPECT_NEAR(po.y, 0.0, TOLERANCE);
    EXPECT_NEAR(po.z, 0.0, TOLERANCE);
}

TEST_F(CompGeomTest, DisPLine_PointCloserToP1) {
    Cmpnts p = {-1.0, 0.0, 0.0};  // Point before line start
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts po;
    PetscReal d;

    Dis_P_Line(p, p1, p2, &po, &d);

    EXPECT_NEAR(d, 1.0, TOLERANCE);
    EXPECT_NEAR(po.x, 0.0, TOLERANCE);  // Closest point is p1
    EXPECT_NEAR(po.y, 0.0, TOLERANCE);
    EXPECT_NEAR(po.z, 0.0, TOLERANCE);
}

TEST_F(CompGeomTest, DisPLine_PointCloserToP2) {
    Cmpnts p = {2.0, 0.0, 0.0};  // Point after line end
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts po;
    PetscReal d;

    Dis_P_Line(p, p1, p2, &po, &d);

    EXPECT_NEAR(d, 1.0, TOLERANCE);
    EXPECT_NEAR(po.x, 1.0, TOLERANCE);  // Closest point is p2
    EXPECT_NEAR(po.y, 0.0, TOLERANCE);
    EXPECT_NEAR(po.z, 0.0, TOLERANCE);
}

TEST_F(CompGeomTest, DisPLine_3DCase) {
    Cmpnts p = {0.5, 0.5, 0.5};
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts po;
    PetscReal d;

    Dis_P_Line(p, p1, p2, &po, &d);

    // Distance should be sqrt(0.5^2 + 0.5^2) = sqrt(0.5)
    EXPECT_NEAR(d, std::sqrt(0.5), TOLERANCE);
    EXPECT_NEAR(po.x, 0.5, TOLERANCE);
    EXPECT_NEAR(po.y, 0.0, TOLERANCE);
    EXPECT_NEAR(po.z, 0.0, TOLERANCE);
}

// ============================================================================
// Ray-Triangle Intersection Tests
// ============================================================================

TEST_F(CompGeomTest, IntsectTriangle_RayHitsTriangle) {
    // Ray from (0.25, 0.25, -1) in direction (0, 0, 1) should hit triangle
    PetscReal orig[3] = {0.25, 0.25, -1.0};
    PetscReal dir[3] = {0.0, 0.0, 1.0};
    PetscReal vert0[3] = {0.0, 0.0, 0.0};
    PetscReal vert1[3] = {1.0, 0.0, 0.0};
    PetscReal vert2[3] = {0.0, 1.0, 0.0};
    PetscReal t, u, v;

    PetscInt result = intsect_triangle(orig, dir, vert0, vert1, vert2, &t, &u, &v);

    EXPECT_EQ(result, 1);
    EXPECT_NEAR(t, 1.0, TOLERANCE);  // Distance to intersection
    EXPECT_NEAR(u, 0.25, TOLERANCE); // Barycentric coordinate
    EXPECT_NEAR(v, 0.25, TOLERANCE); // Barycentric coordinate
}

TEST_F(CompGeomTest, IntsectTriangle_RayMissesTriangle) {
    // Ray from (2, 2, -1) in direction (0, 0, 1) should miss triangle
    PetscReal orig[3] = {2.0, 2.0, -1.0};
    PetscReal dir[3] = {0.0, 0.0, 1.0};
    PetscReal vert0[3] = {0.0, 0.0, 0.0};
    PetscReal vert1[3] = {1.0, 0.0, 0.0};
    PetscReal vert2[3] = {0.0, 1.0, 0.0};
    PetscReal t, u, v;

    PetscInt result = intsect_triangle(orig, dir, vert0, vert1, vert2, &t, &u, &v);

    EXPECT_EQ(result, 0);
}

TEST_F(CompGeomTest, IntsectTriangle_RayParallelToTriangle) {
    // Ray parallel to triangle plane should not intersect
    PetscReal orig[3] = {0.0, 0.0, 1.0};
    PetscReal dir[3] = {1.0, 0.0, 0.0};  // Parallel to XY plane
    PetscReal vert0[3] = {0.0, 0.0, 0.0};
    PetscReal vert1[3] = {1.0, 0.0, 0.0};
    PetscReal vert2[3] = {0.0, 1.0, 0.0};
    PetscReal t, u, v;

    PetscInt result = intsect_triangle(orig, dir, vert0, vert1, vert2, &t, &u, &v);

    EXPECT_EQ(result, 0);
}

TEST_F(CompGeomTest, IntsectTriangle_RayHitsVertex) {
    // Ray aimed at vertex 0
    PetscReal orig[3] = {0.0, 0.0, -1.0};
    PetscReal dir[3] = {0.0, 0.0, 1.0};
    PetscReal vert0[3] = {0.0, 0.0, 0.0};
    PetscReal vert1[3] = {1.0, 0.0, 0.0};
    PetscReal vert2[3] = {0.0, 1.0, 0.0};
    PetscReal t, u, v;

    PetscInt result = intsect_triangle(orig, dir, vert0, vert1, vert2, &t, &u, &v);

    EXPECT_EQ(result, 1);
    EXPECT_NEAR(u, 0.0, TOLERANCE);
    EXPECT_NEAR(v, 0.0, TOLERANCE);
}

// ============================================================================
// Point in Triangle Tests (2D)
// ============================================================================

TEST_F(CompGeomTest, ISInsideTriangle2D_PointInside) {
    Cpt2D p = {0.25, 0.25};
    Cpt2D pa = {0.0, 0.0};
    Cpt2D pb = {1.0, 0.0};
    Cpt2D pc = {0.0, 1.0};

    PetscInt result = ISInsideTriangle2D(p, pa, pb, pc);

    EXPECT_GT(result, 0);
}

TEST_F(CompGeomTest, ISInsideTriangle2D_PointOutside) {
    Cpt2D p = {1.0, 1.0};  // Outside the triangle
    Cpt2D pa = {0.0, 0.0};
    Cpt2D pb = {1.0, 0.0};
    Cpt2D pc = {0.0, 1.0};

    PetscInt result = ISInsideTriangle2D(p, pa, pb, pc);

    EXPECT_LT(result, 0);
}

TEST_F(CompGeomTest, ISInsideTriangle2D_PointOnEdge) {
    Cpt2D p = {0.5, 0.0};  // On edge pa-pb
    Cpt2D pa = {0.0, 0.0};
    Cpt2D pb = {1.0, 0.0};
    Cpt2D pc = {0.0, 1.0};

    PetscInt result = ISInsideTriangle2D(p, pa, pb, pc);

    // Point on edge should be considered inside
    EXPECT_GT(result, 0);
}

TEST_F(CompGeomTest, ISInsideTriangle2D_PointAtVertex) {
    Cpt2D p = {0.0, 0.0};  // At vertex pa
    Cpt2D pa = {0.0, 0.0};
    Cpt2D pb = {1.0, 0.0};
    Cpt2D pc = {0.0, 1.0};

    PetscInt result = ISInsideTriangle2D(p, pa, pb, pc);

    // Point at vertex should be considered inside
    EXPECT_GT(result, 0);
}

// ============================================================================
// 3D Triangle Interpolation Tests
// ============================================================================

TEST_F(CompGeomTest, TriangleIntp3D_Centroid) {
    IBMInfo ibminfo;

    // Point at centroid of triangle
    double x = 1.0/3.0, y = 1.0/3.0, z = 0.0;
    double x1 = 0.0, y1 = 0.0, z1 = 0.0;
    double x2 = 1.0, y2 = 0.0, z2 = 0.0;
    double x3 = 0.0, y3 = 1.0, z3 = 0.0;

    triangle_intp3D(x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3, &ibminfo);

    // At centroid, all weights should be equal to 1/3
    EXPECT_NEAR(ibminfo.cr1, 1.0/3.0, 0.01);
    EXPECT_NEAR(ibminfo.cr2, 1.0/3.0, 0.01);
    EXPECT_NEAR(ibminfo.cr3, 1.0/3.0, 0.01);

    // Weights should sum to 1
    EXPECT_NEAR(ibminfo.cr1 + ibminfo.cr2 + ibminfo.cr3, 1.0, TOLERANCE);
}

TEST_F(CompGeomTest, TriangleIntp3D_AtVertex) {
    IBMInfo ibminfo;

    // Point at vertex 1
    double x = 0.0, y = 0.0, z = 0.0;
    double x1 = 0.0, y1 = 0.0, z1 = 0.0;
    double x2 = 1.0, y2 = 0.0, z2 = 0.0;
    double x3 = 0.0, y3 = 1.0, z3 = 0.0;

    triangle_intp3D(x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3, &ibminfo);

    // At vertex 1, weight 1 should be 1, others 0
    EXPECT_NEAR(ibminfo.cr1, 1.0, 0.01);
    EXPECT_NEAR(ibminfo.cr2, 0.0, 0.01);
    EXPECT_NEAR(ibminfo.cr3, 0.0, 0.01);
}

TEST_F(CompGeomTest, TriangleIntp3D_MidpointOfEdge) {
    IBMInfo ibminfo;

    // Point at midpoint of edge v1-v2
    double x = 0.5, y = 0.0, z = 0.0;
    double x1 = 0.0, y1 = 0.0, z1 = 0.0;
    double x2 = 1.0, y2 = 0.0, z2 = 0.0;
    double x3 = 0.0, y3 = 1.0, z3 = 0.0;

    triangle_intp3D(x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3, &ibminfo);

    // At midpoint of v1-v2, weights 1 and 2 should be 0.5, weight 3 should be 0
    EXPECT_NEAR(ibminfo.cr1, 0.5, 0.01);
    EXPECT_NEAR(ibminfo.cr2, 0.5, 0.01);
    EXPECT_NEAR(ibminfo.cr3, 0.0, 0.01);
}

// ============================================================================
// Point in Triangle Tests (3D)
// ============================================================================

TEST_F(CompGeomTest, ISPointInTriangle_PointInside) {
    Cmpnts p = {0.25, 0.25, 0.0};
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts p3 = {0.0, 1.0, 0.0};

    // Normal vector pointing in Z direction
    PetscReal nfx = 0.0, nfy = 0.0, nfz = 1.0;

    PetscInt result = ISPointInTriangle(p, p1, p2, p3, nfx, nfy, nfz);

    EXPECT_GT(result, 0);
}

TEST_F(CompGeomTest, ISPointInTriangle_PointOutside) {
    Cmpnts p = {1.0, 1.0, 0.0};  // Outside
    Cmpnts p1 = {0.0, 0.0, 0.0};
    Cmpnts p2 = {1.0, 0.0, 0.0};
    Cmpnts p3 = {0.0, 1.0, 0.0};

    PetscReal nfx = 0.0, nfy = 0.0, nfz = 1.0;

    PetscInt result = ISPointInTriangle(p, p1, p2, p3, nfx, nfy, nfz);

    EXPECT_LT(result, 0);
}
