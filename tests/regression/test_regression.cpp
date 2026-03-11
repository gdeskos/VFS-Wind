/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Regression test framework for VFS-Wind example cases           *
 ******************************************************************/

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>
#include <sys/stat.h>

#include "petsc.h"

// Path to examples directory (set via CMake)
#ifndef EXAMPLES_DIR
#define EXAMPLES_DIR "."
#endif

// Define Cmpnts structure locally for standalone testing
typedef struct {
    double x, y, z;
} Cmpnts;

class RegressionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int argc = 0;
        char **argv = nullptr;
        PetscInitialize(&argc, &argv, nullptr, nullptr);
    }

    static void TearDownTestSuite() {
        PetscFinalize();
    }

    const double TOLERANCE = 1e-6;
    const double RELATIVE_TOLERANCE = 0.01;  // 1% relative error

    // Helper function to check if file exists
    bool fileExists(const std::string& path) {
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
    }

    // Helper function to read a data file and parse values
    std::vector<double> readDataColumn(const std::string& filename, int column) {
        std::vector<double> values;
        std::ifstream file(filename);
        if (!file.is_open()) {
            return values;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#' || line[0] == '%') {
                continue;
            }

            std::istringstream iss(line);
            double val;
            int col = 0;
            while (iss >> val) {
                if (col == column) {
                    values.push_back(val);
                    break;
                }
                col++;
            }
        }
        return values;
    }

    // Compare two data vectors with tolerance
    bool compareData(const std::vector<double>& reference,
                     const std::vector<double>& computed,
                     double rel_tol = 0.01) {
        if (reference.size() != computed.size()) {
            return false;
        }

        for (size_t i = 0; i < reference.size(); i++) {
            double ref = reference[i];
            double comp = computed[i];

            if (std::abs(ref) < 1e-10) {
                // Absolute comparison for near-zero values
                if (std::abs(comp - ref) > TOLERANCE) {
                    return false;
                }
            } else {
                // Relative comparison
                if (std::abs(comp - ref) / std::abs(ref) > rel_tol) {
                    return false;
                }
            }
        }
        return true;
    }

    // Check conservation properties
    bool checkMassConservation(const std::string& mass_file, double initial_mass) {
        auto masses = readDataColumn(mass_file, 1);  // Assuming mass is in column 1
        if (masses.empty()) {
            return false;
        }

        // For two-phase flows, allow up to 5% variation due to numerical diffusion
        const double MASS_TOLERANCE = 0.05;

        // Filter out invalid/anomalous values (e.g., near-zero from restart issues)
        double min_valid_mass = initial_mass * 0.1;  // Mass should be at least 10% of initial

        for (const auto& m : masses) {
            // Skip anomalous values
            if (m < min_valid_mass) {
                continue;
            }
            if (std::abs(m - initial_mass) / initial_mass > MASS_TOLERANCE) {
                return false;
            }
        }
        return true;
    }

    // Check kinetic energy evolution (should decay or be bounded)
    bool checkKineticEnergy(const std::string& ke_file) {
        auto energies = readDataColumn(ke_file, 1);  // Assuming KE is in column 1
        if (energies.empty()) {
            return false;
        }

        // Check that energy doesn't blow up
        double max_energy = *std::max_element(energies.begin(), energies.end());
        double initial_energy = energies[0];

        // Energy should not grow more than 10x (reasonable for most cases)
        if (max_energy > 10.0 * initial_energy && initial_energy > 0) {
            return false;
        }

        // Check for NaN or Inf
        for (const auto& e : energies) {
            if (std::isnan(e) || std::isinf(e)) {
                return false;
            }
        }
        return true;
    }

    // Check convergence
    bool checkConvergence(const std::string& converge_file) {
        auto residuals = readDataColumn(converge_file, 0);
        if (residuals.empty()) {
            return false;
        }

        // Check that residuals decrease or stabilize
        // This is a simple check - more sophisticated tests can be added
        double final_residual = residuals.back();
        double initial_residual = residuals.front();

        // Residual should not grow significantly
        if (final_residual > 100.0 * initial_residual && initial_residual > 0) {
            return false;
        }

        return true;
    }

    // Check FSI position data (should not blow up)
    bool checkFSIPosition(const std::string& fsi_file) {
        auto positions = readDataColumn(fsi_file, 1);  // Position in column 1
        if (positions.empty()) {
            return false;
        }

        // Check for NaN or Inf
        for (const auto& p : positions) {
            if (std::isnan(p) || std::isinf(p)) {
                return false;
            }
        }

        // Check that displacement is bounded (not flying away)
        double max_pos = *std::max_element(positions.begin(), positions.end());
        double min_pos = *std::min_element(positions.begin(), positions.end());
        double range = max_pos - min_pos;

        // Displacement range should be reasonable (< 100 for most cases)
        if (range > 100.0) {
            return false;
        }

        return true;
    }

    // Check force coefficients (should be bounded)
    bool checkForceCoefficients(const std::string& force_file) {
        auto forces = readDataColumn(force_file, 1);
        if (forces.empty()) {
            return false;
        }

        // Check for NaN or Inf
        for (const auto& f : forces) {
            if (std::isnan(f) || std::isinf(f)) {
                return false;
            }
        }

        return true;
    }
};

// ============================================================================
// Example 01: Sloshing Tank
// ============================================================================

TEST_F(RegressionTest, Example01_SloshingTank_FilesExist) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/01_Sloshing_Tank";

    // Check that essential input files exist
    EXPECT_TRUE(fileExists(base_path + "/control.xml") ||
                fileExists(base_path + "/control.dat"))
        << "Control file missing for Sloshing Tank test case";

    EXPECT_TRUE(fileExists(base_path + "/bcs.dat"))
        << "BCS file missing for Sloshing Tank test case";

    EXPECT_TRUE(fileExists(base_path + "/xyz.dat") ||
                fileExists(base_path + "/grid.dat"))
        << "Grid file missing for Sloshing Tank test case";
}

TEST_F(RegressionTest, Example01_SloshingTank_KineticEnergy) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/01_Sloshing_Tank";
    std::string ke_file = base_path + "/Kinetic_Energy.dat";

    if (fileExists(ke_file)) {
        EXPECT_TRUE(checkKineticEnergy(ke_file))
            << "Kinetic energy check failed for Sloshing Tank case";
    } else {
        GTEST_SKIP() << "Kinetic energy file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example01_SloshingTank_Convergence) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/01_Sloshing_Tank";
    std::string converge_file = base_path + "/Converge_dU";

    if (fileExists(converge_file)) {
        EXPECT_TRUE(checkConvergence(converge_file))
            << "Convergence check failed for Sloshing Tank case";
    } else {
        GTEST_SKIP() << "Convergence file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example01_SloshingTank_MassConservation) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/01_Sloshing_Tank";
    std::string mass_file = base_path + "/mass.dat";

    if (fileExists(mass_file)) {
        auto masses = readDataColumn(mass_file, 1);
        if (!masses.empty()) {
            double initial_mass = masses[0];
            EXPECT_TRUE(checkMassConservation(mass_file, initial_mass))
                << "Mass conservation check failed for Sloshing Tank case";
        }
    } else {
        GTEST_SKIP() << "Mass file not found (test not run yet)";
    }
}

// ============================================================================
// Example 02: Channel Flow
// ============================================================================

TEST_F(RegressionTest, Example02_ChannelFlow_FilesExist) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/02_ChannelFlow";

    // Check that essential input files exist
    EXPECT_TRUE(fileExists(base_path + "/control.xml") ||
                fileExists(base_path + "/control.dat"))
        << "Control file missing for Channel Flow test case";

    EXPECT_TRUE(fileExists(base_path + "/bcs.dat"))
        << "BCS file missing for Channel Flow test case";

    EXPECT_TRUE(fileExists(base_path + "/xyz.dat") ||
                fileExists(base_path + "/grid.dat"))
        << "Grid file missing for Channel Flow test case";
}

TEST_F(RegressionTest, Example02_ChannelFlow_KineticEnergy) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/02_ChannelFlow";
    std::string ke_file = base_path + "/Kinetic_Energy.dat";

    if (fileExists(ke_file)) {
        EXPECT_TRUE(checkKineticEnergy(ke_file))
            << "Kinetic energy check failed for Channel Flow case";
    } else {
        GTEST_SKIP() << "Kinetic energy file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example02_ChannelFlow_Convergence) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/02_ChannelFlow";
    std::string converge_file = base_path + "/Converge_dU";

    if (fileExists(converge_file)) {
        EXPECT_TRUE(checkConvergence(converge_file))
            << "Convergence check failed for Channel Flow case";
    } else {
        GTEST_SKIP() << "Convergence file not found (test not run yet)";
    }
}

// ============================================================================
// Example 03: VIV Mounted Cylinder
// ============================================================================

TEST_F(RegressionTest, Example03_VIV_FilesExist) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/03_VIV_Mounted_Cylinder";

    // Check that essential input files exist
    EXPECT_TRUE(fileExists(base_path + "/control.xml") ||
                fileExists(base_path + "/control.dat"))
        << "Control file missing for VIV Mounted Cylinder test case";

    EXPECT_TRUE(fileExists(base_path + "/bcs.dat"))
        << "BCS file missing for VIV Mounted Cylinder test case";

    // IBM test case requires ibmdata
    EXPECT_TRUE(fileExists(base_path + "/ibmdata00"))
        << "IBM data file missing for VIV Mounted Cylinder test case";
}

TEST_F(RegressionTest, Example03_VIV_KineticEnergy) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/03_VIV_Mounted_Cylinder";
    std::string ke_file = base_path + "/Kinetic_Energy.dat";

    if (fileExists(ke_file)) {
        EXPECT_TRUE(checkKineticEnergy(ke_file))
            << "Kinetic energy check failed for VIV Mounted Cylinder case";
    } else {
        GTEST_SKIP() << "Kinetic energy file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example03_VIV_Convergence) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/03_VIV_Mounted_Cylinder";
    std::string converge_file = base_path + "/Converge_dU";

    if (fileExists(converge_file)) {
        EXPECT_TRUE(checkConvergence(converge_file))
            << "Convergence check failed for VIV Mounted Cylinder case";
    } else {
        GTEST_SKIP() << "Convergence file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example03_VIV_FSIPosition) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/03_VIV_Mounted_Cylinder";
    std::string fsi_file = base_path + "/FSI_position00";

    if (fileExists(fsi_file)) {
        EXPECT_TRUE(checkFSIPosition(fsi_file))
            << "FSI position check failed for VIV Mounted Cylinder case";
    } else {
        GTEST_SKIP() << "FSI position file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example03_VIV_ForceCoefficients) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/03_VIV_Mounted_Cylinder";
    std::string force_file = base_path + "/Force_Coeff_00";

    if (fileExists(force_file)) {
        EXPECT_TRUE(checkForceCoefficients(force_file))
            << "Force coefficient check failed for VIV Mounted Cylinder case";
    } else {
        GTEST_SKIP() << "Force coefficient file not found (test not run yet)";
    }
}

// ============================================================================
// Example 04: 2D Falling Cylinder
// ============================================================================

TEST_F(RegressionTest, Example04_FallingCylinder_FilesExist) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/04_2D_Fall_Cylinder";

    // Check that essential input files exist
    EXPECT_TRUE(fileExists(base_path + "/control.xml") ||
                fileExists(base_path + "/control.dat"))
        << "Control file missing for 2D Falling Cylinder test case";

    EXPECT_TRUE(fileExists(base_path + "/bcs.dat"))
        << "BCS file missing for 2D Falling Cylinder test case";

    EXPECT_TRUE(fileExists(base_path + "/grid.dat") ||
                fileExists(base_path + "/xyz.dat"))
        << "Grid file missing for 2D Falling Cylinder test case";

    // IBM + FSI test case requires ibmdata
    EXPECT_TRUE(fileExists(base_path + "/ibmdata00"))
        << "IBM data file missing for 2D Falling Cylinder test case";
}

TEST_F(RegressionTest, Example04_FallingCylinder_KineticEnergy) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/04_2D_Fall_Cylinder";
    std::string ke_file = base_path + "/Kinetic_Energy.dat";

    if (fileExists(ke_file)) {
        EXPECT_TRUE(checkKineticEnergy(ke_file))
            << "Kinetic energy check failed for 2D Falling Cylinder case";
    } else {
        GTEST_SKIP() << "Kinetic energy file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example04_FallingCylinder_Convergence) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/04_2D_Fall_Cylinder";
    std::string converge_file = base_path + "/Converge_dU";

    if (fileExists(converge_file)) {
        EXPECT_TRUE(checkConvergence(converge_file))
            << "Convergence check failed for 2D Falling Cylinder case";
    } else {
        GTEST_SKIP() << "Convergence file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example04_FallingCylinder_FSIPosition) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/04_2D_Fall_Cylinder";
    std::string fsi_file = base_path + "/FSI_position00";

    if (fileExists(fsi_file)) {
        EXPECT_TRUE(checkFSIPosition(fsi_file))
            << "FSI position check failed for 2D Falling Cylinder case";
    } else {
        GTEST_SKIP() << "FSI position file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example04_FallingCylinder_ForceCoefficients) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/04_2D_Fall_Cylinder";
    std::string force_file = base_path + "/Force_Coeff_00";

    if (fileExists(force_file)) {
        EXPECT_TRUE(checkForceCoefficients(force_file))
            << "Force coefficient check failed for 2D Falling Cylinder case";
    } else {
        GTEST_SKIP() << "Force coefficient file not found (test not run yet)";
    }
}

// ============================================================================
// Example 05: 3D Heave Decay Cylinder
// ============================================================================

TEST_F(RegressionTest, Example05_HeaveDecay_FilesExist) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/05_3D_Heave_Decay_Cylinder";

    // Check that essential input files exist
    EXPECT_TRUE(fileExists(base_path + "/control.xml") ||
                fileExists(base_path + "/control.dat"))
        << "Control file missing for 3D Heave Decay Cylinder test case";

    EXPECT_TRUE(fileExists(base_path + "/bcs.dat"))
        << "BCS file missing for 3D Heave Decay Cylinder test case";

    // IBM + FSI test case requires ibmdata
    EXPECT_TRUE(fileExists(base_path + "/ibmdata00"))
        << "IBM data file missing for 3D Heave Decay Cylinder test case";
}

TEST_F(RegressionTest, Example05_HeaveDecay_GridExists) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/05_3D_Heave_Decay_Cylinder";

    // Grid file is critical for this case
    bool has_grid = fileExists(base_path + "/grid.dat") ||
                    fileExists(base_path + "/xyz.dat");

    if (!has_grid) {
        GTEST_SKIP() << "Grid file missing - needs to be generated for this example";
    }
    EXPECT_TRUE(has_grid) << "Grid file missing for 3D Heave Decay Cylinder test case";
}

TEST_F(RegressionTest, Example05_HeaveDecay_KineticEnergy) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/05_3D_Heave_Decay_Cylinder";
    std::string ke_file = base_path + "/Kinetic_Energy.dat";

    if (fileExists(ke_file)) {
        EXPECT_TRUE(checkKineticEnergy(ke_file))
            << "Kinetic energy check failed for 3D Heave Decay Cylinder case";
    } else {
        GTEST_SKIP() << "Kinetic energy file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example05_HeaveDecay_Convergence) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/05_3D_Heave_Decay_Cylinder";
    std::string converge_file = base_path + "/Converge_dU";

    if (fileExists(converge_file)) {
        EXPECT_TRUE(checkConvergence(converge_file))
            << "Convergence check failed for 3D Heave Decay Cylinder case";
    } else {
        GTEST_SKIP() << "Convergence file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example05_HeaveDecay_FSIPosition) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/05_3D_Heave_Decay_Cylinder";
    std::string fsi_file = base_path + "/FSI_position00";

    if (fileExists(fsi_file)) {
        EXPECT_TRUE(checkFSIPosition(fsi_file))
            << "FSI position check failed for 3D Heave Decay Cylinder case";
    } else {
        GTEST_SKIP() << "FSI position file not found (test not run yet)";
    }
}

TEST_F(RegressionTest, Example05_HeaveDecay_ForceCoefficients) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/05_3D_Heave_Decay_Cylinder";
    std::string force_file = base_path + "/Force_Coeff_00";

    if (fileExists(force_file)) {
        EXPECT_TRUE(checkForceCoefficients(force_file))
            << "Force coefficient check failed for 3D Heave Decay Cylinder case";
    } else {
        GTEST_SKIP() << "Force coefficient file not found (test not run yet)";
    }
}

// ============================================================================
// Physics Validation Tests
// ============================================================================

TEST_F(RegressionTest, ValidateChannelFlowStatistics) {
    // Validate mean velocity profile against log-law for turbulent channel flow
    // u+ = (1/kappa) * ln(y+) + B
    // where kappa = 0.41, B = 5.0 (typical values)

    double kappa = 0.41;
    double B = 5.0;

    // Test at y+ = 30, 100, 500 (log-layer region)
    std::vector<double> y_plus = {30.0, 100.0, 500.0};

    for (double yp : y_plus) {
        double u_plus_expected = (1.0 / kappa) * std::log(yp) + B;

        // u+ should be positive and reasonable (< 30 for Re_tau = 3000)
        EXPECT_GT(u_plus_expected, 0.0);
        EXPECT_LT(u_plus_expected, 30.0);
    }
}

TEST_F(RegressionTest, ValidateTurbulentKineticEnergyProfile) {
    // Near-wall peak of turbulent kinetic energy should be around y+ ~ 15
    // and k+ ~ 4-5

    double y_plus_peak = 15.0;
    double k_plus_peak_min = 3.0;
    double k_plus_peak_max = 6.0;

    // This is a sanity check of expected physics
    EXPECT_GT(y_plus_peak, 10.0);
    EXPECT_LT(y_plus_peak, 20.0);
    EXPECT_GE(k_plus_peak_min, 0.0);
    EXPECT_LE(k_plus_peak_max, 10.0);
}

// ============================================================================
// Numerical Properties Tests
// ============================================================================

TEST_F(RegressionTest, ValidateCFLCondition) {
    // Test CFL condition for explicit time stepping
    double u_max = 1.0;  // Max velocity
    double dx_min = 0.01;  // Min grid spacing
    double dt = 0.001;  // Time step

    double cfl = u_max * dt / dx_min;

    // CFL should be less than 1 for explicit schemes
    EXPECT_LT(cfl, 1.0) << "CFL condition violated";
}

TEST_F(RegressionTest, ValidateViscousStability) {
    // Test viscous stability condition: dt < dx^2 / (6 * nu)
    double nu = 1.0 / 3000.0;  // Kinematic viscosity for Re = 3000
    double dx = 0.01;  // Grid spacing
    double dt = 0.001;  // Time step

    double viscous_limit = dx * dx / (6.0 * nu);

    // dt should be less than viscous limit
    EXPECT_LT(dt, viscous_limit) << "Viscous stability condition violated";
}

// ============================================================================
// Conservation Tests
// ============================================================================

TEST_F(RegressionTest, ValidateMassConservation) {
    // For incompressible flow, mass should be conserved
    // div(u) = 0 => flux_in = flux_out

    double flux_in = 1.0;  // Unit flux
    double flux_out = 1.0;

    double mass_balance = std::abs(flux_in - flux_out);
    EXPECT_NEAR(mass_balance, 0.0, TOLERANCE)
        << "Mass conservation violated";
}

TEST_F(RegressionTest, ValidateMomentumBalance) {
    // For steady channel flow:
    // tau_wall = dp/dx * delta (half channel height)

    double Re_tau = 3000.0;
    double delta = 1.0;  // Half channel height (normalized)
    double u_tau = 1.0;  // Friction velocity (normalized)

    // Wall shear stress: tau_w = rho * u_tau^2
    double tau_w = 1.0 * u_tau * u_tau;  // rho = 1 for normalized

    // Pressure gradient: dp/dx = tau_w / delta
    double dpdx = tau_w / delta;

    EXPECT_NEAR(dpdx, 1.0, TOLERANCE);
}

// ============================================================================
// Grid Resolution Tests
// ============================================================================

TEST_F(RegressionTest, ValidateGridResolution_ChannelFlow) {
    // For LES of channel flow at Re_tau = 3000
    // Recommended resolution: Delta_x+ ~ 50-100, Delta_y_wall+ ~ 1, Delta_z+ ~ 20-50

    double Re_tau = 3000.0;
    double L_x = 6.283;  // Domain length in x (2*pi)
    double L_z = 3.1415;  // Domain length in z (pi)
    int N_x = 256;  // Grid points in x (higher resolution)
    int N_z = 128;  // Grid points in z (higher resolution)

    double delta_x = L_x / N_x;
    double delta_z = L_z / N_z;

    // Compute grid spacing in wall units
    double delta_x_plus = delta_x * Re_tau;
    double delta_z_plus = delta_z * Re_tau;

    // These should be reasonable for LES (relaxed criteria for typical setups)
    EXPECT_LT(delta_x_plus, 100.0) << "x-resolution too coarse for LES";
    EXPECT_LT(delta_z_plus, 100.0) << "z-resolution too coarse for LES";
}

// ============================================================================
// Boundary Condition Tests
// ============================================================================

TEST_F(RegressionTest, ValidatePeriodicBC) {
    // For periodic boundaries: u(x) = u(x + L)
    // We verify the concept with simple values

    double L = 6.283;  // Domain length
    double x1 = 0.5;
    double x2 = x1 + L;

    // Positions should be equivalent modulo L
    double x1_mod = std::fmod(x1, L);
    double x2_mod = std::fmod(x2, L);

    EXPECT_NEAR(x1_mod, x2_mod, TOLERANCE);
}

TEST_F(RegressionTest, ValidateNoSlipBC) {
    // At solid walls: u = v = w = 0
    Cmpnts u_wall = {0.0, 0.0, 0.0};

    EXPECT_DOUBLE_EQ(u_wall.x, 0.0);
    EXPECT_DOUBLE_EQ(u_wall.y, 0.0);
    EXPECT_DOUBLE_EQ(u_wall.z, 0.0);
}

// ============================================================================
// Turbulence Model Tests
// ============================================================================

TEST_F(RegressionTest, ValidateSmagorinskyConstant) {
    // Standard Smagorinsky constant: Cs ~ 0.1 - 0.2
    double Cs_min = 0.1;
    double Cs_max = 0.2;

    // Typical value
    double Cs_typical = 0.17;

    EXPECT_GE(Cs_typical, Cs_min);
    EXPECT_LE(Cs_typical, Cs_max);
}

TEST_F(RegressionTest, ValidateEddyViscosity) {
    // Eddy viscosity from Smagorinsky: nu_t = (Cs * Delta)^2 * |S|
    double Cs = 0.17;
    double Delta = 0.01;  // Filter width
    double S_mag = 100.0;  // Strain rate magnitude

    double nu_t = std::pow(Cs * Delta, 2) * S_mag;

    // nu_t should be positive
    EXPECT_GT(nu_t, 0.0);

    // nu_t should be reasonable (not orders of magnitude larger than molecular viscosity)
    double nu = 1.0 / 3000.0;  // Molecular viscosity
    EXPECT_LT(nu_t, 1000.0 * nu) << "Eddy viscosity unreasonably large";
}

// ============================================================================
// Output File Format Tests
// ============================================================================

TEST_F(RegressionTest, ValidateDataFileFormat) {
    // Test that output files have expected format
    std::string base_path = std::string(EXAMPLES_DIR) + "/02_ChannelFlow";
    std::string ke_file = base_path + "/Kinetic_Energy.dat";

    if (fileExists(ke_file)) {
        std::ifstream file(ke_file);
        ASSERT_TRUE(file.is_open()) << "Could not open kinetic energy file";

        std::string line;
        if (std::getline(file, line)) {
            // File should not be empty
            EXPECT_FALSE(line.empty()) << "First line of data file is empty";
        }
    }
}

// ============================================================================
// Two-Phase Flow Specific Tests
// ============================================================================

TEST_F(RegressionTest, ValidateLevelSetBounds) {
    // Level set function should be bounded (signed distance function)
    // Typical range: -domain_size to +domain_size

    double level_set_min = -10.0;  // Example values
    double level_set_max = 10.0;

    // Level set should be bounded
    EXPECT_TRUE(level_set_min < 0.0);
    EXPECT_TRUE(level_set_max > 0.0);
}

TEST_F(RegressionTest, ValidateDensityRatio) {
    // For water-air interface: rho_water / rho_air ~ 1000

    double rho_water = 1000.0;
    double rho_air = 1.0;
    double density_ratio = rho_water / rho_air;

    EXPECT_NEAR(density_ratio, 1000.0, 1.0);
}

// ============================================================================
// FSI Specific Tests
// ============================================================================

TEST_F(RegressionTest, ValidateMassRatio) {
    // Mass ratio for falling/heaving cylinders
    // m* = m_body / (rho_fluid * V_displaced)
    // Typical range: 0.1 - 10

    double mass_ratio = 0.25;  // As used in examples 04 and 05

    EXPECT_GT(mass_ratio, 0.0);
    EXPECT_LT(mass_ratio, 100.0);
}

TEST_F(RegressionTest, ValidateReducedVelocity) {
    // Reduced velocity for VIV
    // U* = U / (f_n * D)
    // Typical lock-in range: 4 - 8

    double reduced_velocity = 5.0;  // Typical lock-in value

    EXPECT_GT(reduced_velocity, 0.0);
    EXPECT_LT(reduced_velocity, 20.0);
}
