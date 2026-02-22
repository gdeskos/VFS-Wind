/*****************************************************************
 * Copyright (C) by Regents of the University of Minnesota.       *
 *                                                                *
 * This Software is released under GNU General Public License 2.0 *
 * http://www.gnu.org/licenses/gpl-2.0.html                       *
 *                                                                *
 * Unit tests for XML input parsing (xml_input.cpp)               *
 ******************************************************************/

#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>
#include <string>

#include "petsc.h"
#include "variables.h"

// Functions from xml_input.cpp - declared in variables.h with extern "C" already

class XMLInputTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int argc = 0;
        char **argv = nullptr;
        PetscInitialize(&argc, &argv, nullptr, nullptr);
    }

    static void TearDownTestSuite() {
        PetscFinalize();
    }

    void SetUp() override {
        // Create test XML files for each test
    }

    void TearDown() override {
        // Clean up test files
    }

    // Helper to create a temporary XML file
    std::string createTempXMLFile(const std::string& content) {
        std::string filename = "temp_test_" + std::to_string(testFileCounter++) + ".xml";
        std::ofstream file(filename);
        file << content;
        file.close();
        tempFiles.push_back(filename);
        return filename;
    }

    // Clean up temp files
    ~XMLInputTest() {
        for (const auto& f : tempFiles) {
            std::remove(f.c_str());
        }
    }

private:
    static int testFileCounter;
    std::vector<std::string> tempFiles;
};

int XMLInputTest::testFileCounter = 0;

// ============================================================================
// File Existence Tests
// ============================================================================

TEST_F(XMLInputTest, FileExists_ExistingFile) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = xml_file_exists(filename.c_str());
    EXPECT_EQ(result, 1);
}

TEST_F(XMLInputTest, FileExists_NonExistingFile) {
    int result = xml_file_exists("nonexistent_file_12345.xml");
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Basic Parsing Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_MinimalValidFile) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);  // Success
}

TEST_F(XMLInputTest, ParseXML_InvalidFile) {
    int result = ParseXMLControlFile("nonexistent_file.xml");
    EXPECT_NE(result, 0);  // Should fail
}

TEST_F(XMLInputTest, ParseXML_MalformedXML) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <simulation>
    <timestep dt="0.001"
  </simulation>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_NE(result, 0);  // Should fail
}

TEST_F(XMLInputTest, ParseXML_MissingRootElement) {
    std::string content = R"(<?xml version="1.0"?>
<notVfswind>
</notVfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_NE(result, 0);  // Should fail - no <vfswind> root element
}

// ============================================================================
// Simulation Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_SimulationTimestep) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <simulation>
    <timestep dt="0.001" totalsteps="1000" output_interval="10"/>
  </simulation>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_SimulationRestart) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <simulation>
    <timestep dt="0.001" totalsteps="1000"/>
    <restart enabled="1" timestep="500"/>
  </simulation>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_SimulationOptions) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <simulation>
    <timestep dt="0.001" totalsteps="1000"/>
    <options delete_previous="0" second_order="1" laplacian="0" averaging="1"/>
  </simulation>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Physics Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_PhysicsReynolds) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <physics>
    <reynolds_number>10000</reynolds_number>
  </physics>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_PhysicsGravity) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <physics>
    <gravity x="0.0" y="-9.81" z="0.0"/>
  </physics>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Turbulence Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_TurbulenceLES) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <turbulence>
    <les enabled="1" model="smagorinsky" max_cs="0.2"/>
  </turbulence>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_TurbulenceDynamicLES) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <turbulence>
    <les enabled="1" model="dynamic"/>
  </turbulence>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_TurbulenceRANS) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <turbulence>
    <rans enabled="1"/>
  </turbulence>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_TurbulenceWallFunction) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <turbulence>
    <wall_function enabled="1" channel_height="2.0"/>
  </turbulence>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Levelset Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_LevelsetBasic) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <levelset enabled="1" sloshing="1">
    <fluid0 density="1000.0" viscosity="0.001"/>
    <fluid1 density="1.2" viscosity="1.8e-5"/>
    <iterations>10</iterations>
  </levelset>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_LevelsetAdvanced) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <levelset enabled="1" surface_tension="1" dthick="0.01" tau="0.5">
    <fluid0 density="1000.0" viscosity="0.001"/>
    <fluid1 density="1.2" viscosity="1.8e-5"/>
    <iterations>10</iterations>
  </levelset>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Solvers Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_SolversPoisson) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <solvers>
    <poisson type="1" iterations="15" tolerance="5e-9"/>
  </solvers>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

TEST_F(XMLInputTest, ParseXML_SolversMomentum) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <solvers>
    <momentum implicit="1" max_iterations="50" tolerance="1e-5"/>
  </solvers>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Parallel Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_ParallelPeriodic) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <parallel>
    <periodic i="1" j="0" k="1"/>
  </parallel>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Output Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_OutputVTK) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <output>
    <vtk enabled="1"/>
    <binary enabled="1"/>
    <path>./output</path>
  </output>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Grid and Files Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_GridFile) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <grid file="grid.dat" format="xyz"/>
  <boundaries file="bcs.dat"/>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// IBM Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_ImmersedBoundary) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <immersed_boundary enabled="1" bodies="2"/>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Channel Flow Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_ChannelFlow) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <channel_flow enabled="1" flux="1.0" inlet="1" perturb="1"/>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Rotor Model Section Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_RotorModel) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <rotor_model enabled="1" turbines="1" blades="3" tip_speed_ratio="7.5"/>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}

// ============================================================================
// Complete Configuration Tests
// ============================================================================

TEST_F(XMLInputTest, ParseXML_CompleteConfiguration) {
    std::string content = R"(<?xml version="1.0"?>
<vfswind version="1.0">
  <simulation>
    <timestep dt="0.001" totalsteps="10000" output_interval="100"/>
    <restart enabled="0"/>
    <options delete_previous="0" second_order="1"/>
  </simulation>
  <physics>
    <reynolds_number>3000</reynolds_number>
    <gravity x="0.0" y="0.0" z="0.0"/>
  </physics>
  <turbulence>
    <les enabled="1" model="dynamic"/>
    <wall_function enabled="1" channel_height="2.0"/>
  </turbulence>
  <solvers>
    <poisson type="1" iterations="15" tolerance="5e-9"/>
    <momentum implicit="0"/>
  </solvers>
  <parallel>
    <periodic i="1" j="0" k="1"/>
  </parallel>
  <output>
    <vtk enabled="1"/>
    <binary enabled="1"/>
  </output>
  <grid file="grid.dat" format="xyz"/>
  <boundaries file="bcs.dat"/>
</vfswind>
)";
    std::string filename = createTempXMLFile(content);

    int result = ParseXMLControlFile(filename.c_str());
    EXPECT_EQ(result, 0);
}
