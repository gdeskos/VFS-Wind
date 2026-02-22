# VFS-Wind Test Suite

This directory contains unit, integration, and regression tests for the VFS-Wind CFD solver.

## Test Structure

```
tests/
├── CMakeLists.txt          # Test build configuration
├── README.md               # This file
├── unit/                   # Unit tests
│   ├── test_compgeom.cpp   # Computational geometry tests
│   ├── test_xml_input.cpp  # XML input parsing tests
│   └── test_math_utils.cpp # Mathematical utility tests
├── integration/            # Integration tests
│   └── test_metrics.cpp    # Grid metrics computation tests
├── regression/             # Regression tests
│   └── test_regression.cpp # Example case validation tests
└── data/                   # Test data files
    └── test_control.xml    # Sample XML configuration
```

## Building Tests

Tests are enabled by default. To build with tests:

```bash
mkdir build && cd build
cmake .. -DENABLE_TESTING=ON
make
```

To disable tests:

```bash
cmake .. -DENABLE_TESTING=OFF
```

## Running Tests

### Run All Tests

```bash
cd build
ctest --output-on-failure
```

Or use the custom target:

```bash
make run_tests
```

### Run Specific Test Categories

**Unit tests only:**
```bash
ctest -L unit --output-on-failure
# or
make run_unit_tests
```

**Integration tests only:**
```bash
ctest -L integration --output-on-failure
# or
make run_integration_tests
```

**Regression tests only:**
```bash
ctest -L regression --output-on-failure
# or
make run_regression_tests
```

### Run Individual Test Executables

```bash
./test_compgeom      # Computational geometry tests
./test_xml_input     # XML input tests
./test_math_utils    # Math utility tests
./test_metrics_integration  # Metrics integration tests
./test_regression    # Regression tests
```

### Run Specific Test Cases

```bash
./test_compgeom --gtest_filter="*TriArea*"
./test_xml_input --gtest_filter="*ParseXML*"
```

## Test Categories

### Unit Tests

Unit tests verify individual functions in isolation:

- **test_compgeom.cpp**: Tests computational geometry functions
  - Triangle area calculation (`tri_area`)
  - Point-to-line distance (`Dis_P_Line`)
  - Ray-triangle intersection (`intsect_triangle`)
  - Point-in-triangle tests (`ISInsideTriangle2D`, `ISPointInTriangle`)
  - Triangle interpolation (`triangle_intp3D`)

- **test_xml_input.cpp**: Tests XML configuration parsing
  - File existence checking
  - Parsing of all configuration sections (simulation, physics, turbulence, etc.)
  - Error handling for invalid XML

- **test_math_utils.cpp**: Tests mathematical utilities
  - Vector operations (dot product, cross product)
  - Heaviside function
  - Sign function
  - Upwind scheme
  - CFL condition calculations
  - IBMInfo interpolation

### Integration Tests

Integration tests verify module interactions:

- **test_metrics.cpp**: Tests grid metrics computation
  - Jacobian properties
  - Coordinate transformations
  - Grid quality metrics
  - Face area and volume computation
  - Gradient computation
  - Metric identities

### Regression Tests

Regression tests validate against example cases:

- **test_regression.cpp**: Tests example case validation
  - File existence for test cases
  - Kinetic energy evolution
  - Convergence behavior
  - Physics validation (log-law, TKE profiles)
  - Conservation properties (mass, momentum)
  - Grid resolution adequacy

## Adding New Tests

### Adding a Unit Test

1. Create a new test file in `tests/unit/`:

```cpp
#include <gtest/gtest.h>
#include "petsc.h"
#include "variables.h"

class MyTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        int argc = 0;
        char **argv = nullptr;
        PetscInitialize(&argc, &argv, nullptr, nullptr);
    }
    static void TearDownTestSuite() {
        PetscFinalize();
    }
};

TEST_F(MyTest, TestName) {
    // Your test code here
    EXPECT_EQ(expected, actual);
}
```

2. Add the test to `tests/CMakeLists.txt`:

```cmake
add_executable(test_my_module
    unit/test_my_module.cpp
    ${CMAKE_SOURCE_DIR}/Source/my_module.c
)
target_include_directories(test_my_module PRIVATE ${TEST_INCLUDE_DIRS})
target_link_directories(test_my_module PRIVATE ${TEST_LINK_DIRS})
target_link_libraries(test_my_module PRIVATE ${TEST_LINK_LIBS})

gtest_discover_tests(test_my_module
    TEST_PREFIX "unit/"
    PROPERTIES LABELS "unit"
)
```

### Adding a Regression Test

Add test cases to `tests/regression/test_regression.cpp`:

```cpp
TEST_F(RegressionTest, MyNewTestCase) {
    std::string base_path = std::string(EXAMPLES_DIR) + "/My_Test_Case";

    // Check files exist
    EXPECT_TRUE(fileExists(base_path + "/control.xml"));

    // Check physics
    EXPECT_TRUE(checkKineticEnergy(base_path + "/Kinetic_Energy.dat"));
}
```

## Test Dependencies

- **Google Test (GTest)**: Automatically fetched via CMake FetchContent
- **PETSc**: Required for all tests (provides data types and MPI support)
- **HYPRE**: Required for solver-related tests

## Continuous Integration

For CI/CD pipelines, use:

```bash
cmake -B build -DENABLE_TESTING=ON
cmake --build build
cd build && ctest --output-on-failure -j$(nproc)
```

## Troubleshooting

### Tests fail to build

Ensure PETSc and HYPRE are properly installed and detected:
```bash
cmake .. -DPETSC_DIR=/path/to/petsc -DHYPRE_DIR=/path/to/hypre
```

### MPI-related test failures

Some tests require MPI. Ensure MPI is available:
```bash
mpirun -np 1 ./test_metrics_integration
```

### Missing example data

Regression tests may skip if example output files don't exist. Run the examples first:
```bash
cd examples/Test_10_ChannelFlow_Retau3000
mpirun -np 4 ../../build/vwis
```
