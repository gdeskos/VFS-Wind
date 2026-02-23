##################################################################
# Copyright (C) by Regents of the University of Minnesota.       #
#                                                                #
# This Software is released under GNU General Public License 2.0 #
# http://www.gnu.org/licenses/gpl-2.0.html                       #
#                                                                #
# VFS-Wind Test Suite                                            #
##################################################################

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Include GoogleTest module for gtest_discover_tests
include(GoogleTest)

# Common include directories for all tests
set(TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/Source
    ${MPI_CXX_INCLUDE_DIRS}
    ${PETSC_INCLUDE_DIRS}
    ${HYPRE_INCLUDE_DIRS}
)

# Common link directories
set(TEST_LINK_DIRS
    ${PETSC_LIBRARY_DIRS}
)

# Common link libraries
set(TEST_LINK_LIBS
    ${PETSC_LIBRARIES}
    ${HYPRE_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    MPI::MPI_CXX
    MPI::MPI_C
    GTest::gtest_main
    GTest::gmock
)

# ============================================================================
# UNIT TESTS
# ============================================================================

# Test for computational geometry functions
add_executable(test_compgeom
    unit/test_compgeom.cpp
    ${CMAKE_SOURCE_DIR}/Source/compgeom.c
)
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/Source/compgeom.c
    PROPERTIES LANGUAGE CXX
)
target_include_directories(test_compgeom PRIVATE ${TEST_INCLUDE_DIRS})
target_link_directories(test_compgeom PRIVATE ${TEST_LINK_DIRS})
target_link_libraries(test_compgeom PRIVATE ${TEST_LINK_LIBS})
target_compile_definitions(test_compgeom PRIVATE ENABLE_XML_INPUT ENABLE_VTK_OUTPUT)

# Test for XML input parsing
if(ENABLE_XML_INPUT)
    add_executable(test_xml_input
        unit/test_xml_input.cpp
        ${CMAKE_SOURCE_DIR}/Source/xml_input.cpp
        ${CMAKE_SOURCE_DIR}/Source/tinyxml2/tinyxml2.cpp
    )
    target_include_directories(test_xml_input PRIVATE ${TEST_INCLUDE_DIRS})
    target_link_directories(test_xml_input PRIVATE ${TEST_LINK_DIRS})
    target_link_libraries(test_xml_input PRIVATE ${TEST_LINK_LIBS})
    target_compile_definitions(test_xml_input PRIVATE ENABLE_XML_INPUT)
    # Copy test data files to build directory
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/data/test_control.xml
        ${CMAKE_CURRENT_BINARY_DIR}/test_control.xml
        COPYONLY
    )
endif()

# Test for mathematical utilities
add_executable(test_math_utils
    unit/test_math_utils.cpp
)
target_include_directories(test_math_utils PRIVATE ${TEST_INCLUDE_DIRS})
target_link_directories(test_math_utils PRIVATE ${TEST_LINK_DIRS})
target_link_libraries(test_math_utils PRIVATE ${TEST_LINK_LIBS})
target_compile_definitions(test_math_utils PRIVATE ENABLE_XML_INPUT ENABLE_VTK_OUTPUT)

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

# Test for metrics computation - standalone mathematical tests
# (does not require linking with full VFS-Wind source)
add_executable(test_metrics_integration
    integration/test_metrics.cpp
)
target_include_directories(test_metrics_integration PRIVATE ${TEST_INCLUDE_DIRS})
target_link_directories(test_metrics_integration PRIVATE ${TEST_LINK_DIRS})
target_link_libraries(test_metrics_integration PRIVATE ${TEST_LINK_LIBS})
target_compile_definitions(test_metrics_integration PRIVATE ENABLE_XML_INPUT ENABLE_VTK_OUTPUT)

# ============================================================================
# REGRESSION TESTS
# ============================================================================

# Regression test runner script
add_executable(test_regression
    regression/test_regression.cpp
)
target_include_directories(test_regression PRIVATE ${TEST_INCLUDE_DIRS})
target_link_directories(test_regression PRIVATE ${TEST_LINK_DIRS})
target_link_libraries(test_regression PRIVATE ${TEST_LINK_LIBS})
target_compile_definitions(test_regression PRIVATE
    ENABLE_XML_INPUT
    ENABLE_VTK_OUTPUT
    EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/examples"
)

# ============================================================================
# GPU TESTS (optional, only when ENABLE_GPU is ON)
# ============================================================================

if(ENABLE_GPU)
    # GPU Smoke Test
    add_executable(test_gpu_smoke
        gpu/test_gpu_smoke.cpp
        ${CMAKE_SOURCE_DIR}/Source/gpu/kokkos_init.cpp
    )
    target_include_directories(test_gpu_smoke PRIVATE
        ${TEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/Source/gpu
    )
    target_link_directories(test_gpu_smoke PRIVATE ${TEST_LINK_DIRS})
    target_link_libraries(test_gpu_smoke PRIVATE
        ${TEST_LINK_LIBS}
        Kokkos::kokkos
    )
    target_compile_definitions(test_gpu_smoke PRIVATE ENABLE_GPU)

    gtest_discover_tests(test_gpu_smoke
        TEST_PREFIX "gpu/"
        PROPERTIES LABELS "gpu"
    )

    # GPU Kernel Tests (Convection, Viscous, Pressure Gradient)
    add_executable(test_gpu_kernels
        gpu/test_kernels.cpp
    )
    target_include_directories(test_gpu_kernels PRIVATE
        ${TEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/Source/gpu
        ${CMAKE_SOURCE_DIR}/Source/gpu/kernels
    )
    target_link_directories(test_gpu_kernels PRIVATE ${TEST_LINK_DIRS})
    target_link_libraries(test_gpu_kernels PRIVATE
        ${TEST_LINK_LIBS}
        Kokkos::kokkos
    )
    target_compile_definitions(test_gpu_kernels PRIVATE ENABLE_GPU)

    gtest_discover_tests(test_gpu_kernels
        TEST_PREFIX "gpu/kernels/"
        PROPERTIES LABELS "gpu;kernels"
    )

    # GPU Solver Tests (Phase 2: Linear Solver GPU Acceleration)
    add_executable(test_gpu_solver
        gpu/test_gpu_solver.cpp
        ${CMAKE_SOURCE_DIR}/Source/gpu/kokkos_init.cpp
        ${CMAKE_SOURCE_DIR}/Source/gpu/gpu_solver.cpp
    )
    target_include_directories(test_gpu_solver PRIVATE
        ${TEST_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/Source/gpu
        ${CMAKE_SOURCE_DIR}/Source/gpu/kernels
    )
    target_link_directories(test_gpu_solver PRIVATE ${TEST_LINK_DIRS})
    target_link_libraries(test_gpu_solver PRIVATE
        ${TEST_LINK_LIBS}
        Kokkos::kokkos
    )
    target_compile_definitions(test_gpu_solver PRIVATE ENABLE_GPU)

    gtest_discover_tests(test_gpu_solver
        TEST_PREFIX "gpu/solver/"
        PROPERTIES LABELS "gpu;solver"
    )

    # Custom target to run GPU tests
    add_custom_target(run_gpu_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -L gpu --output-on-failure
        DEPENDS test_gpu_smoke test_gpu_kernels test_gpu_solver
        COMMENT "Running GPU tests..."
    )

    message(STATUS "GPU tests: ENABLED (smoke + kernels + solver)")
else()
    message(STATUS "GPU tests: DISABLED (use -DENABLE_GPU=ON to enable)")
endif()

# ============================================================================
# Register tests with CTest
# ============================================================================

# Unit tests
gtest_discover_tests(test_compgeom
    TEST_PREFIX "unit/"
    PROPERTIES LABELS "unit"
)

if(ENABLE_XML_INPUT)
    gtest_discover_tests(test_xml_input
        TEST_PREFIX "unit/"
        PROPERTIES LABELS "unit"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

gtest_discover_tests(test_math_utils
    TEST_PREFIX "unit/"
    PROPERTIES LABELS "unit"
)

# Integration tests (run with mpirun for 1 process)
gtest_discover_tests(test_metrics_integration
    TEST_PREFIX "integration/"
    PROPERTIES LABELS "integration"
)

# Regression tests
gtest_discover_tests(test_regression
    TEST_PREFIX "regression/"
    PROPERTIES LABELS "regression"
)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_compgeom test_math_utils test_metrics_integration test_regression
    COMMENT "Running all tests..."
)

# Custom target to run only unit tests
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit --output-on-failure
    DEPENDS test_compgeom test_math_utils
    COMMENT "Running unit tests..."
)

# Custom target to run only integration tests
add_custom_target(run_integration_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration --output-on-failure
    DEPENDS test_metrics_integration
    COMMENT "Running integration tests..."
)

# Custom target to run only regression tests
add_custom_target(run_regression_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L regression --output-on-failure
    DEPENDS test_regression
    COMMENT "Running regression tests..."
)
