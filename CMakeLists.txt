##################################################################
# Copyright (C) by Regents of the University of Minnesota.       #
#                                                                #
# This Software is released under GNU General Public License 2.0 #
# http://www.gnu.org/licenses/gpl-2.0.html                       #
#                                                                #
##################################################################

cmake_minimum_required(VERSION 3.16)
project(VFS-Wind LANGUAGES C CXX)

# C++ standard (Kokkos requires C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Options
option(ENABLE_TECPLOT "Enable Tecplot output" OFF)
option(ENABLE_XML_INPUT "Enable XML input file parsing (bundled TinyXML-2)" ON)
option(ENABLE_VTK_OUTPUT "Enable VTK XML output for ParaView" ON)
option(ENABLE_TESTING "Enable unit, integration, and regression tests" ON)

# GPU Options
option(ENABLE_GPU "Enable GPU acceleration via Kokkos" OFF)
set(KOKKOS_BACKEND "CUDA" CACHE STRING "Kokkos backend: CUDA, HIP, SYCL, OPENMP, SERIAL")
set_property(CACHE KOKKOS_BACKEND PROPERTY STRINGS CUDA HIP SYCL OPENMP SERIAL)

# Find MPI
find_package(MPI REQUIRED)

# Find PkgConfig for PETSc and HYPRE detection
find_package(PkgConfig REQUIRED)

# Find PETSc
pkg_check_modules(PETSC REQUIRED PETSc)
message(STATUS "Found PETSc version: ${PETSC_VERSION}")
message(STATUS "PETSc include dirs: ${PETSC_INCLUDE_DIRS}")
message(STATUS "PETSc libraries: ${PETSC_LIBRARIES}")
message(STATUS "PETSc library dirs: ${PETSC_LIBRARY_DIRS}")

# Find HYPRE
# Try pkg-config first, then fall back to manual detection
pkg_check_modules(HYPRE QUIET HYPRE)
if(NOT HYPRE_FOUND)
    # Manual HYPRE detection
    find_path(HYPRE_INCLUDE_DIRS HYPRE.h
        HINTS
            ENV HYPRE_DIR
            ${PETSC_INCLUDE_DIRS}
        PATH_SUFFIXES include
    )
    find_library(HYPRE_LIBRARIES NAMES HYPRE
        HINTS
            ENV HYPRE_DIR
            ${PETSC_LIBRARY_DIRS}
        PATH_SUFFIXES lib lib64
    )
    if(HYPRE_INCLUDE_DIRS AND HYPRE_LIBRARIES)
        set(HYPRE_FOUND TRUE)
        message(STATUS "Found HYPRE (manual): ${HYPRE_LIBRARIES}")
    else()
        message(FATAL_ERROR "HYPRE not found. Set HYPRE_DIR environment variable.")
    endif()
else()
    message(STATUS "Found HYPRE version: ${HYPRE_VERSION}")
endif()

# Find BLAS and LAPACK
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")

# macOS specific: prefer Accelerate framework
if(APPLE)
    # Homebrew paths
    list(APPEND CMAKE_PREFIX_PATH
        "/opt/homebrew"
        "/usr/local"
    )
endif()

##################################################################
# GPU/Kokkos Configuration
##################################################################
if(ENABLE_GPU)
    message(STATUS "GPU acceleration enabled with backend: ${KOKKOS_BACKEND}")

    # macOS OpenMP setup (Apple clang doesn't include OpenMP)
    if(APPLE AND (KOKKOS_BACKEND STREQUAL "OPENMP"))
        message(STATUS "macOS detected with OpenMP backend - configuring Homebrew libomp")

        # Find Homebrew libomp (works for both Intel and Apple Silicon Macs)
        if(EXISTS "/opt/homebrew/opt/libomp")
            # Apple Silicon (M1/M2/M3)
            set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
        elseif(EXISTS "/usr/local/opt/libomp")
            # Intel Mac
            set(OpenMP_ROOT "/usr/local/opt/libomp")
        endif()

        if(DEFINED OpenMP_ROOT)
            message(STATUS "Found Homebrew libomp at: ${OpenMP_ROOT}")

            # Set OpenMP flags for C and C++
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include" CACHE STRING "" FORCE)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include" CACHE STRING "" FORCE)
            set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/libomp.dylib" CACHE STRING "" FORCE)

            # Help CMake find it
            list(APPEND CMAKE_PREFIX_PATH "${OpenMP_ROOT}")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I${OpenMP_ROOT}/include")
            link_directories("${OpenMP_ROOT}/lib")
        else()
            message(FATAL_ERROR
                "OpenMP backend requested on macOS but libomp not found.\n"
                "Please install it with: brew install libomp\n"
                "Or use SERIAL backend instead: -DKOKKOS_BACKEND=SERIAL")
        endif()
    endif()

    # Try to find installed Kokkos first
    find_package(Kokkos QUIET)

    if(Kokkos_FOUND)
        message(STATUS "Found installed Kokkos: ${Kokkos_DIR}")
    else()
        # Fetch Kokkos if not found
        message(STATUS "Kokkos not found, fetching from GitHub...")
        include(FetchContent)

        # Configure Kokkos options based on selected backend
        if(KOKKOS_BACKEND STREQUAL "CUDA")
            set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
            set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "")
            # Auto-detect CUDA architecture if not specified
            if(NOT DEFINED Kokkos_ARCH_AMPERE80 AND NOT DEFINED Kokkos_ARCH_VOLTA70)
                set(Kokkos_ARCH_AMPERE80 ON CACHE BOOL "Target A100 GPU")
            endif()
        elseif(KOKKOS_BACKEND STREQUAL "HIP")
            set(Kokkos_ENABLE_HIP ON CACHE BOOL "")
        elseif(KOKKOS_BACKEND STREQUAL "SYCL")
            set(Kokkos_ENABLE_SYCL ON CACHE BOOL "")
        elseif(KOKKOS_BACKEND STREQUAL "OPENMP")
            set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "")
        else()
            set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "")
        endif()

        FetchContent_Declare(
            kokkos
            GIT_REPOSITORY https://github.com/kokkos/kokkos.git
            GIT_TAG        4.3.00  # Latest stable release
        )
        FetchContent_MakeAvailable(kokkos)
    endif()

    # Define GPU compile definition
    add_compile_definitions(ENABLE_GPU)
    add_compile_definitions(KOKKOS_BACKEND_${KOKKOS_BACKEND})

    message(STATUS "Kokkos configuration complete")
else()
    message(STATUS "GPU acceleration disabled (use -DENABLE_GPU=ON to enable)")
endif()

# Add source subdirectory
add_subdirectory(Source)

# Add tests subdirectory (optional)
if(ENABLE_TESTING)
    message(STATUS "Testing enabled - building test suite")
    add_subdirectory(tests)
endif()
