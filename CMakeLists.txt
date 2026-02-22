##################################################################
# Copyright (C) by Regents of the University of Minnesota.       #
#                                                                #
# This Software is released under GNU General Public License 2.0 #
# http://www.gnu.org/licenses/gpl-2.0.html                       #
#                                                                #
##################################################################

cmake_minimum_required(VERSION 3.16)
project(VFS-Wind LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# Options
option(ENABLE_TECPLOT "Enable Tecplot output" OFF)
option(ENABLE_XML_INPUT "Enable XML input file parsing (bundled TinyXML-2)" ON)
option(ENABLE_VTK_OUTPUT "Enable VTK XML output for ParaView" ON)
option(ENABLE_TESTING "Enable unit, integration, and regression tests" ON)

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

# Add source subdirectory
add_subdirectory(Source)

# Add tests subdirectory (optional)
if(ENABLE_TESTING)
    message(STATUS "Testing enabled - building test suite")
    add_subdirectory(tests)
endif()
