##################################################################
# Copyright (C) by Regents of the University of Minnesota.       #
#                                                                #
# This Software is released under GNU General Public License 2.0 #
# http://www.gnu.org/licenses/gpl-2.0.html                       #
#                                                                #
##################################################################

# Source files for vwis executable
# Note: These .c files contain C++ code (std::vector, etc.) so we compile them as C++
set(VWIS_SOURCES
    bcs.c
    bmv.c
    compgeom.c
    distance.c
    fsi.c
    fsi_move.c
    ibm.c
    ibm_io.c
    implicitsolver.c
    init.c
    k-omega.c
    les.c
    level.c
    main.c
    metrics.c
    momentum.c
    poisson.c
    poisson_hypre.c
    rhs.c
    rhs2.c
    rotor_model.c
    solvers.c
    timeadvancing.c
    timeadvancing1.c
    variables.c
    wallfunction.c
    wave.c
)

# XML Input support (optional)
if(ENABLE_XML_INPUT)
    list(APPEND VWIS_SOURCES
        tinyxml2/tinyxml2.cpp
        xml_input.cpp
    )
endif()

# VTK Output support (optional)
if(ENABLE_VTK_OUTPUT)
    list(APPEND VWIS_SOURCES
        vtk_output.cpp
    )
endif()

# GPU/Kokkos support (optional)
if(ENABLE_GPU)
    list(APPEND VWIS_SOURCES
        gpu/kokkos_init.cpp
        gpu/gpu_solver.cpp
    )
    message(STATUS "GPU support: ENABLED (backend: ${KOKKOS_BACKEND})")
endif()

# Create vwis executable
add_executable(vwis ${VWIS_SOURCES})

# Force C++ compilation for .c files (they contain C++ code like std::vector)
set_source_files_properties(${VWIS_SOURCES} PROPERTIES LANGUAGE CXX)

# Include directories
target_include_directories(vwis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    ${MPI_CXX_INCLUDE_DIRS}
    ${PETSC_INCLUDE_DIRS}
    ${HYPRE_INCLUDE_DIRS}
)

# Link directories
target_link_directories(vwis PRIVATE
    ${PETSC_LIBRARY_DIRS}
)

# Link libraries
target_link_libraries(vwis PRIVATE
    ${PETSC_LIBRARIES}
    ${HYPRE_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    MPI::MPI_CXX
    MPI::MPI_C
)

# Link Kokkos if GPU is enabled
if(ENABLE_GPU)
    target_link_libraries(vwis PRIVATE Kokkos::kokkos)
    # On macOS with OpenMP backend, explicitly link libomp
    if(APPLE AND KOKKOS_BACKEND STREQUAL "OPENMP")
        if(EXISTS "/opt/homebrew/opt/libomp/lib/libomp.dylib")
            target_link_libraries(vwis PRIVATE "/opt/homebrew/opt/libomp/lib/libomp.dylib")
        elseif(EXISTS "/usr/local/opt/libomp/lib/libomp.dylib")
            target_link_libraries(vwis PRIVATE "/usr/local/opt/libomp/lib/libomp.dylib")
        endif()
    endif()
endif()

# Compile definitions
target_compile_definitions(vwis PRIVATE NDEBUG)

# XML Input compile definition
if(ENABLE_XML_INPUT)
    target_compile_definitions(vwis PRIVATE ENABLE_XML_INPUT)
    message(STATUS "XML input support: ENABLED")
endif()

# VTK Output compile definition
if(ENABLE_VTK_OUTPUT)
    target_compile_definitions(vwis PRIVATE ENABLE_VTK_OUTPUT)
    message(STATUS "VTK output support: ENABLED")
endif()

# Compiler flags
target_compile_options(vwis PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Tecplot support (optional)
if(ENABLE_TECPLOT)
    target_compile_definitions(vwis PRIVATE TECIO)

    # Find Tecplot
    find_path(TECPLOT_INCLUDE_DIR TECIO.h
        HINTS ENV TEC360HOME
        PATH_SUFFIXES include
    )
    find_library(TECPLOT_LIBRARY NAMES tecio64 tecio
        HINTS ENV TEC360HOME
        PATH_SUFFIXES lib lib64
    )

    if(TECPLOT_INCLUDE_DIR AND TECPLOT_LIBRARY)
        target_include_directories(vwis PRIVATE ${TECPLOT_INCLUDE_DIR})
        target_link_libraries(vwis PRIVATE ${TECPLOT_LIBRARY})
        message(STATUS "Found Tecplot: ${TECPLOT_LIBRARY}")
    else()
        message(FATAL_ERROR "Tecplot enabled but not found. Set TEC360HOME environment variable.")
    endif()
endif()

# Post-processing executable: data
# VTK output works without Tecplot; Tecplot output requires ENABLE_TECPLOT
set(DATA_SOURCES data.c)

# Add XML input support to data executable
if(ENABLE_XML_INPUT)
    list(APPEND DATA_SOURCES
        tinyxml2/tinyxml2.cpp
        xml_input.cpp
    )
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data.c)
    add_executable(data ${DATA_SOURCES})
    # Force C++ compilation (contains C++ code)
    set_source_files_properties(${DATA_SOURCES} PROPERTIES LANGUAGE CXX)
    target_include_directories(data PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
        ${PETSC_INCLUDE_DIRS}
        ${HYPRE_INCLUDE_DIRS}
    )
    target_link_directories(data PRIVATE ${PETSC_LIBRARY_DIRS})
    target_link_libraries(data PRIVATE
        ${PETSC_LIBRARIES}
        ${HYPRE_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        MPI::MPI_CXX
        MPI::MPI_C
    )
    target_compile_definitions(data PRIVATE NDEBUG)
    target_compile_options(data PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    # Add XML input support if enabled
    if(ENABLE_XML_INPUT)
        target_compile_definitions(data PRIVATE ENABLE_XML_INPUT)
    endif()
    # Add Tecplot support if enabled
    if(ENABLE_TECPLOT AND TECPLOT_LIBRARY)
        target_compile_definitions(data PRIVATE TECIO)
        target_include_directories(data PRIVATE ${TECPLOT_INCLUDE_DIR})
        target_link_libraries(data PRIVATE ${TECPLOT_LIBRARY})
    endif()
    message(STATUS "Post-processing executable 'data' will be built (VTK output available)")
endif()

# Post-processing executable: data05 (legacy version - requires Tecplot)
set(DATA05_SOURCES data05.c)
if(ENABLE_TECPLOT AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data05.c)
    add_executable(data05 ${DATA05_SOURCES})
    # Force C++ compilation (contains C++ code)
    set_source_files_properties(${DATA05_SOURCES} PROPERTIES LANGUAGE CXX)
    target_include_directories(data05 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
        ${PETSC_INCLUDE_DIRS}
        ${HYPRE_INCLUDE_DIRS}
    )
    target_link_directories(data05 PRIVATE ${PETSC_LIBRARY_DIRS})
    target_link_libraries(data05 PRIVATE
        ${PETSC_LIBRARIES}
        ${HYPRE_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        MPI::MPI_CXX
        MPI::MPI_C
    )
    target_compile_definitions(data05 PRIVATE NDEBUG)
    target_compile_options(data05 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
    # Add Tecplot support if enabled
    if(ENABLE_TECPLOT AND TECPLOT_LIBRARY)
        target_compile_definitions(data05 PRIVATE TECIO)
        target_include_directories(data05 PRIVATE ${TECPLOT_INCLUDE_DIR})
        target_link_libraries(data05 PRIVATE ${TECPLOT_LIBRARY})
    endif()
endif()

# Install targets
install(TARGETS vwis DESTINATION bin)
if(TARGET data)
    install(TARGETS data DESTINATION bin)
endif()
if(TARGET data05)
    install(TARGETS data05 DESTINATION bin)
endif()
