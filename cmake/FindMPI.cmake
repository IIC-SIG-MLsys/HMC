# 先清空变量，避免缓存影响
unset(MPI_INCLUDE_DIR CACHE)
unset(MPI_LIBRARY CACHE)
unset(MPI_CXX_LIBRARY CACHE)

# 检测 CPU 架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Detected x86_64 architecture, setting MPI paths for Ubuntu")
    find_path(MPI_INCLUDE_DIR
        NAMES mpi.h
        PATHS /usr/lib/x86_64-linux-gnu/openmpi/include
        PATH_SUFFIXES openmpi
    )
    find_library(MPI_LIBRARY
        NAMES mpi
        PATHS /usr/lib/x86_64-linux-gnu/openmpi/lib
    )
    find_library(MPI_CXX_LIBRARY
	NAMES mpi_cxx
	PATHS /usr/lib/x86_64-linux-gnu/openmpi/lib
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Detected aarch64 architecture, setting MPI paths for OpenEuler")
    find_path(MPI_INCLUDE_DIR
        NAMES mpi.h
        PATHS /usr/include/openmpi-aarch64
    )
    find_library(MPI_LIBRARY
        NAMES mpi
        PATHS /usr/lib64/openmpi/lib
    )
    find_library(MPI_CXX_LIBRARY
	NAMES mpi_cxx
	PATHS /usr/lib64/openmpi/lib
    )
else()
    message(FATAL_ERROR "Unsupported architecture for MPI detection: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# 检查是否找到 MPI
if(MPI_INCLUDE_DIR AND MPI_LIBRARY AND MPI_CXX_LIBRARY)
    message(STATUS "Found MPI: includes in ${MPI_INCLUDE_DIR}, libraries in ${MPI_LIBRARY}, CXX library in ${MPI_CXX_LIBRARY}")
    mark_as_advanced(MPI_INCLUDE_DIR MPI_LIBRARY MPI_CXX_LIBRARY)
else()
    message(FATAL_ERROR "Could not find MPI or MPI C++ library. Please make sure MPI is installed and properly configured.")
endif()
