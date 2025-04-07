# cmake/FindRocm.cmake

# default ROCm path
set(ROCM_PATH "" CACHE PATH "Path to ROCm installation (optional)")
if(ROCM_PATH)
    set(ENV{HIP_PATH} "${ROCM_PATH}")
    set(ENV{HCC_HOME} "${ROCM_PATH}/hcc")
    set(ENV{ROCM_PATH} "${ROCM_PATH}")
endif()

# find HIP
find_package(HIP QUIET HINTS ${ROCM_PATH})

if(HIP_FOUND)
    find_package(rocblas REQUIRED)
    find_package(rocrand REQUIRED)

    set(ROCM_FOUND TRUE)
endif()

# exoport variables
mark_as_advanced(ROCM_FOUND ROCM_PATH)