# Find Huawei Ascend Toolkit
# This module defines the following variables:
#  HUAWEI_FOUND - True if Huawei toolkit is found
#  HUAWEI_INCLUDE_DIRS - The include directories for Huawei toolkit
#  HUAWEI_LIBRARIES - The libraries for Huawei toolkit
#  HUAWEI_HAL_LIBRARIES - The libascend_hal.so library for Huawei toolkit

# Find acl.h include directory
find_path(HUAWEI_INCLUDE_DIRS
    NAMES "acl/acl.h"
    PATHS
        $ENV{ASCEND_HOME}/ascend-toolkit/latest/include
        /usr/local/Ascend/ascend-toolkit/latest/include
)

# Find acl library
find_library(HUAWEI_LIBRARIES
    NAMES "acl"
    PATHS
        $ENV{ASCEND_HOME}/ascend-toolkit/latest/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
        /usr/lib/aarch64-linux-gnu
)

# Find libascendcl.so library
find_library(ASCENDCL_LIBRARY
    NAMES "ascendcl"
    PATHS
        $ENV{ASCEND_HOME}/ascend-toolkit/latest/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
)

# Find libascend_hal.so library
find_library(HUAWEI_HAL_LIBRARIES
    NAMES "libascend_hal.so"
    PATHS
        $ENV{ASCEND_HOME}/ascend-toolkit/latest/aarch64-linux/devlib
        $ENV{ASCEND_HOME}/ascend-toolkit/latest/runtime/lib64/stub
        /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib
        /usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/stub
        /usr/local/Ascend/driver/lib64/driver
)

set(HUAWEI_LIB_DIRS
    $ENV{ASCEND_HOME}/ascend-toolkit/latest/lib64
    /usr/local/Ascend/ascend-toolkit/latest/lib64
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/devlib
)

# 合并 HUAWEI_LIBRARIES 变量，确保包含 acl 和 ascendcl
set(HUAWEI_LIBRARIES ${HUAWEI_ACL_LIBRARY} ${ASCENDCL_LIBRARY})

# Check if everything is found
if(HUAWEI_INCLUDE_DIRS AND HUAWEI_LIBRARIES AND HUAWEI_HAL_LIBRARIES)
    set(HUAWEI_FOUND TRUE)
    message(STATUS "Found Huawei Ascend Toolkit")
    message(STATUS "  Include dirs: ${HUAWEI_INCLUDE_DIRS}")
    message(STATUS "  ACL library: ${HUAWEI_ACL_LIBRARY}")
    message(STATUS "  AscendCL library: ${ASCENDCL_LIBRARY}")
    message(STATUS "  HAL library: ${HUAWEI_HAL_LIBRARIES}")
else()
    set(HUAWEI_FOUND FALSE)
    message(STATUS "Huawei Ascend Toolkit not found.")
    if(NOT HUAWEI_INCLUDE_DIRS)
        message(STATUS "  Missing include dirs (acl/acl.h).")
    endif()
    if(NOT HUAWEI_ACL_LIBRARY)
        message(STATUS "  Missing ACL library (libacl.so).")
    endif()
    if(NOT ASCENDCL_LIBRARY)
        message(STATUS "  Missing AscendCL library (libascendcl.so).")
    endif()
    if(NOT HUAWEI_HAL_LIBRARIES)
        message(STATUS "  Missing HAL library (libascend_hal.so).")
    endif()
endif()
# Mark variables as advanced
mark_as_advanced(HUAWEI_INCLUDE_DIRS HUAWEI_LIBRARIES HUAWEI_HAL_LIBRARIES)

# Check if ASCEND_HOME environment variable is set
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "The environment variable ASCEND_HOME is not set. Please set it to the root directory of your Huawei Ascend installation.")
else()
    message(STATUS "Using ASCEND_HOME: $ENV{ASCEND_HOME}")
endif()

