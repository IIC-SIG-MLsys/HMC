cmake_minimum_required(VERSION 3.21)
project(HMC)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CMake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# options
if(BUILD_PYTHON_MOD)
    set(BUILD_STATIC_LIB ON)
endif()

# UCX 支持
set(UCX_ROOT /usr/local/ucx)
if(EXISTS "${UCX_ROOT}/include/ucp/api/ucp.h")
    message(STATUS "UCX include dir detected: ${UCX_ROOT}/include")
    include_directories(${UCX_ROOT}/include)
    link_directories(${UCX_ROOT}/lib)
else()
    message(WARNING "UCX not found at ${UCX_ROOT}. If installed elsewhere, update UCX_ROOT.")
endif()

# 查找平台支持模块
include(FindRocm)
include(FindCuda)
include(FindNeuware)
include(FindHuawei)
include(FindMUSA)

# 后端平台设置
if(ROCM_FOUND)
    add_definitions(-DENABLE_ROCM)
    set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE)
    enable_language(HIP)
    add_definitions(-D__HIP_PLATFORM_AMD__)
elseif(CUDA_FOUND)
    add_definitions(-DENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    enable_language(CUDA)
elseif(NEUWARE_FOUND)
    add_definitions(-DENABLE_NEUWARE)
    include_directories(${NEUWARE_INCLUDE_DIRS})
elseif(HUAWEI_FOUND)
    add_definitions(-DENABLE_HUAWEI)
    include_directories(${HUAWEI_INCLUDE_DIRS})
    link_directories(${HUAWEI_LIB_DIRS})
elseif(MUSA_FOUND)
    add_definitions(-DENABLE_MUSA)
    include_directories(${MUSA_INCLUDE_DIRS})
    link_directories(${MUSA_LIB_DIRS})
else()
    message(WARNING "ROCm or CUDA or NEUWARE or HUAWEI or MUSA is required but was not found. Only support CPU")
endif()

# 公共头文件
include_directories(include)
include_directories(src)

# RDMA 支持
include(FindRdma)

file(GLOB_RECURSE SRC_RECURSE "src/*.cpp")
if(NOT SRC_RECURSE)
    message(FATAL_ERROR "No sources files found in the 'src' directory.")
endif()

# ===== STATIC LIB =====
if(BUILD_STATIC_LIB)
    add_library(hmc_static_lib STATIC ${SRC_RECURSE})
    target_include_directories(hmc_static_lib PUBLIC include)

    # RDMA
    if(RDMA_FOUND)
        target_include_directories(hmc_static_lib PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
        target_link_libraries(hmc_static_lib PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})
    else()
        message(FATAL_ERROR "RDMA is required but was not found.")
    endif()

    # GLOG
    find_package(glog REQUIRED)
    target_include_directories(hmc_static_lib PUBLIC ${GLOG_INCLUDE_DIRS})
    target_link_libraries(hmc_static_lib PUBLIC glog)

    # MUSA
    if(MUSA_FOUND)
        target_link_libraries(hmc_static_lib PUBLIC hmc_musa_shared ${MUSA_LIBRARIES})
    endif()

    install(TARGETS hmc_static_lib DESTINATION lib)
endif()

# ===== SHARED LIB =====
add_library(hmc_shared_lib SHARED ${SRC_RECURSE})
target_include_directories(hmc_shared_lib PUBLIC include)

if(RDMA_FOUND)
    target_include_directories(hmc_shared_lib PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
    target_link_libraries(hmc_shared_lib PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})
else()
    message(FATAL_ERROR "RDMA is required but was not found.")
endif()

find_package(glog REQUIRED)
target_include_directories(hmc_shared_lib PUBLIC ${GLOG_INCLUDE_DIRS})
target_link_libraries(hmc_shared_lib PUBLIC glog)

if(MUSA_FOUND)
    target_link_libraries(hmc_shared_lib PUBLIC ${MUSA_LIBRARIES})
    #target_link_libraries(hmc_shared_lib PUBLIC hmc_musa_shared ${MUSA_LIBRARIES})
endif()

# ===== PYTHON MODULE =====
if(BUILD_PYTHON_MOD)
    execute_process(
        COMMAND git submodule update --init
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMOD_RESULT
    )
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()

    add_subdirectory(extern/pybind11)
    set(PYBIND11_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/extern/pybind11/include)
    include_directories(${PYBIND11_INCLUDE_DIR})

    file(GLOB_RECURSE PYTHON_SRC_RECURSE "python/*.cpp")
    if(NOT PYTHON_SRC_RECURSE)
        message(FATAL_ERROR "No sources files found in the 'python' directory.")
    endif()

    pybind11_add_module(hmc ${PYTHON_SRC_RECURSE})

    target_link_libraries(hmc PRIVATE hmc_static_lib)
    
    if(CUDA_FOUND)
        target_link_libraries(hmc PRIVATE ${CUDA_LIBRARIES})
    endif()

    if(MUSA_FOUND)
        target_link_libraries(hmc PRIVATE hmc_musa_shared ${MUSA_LIBRARIES})
    endif()

    install(TARGETS hmc LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/pkg)
endif()

install(DIRECTORY include/ DESTINATION include)

# ===== APPS =====
# add_subdirectory(apps/rdma_app)
add_subdirectory(apps/uhm_app)
# add_subdirectory(apps/comm_app)
# add_subdirectory(apps/python_app)

# ===== TESTS =====
# enable_testing()
# add_subdirectory(tests)
