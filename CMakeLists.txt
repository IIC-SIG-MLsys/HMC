cmake_minimum_required(VERSION 3.21)
project(HMC)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# # cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


# # option: you can use follow cmd to open some features
# -DBUILD_STATIC_LIB=ON
# -DBUILD_PYTHON_MOD=ON
if(BUILD_PYTHON_MOD)
    set(BUILD_STATIC_LIB ON)
endif()

# using clang
# set(CMAKE_CXX_COMPILER "clang++" CACHE STRING "C++ compiler" FORCE)

# # build gpu-related module
include(FindRocm)
include(FindCuda)
include(FindNeuware)
include(FindHuawei)

if(ROCM_FOUND)
    # pass var to code
    add_definitions(-DENABLE_ROCM)
    set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE) # we should use hipcc
    enable_language(HIP)
    add_definitions(-D__HIP_PLATFORM_AMD__)
elseif(CUDA_FOUND)
    # pass var to code
    add_definitions(-DENABLE_CUDA)
    enable_language(CUDA)
elseif(NEUWARE_FOUND)
    # pass var to code
    add_definitions(-DENABLE_NEUWARE)
    include_directories(${NEUWARE_INCLUDE_DIRS})
elseif(HUAWEI_FOUND)
    add_definitions(-DENABLE_HUAWEI)
    include_directories(${HUAWEI_INCLUDE_DIRS})
    link_directories(${HUAWEI_LIB_DIRS})
else()
    message(WARNING "ROCm or CUDA or NEUWARE or HUAWEI is required but was not found. Only support CPU")
endif()

# # build src
include_directories(include) # 向外暴露的头文件
include_directories(src) # 源文件下的其他头文件

# with rdma
include(FindRdma)

file(GLOB_RECURSE SRC_RECURSE "src/*.cpp")
if(NOT SRC_RECURSE)
    message(FATAL_ERROR "No sources files found in the 'src' directory.")
endif()

# static
if(BUILD_STATIC_LIB)
    add_library(hmc_static_lib STATIC ${SRC_RECURSE})
    target_include_directories(hmc_static_lib PUBLIC include)

    # with rdma
    if(RDMA_FOUND)
        target_include_directories(hmc_static_lib PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
        target_link_libraries(hmc_static_lib PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})
    else()
        message(FATAL_ERROR "RDMA is required but was not found.")
    endif()

    # with glog
    find_package(glog REQUIRED)
    target_include_directories(hmc_static_lib PUBLIC ${GLOG_INCLUDE_DIRS})
    target_link_libraries(hmc_static_lib PUBLIC glog)

    install(TARGETS hmc_static_lib DESTINATION lib)
endif()

# dynamic
add_library(hmc_shared_lib SHARED ${SRC_RECURSE})
target_include_directories(hmc_shared_lib PUBLIC include)

# with rdma
if(RDMA_FOUND)
    target_include_directories(hmc_shared_lib PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
    target_link_libraries(hmc_shared_lib PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})
else()
    message(FATAL_ERROR "RDMA is required but was not found.")
endif()

# with glog
find_package(glog REQUIRED)
target_include_directories(hmc_shared_lib PUBLIC ${GLOG_INCLUDE_DIRS})
target_link_libraries(hmc_shared_lib PUBLIC glog)

# python interface
if(BUILD_PYTHON_MOD)
    # # pybind
    # git submodule add -b stable ../../pybind/pybind11 extern/pybind11
    # git submodule update --init
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
    # add python module
    file(GLOB_RECURSE PYTHON_SRC_RECURSE "python/*.cpp")
    if(NOT PYTHON_SRC_RECURSE)
        message(FATAL_ERROR "No sources files found in the 'python' directory.")
    endif()
    pybind11_add_module(hmc ${PYTHON_SRC_RECURSE})

    # link C++ lib to Python module
    target_link_libraries(hmc PRIVATE hmc_static_lib)
    install(TARGETS hmc LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/pkg)
endif()

install(DIRECTORY include/ DESTINATION include)


# # build apps
add_subdirectory(apps/rdma_app)
add_subdirectory(apps/comm_app)
# add_subdirectory(apps/python_app)

# # build test
# enable_testing()
# add_subdirectory(tests)
