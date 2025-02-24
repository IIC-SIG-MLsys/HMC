cmake_minimum_required(VERSION 3.21)
project(HDDT)
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

if(ROCM_FOUND)
    # pass var to code
    add_definitions(-DENABLE_ROCM)
    set(CMAKE_CXX_COMPILER "hipcc" CACHE STRING "C++ compiler" FORCE) # we should use hipcc
    enable_language(HIP)
    add_definitions(-D__HIP_PLATFORM_AMD__)

    file(GLOB_RECURSE ROCM_SOURCES "src/*/rocm/*.hip")

    # static
    if(BUILD_STATIC_LIB)
        add_library(hddt_rocm_static STATIC ${ROCM_SOURCES})
        target_include_directories(hddt_rocm_static PUBLIC ${HIP_INCLUDE_DIRS})
        target_link_libraries(hddt_rocm_static PRIVATE ${HIP_LIBRARIES})
        install(TARGETS hddt_rocm_static DESTINATION lib)
    endif()

    # dynamic
    add_library(hddt_rocm_shared SHARED ${ROCM_SOURCES})
    target_include_directories(hddt_rocm_shared PUBLIC ${HIP_INCLUDE_DIRS})
    target_link_libraries(hddt_rocm_shared PRIVATE ${HIP_LIBRARIES})
    install(TARGETS hddt_rocm_shared DESTINATION lib)

elseif(CUDA_FOUND)
    # pass var to code
    add_definitions(-DENABLE_CUDA)

    enable_language(CUDA)

    file(GLOB_RECURSE CUDA_SOURCES "src/*/cuda/*.cu")

    # static
    if(BUILD_STATIC_LIB)
        add_library(hddt_cuda_static STATIC ${CUDA_SOURCES})
        target_include_directories(hddt_cuda_static PUBLIC ${CUDA_INCLUDE_DIRS})
        target_link_libraries(hddt_cuda_static PRIVATE ${CUDA_LIBRARIES})
        install(TARGETS hddt_cuda_static DESTINATION lib)
    endif()

    # dynamic
    add_library(hddt_cuda_shared SHARED ${CUDA_SOURCES})
    target_include_directories(hddt_cuda_shared PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(hddt_cuda_shared PRIVATE ${CUDA_LIBRARIES})
    install(TARGETS hddt_cuda_shared DESTINATION lib)

elseif(NEUWARE_FOUND)
    # pass var to code
    add_definitions(-DENABLE_NEUWARE)
    file(GLOB_RECURSE NEUWARE_SOURCES "src/*/neuware/*.mlu")

    include_directories(${NEUWARE_INCLUDE_DIRS})
    # static
    if(BUILD_STATIC_LIB)
        bang_add_library(hddt_neuware_static STATIC ${NEUWARE_SOURCES})  # we should use bang_add_library for neuware
        target_link_libraries(hddt_neuware_static PRIVATE ${NEUWARE_LIBRARIES})
        install(TARGETS hddt_neuware_static DESTINATION lib)
    endif()

    # dynamic
    bang_add_library(hddt_neuware_shared SHARED ${NEUWARE_SOURCES})
    target_link_libraries(hddt_neuware_shared PRIVATE ${NEUWARE_LIBRARIES})
    install(TARGETS hddt_neuware_shared DESTINATION lib)

else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
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
    add_library(hddt_static_lib STATIC ${SRC_RECURSE})
    target_include_directories(hddt_static_lib PUBLIC include)
    if(ROCM_FOUND)
        target_link_libraries(hddt_static_lib PUBLIC hddt_rocm_static)
    elseif(CUDA_FOUND)
        target_link_libraries(hddt_static_lib PUBLIC hddt_cuda_static)
    elseif(NEUWARE_FOUND)
        target_link_libraries(hddt_static_lib PUBLIC hddt_neuware_static)
    else()
        message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
    endif()

    # with rdma
    if(RDMA_FOUND)
        target_include_directories(hddt_static_lib PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
        target_link_libraries(hddt_static_lib PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})
    else()
        message(FATAL_ERROR "RDMA is required but was not found.")
    endif()

    # with glog
    find_package(glog REQUIRED)
    target_include_directories(hddt_static_lib PUBLIC ${GLOG_INCLUDE_DIRS})
    target_link_libraries(hddt_static_lib PUBLIC glog)

    # with openmpi
    include(FindMPI)
    target_include_directories(hddt_static_lib PUBLIC ${MPI_INCLUDE_DIR})
    target_link_libraries(hddt_static_lib PUBLIC ${MPI_LIBRARY})
    
    install(TARGETS hddt_static_lib DESTINATION lib)
endif()

# dynamic
add_library(hddt_shared_lib SHARED ${SRC_RECURSE})
target_include_directories(hddt_shared_lib PUBLIC include)
if(ROCM_FOUND)
    target_link_libraries(hddt_shared_lib PUBLIC hddt_rocm_shared)
elseif(CUDA_FOUND)
    target_link_libraries(hddt_shared_lib PUBLIC hddt_cuda_shared)
elseif(NEUWARE_FOUND)
    target_link_libraries(hddt_shared_lib PUBLIC hddt_neuware_shared)
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
endif()

# with rdma
if(RDMA_FOUND)
    target_include_directories(hddt_shared_lib PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
    target_link_libraries(hddt_shared_lib PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})
else()
    message(FATAL_ERROR "RDMA is required but was not found.")
endif()

# with glog
find_package(glog REQUIRED)
target_include_directories(hddt_shared_lib PUBLIC ${GLOG_INCLUDE_DIRS})
target_link_libraries(hddt_shared_lib PUBLIC glog)

# with openmpi
include(FindMPI)
include_directories(${MPI_INCLUDE_DIR})
target_link_libraries(hddt_shared_lib PUBLIC ${MPI_LIBRARY})

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
    pybind11_add_module(hddt ${PYTHON_SRC_RECURSE})

    # link C++ lib to Python module
    target_link_libraries(hddt PRIVATE hddt_static_lib)
    install(TARGETS hddt LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/pkg)
endif()

install(DIRECTORY include/ DESTINATION include)


# # build apps
# add_subdirectory(apps/train_cnn)
# add_subdirectory(apps/simple_inference)
# add_subdirectory(apps/torch_app)
# add_subdirectory(apps/rdma_app)
# add_subdirectory(apps/coll_app)
add_subdirectory(apps/comm_app)

# # build test
# add_subdirectory(tests)
