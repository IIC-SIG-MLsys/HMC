cmake_minimum_required(VERSION 3.21)
project(HDD_Test)

# 启用 C++ 语言
enable_language(CXX)

include_directories(include)
# 指定生成的可执行文件名称
set(BINARY ${CMAKE_PROJECT_NAME}_test)

# 查找测试源文件
file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/src/mem/*.cpp ${CMAKE_SOURCE_DIR}/src/* .cpp)

# 查找测试文件
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/tests/huawei_memory_test.cpp)

# 创建可执行文件
add_executable(${BINARY} ${TEST_SOURCES} ${SOURCES})

# 添加测试
enable_testing()
add_test(NAME ${BINARY} COMMAND ${BINARY})


# 添加包含目录
target_include_directories(${BINARY} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 链接 Google Test 库
find_package(GTest REQUIRED)
target_link_libraries(${BINARY} PRIVATE GTest::GTest GTest::Main)

# 检查环境变量
if(NOT DEFINED ENV{ASCEND_HOME})
    message(FATAL_ERROR "Environment variable ASCEND_HOME is not set. Please set it to the Huawei Ascend installation directory.")
else()
    message(STATUS "Using ASCEND_HOME: $ENV{ASCEND_HOME}")
endif()

# 确保 Ascend 库路径存在
set(ASCEND_LIBRARY_PATH "$ENV{ASCEND_HOME}/ascend-toolkit/latest/lib64")
set(ASCEND_DEVLIB_PATH "$ENV{ASCEND_HOME}/ascend-toolkit/latest/aarch64-linux/devlib")
if(NOT EXISTS ${ASCEND_LIBRARY_PATH})
    message(FATAL_ERROR "Ascend library path not found: ${ASCEND_LIBRARY_PATH}")
endif()
if(NOT EXISTS ${ASCEND_DEVLIB_PATH})
    message(FATAL_ERROR "Ascend devlib path not found: ${ASCEND_DEVLIB_PATH}")
endif()

# 显示当前链接路径
message(STATUS "Ascend Library Path: ${ASCEND_LIBRARY_PATH}")
message(STATUS "Ascend Devlib Path: ${ASCEND_DEVLIB_PATH}")

# 链接 Huawei Ascend 库
find_library(ASCENDCL_LIBRARY ascendcl HINTS ${ASCEND_LIBRARY_PATH})
find_library(ASCEND_HAL_LIBRARY ascend_hal HINTS ${ASCEND_DEVLIB_PATH})

if(NOT ASCENDCL_LIBRARY OR NOT ASCEND_HAL_LIBRARY)
    message(FATAL_ERROR "Huawei Ascend libraries not found.")
endif()

target_link_libraries(${BINARY} PUBLIC 
    ${ASCENDCL_LIBRARY} 
    ${ASCEND_HAL_LIBRARY}
)

# 添加运行时 RPATH
set_target_properties(${BINARY} PROPERTIES
    BUILD_RPATH "${ASCEND_LIBRARY_PATH}:${ASCEND_DEVLIB_PATH}"
    INSTALL_RPATH "${ASCEND_LIBRARY_PATH}:${ASCEND_DEVLIB_PATH}"
)

# 添加 MPI 配置
find_package(MPI REQUIRED)

# 确保变量正确传递
set(MPI_CXX_LIBRARY ${MPI_CXX_LIBRARY} CACHE INTERNAL "MPI C++ Library")
set(MPI_INCLUDE_DIR ${MPI_INCLUDE_DIR} CACHE INTERNAL "MPI Include Directory")
set(MPI_LIBRARY ${MPI_LIBRARY} CACHE INTERNAL "MPI Library")

# 确保 MPI 变量不为空
if(NOT DEFINED MPI_CXX_LIBRARY OR "${MPI_CXX_LIBRARY}" STREQUAL "")
    message(FATAL_ERROR "MPI C++ library not found or empty: ${MPI_CXX_LIBRARY}")
endif()

if(NOT DEFINED MPI_LIBRARY OR "${MPI_LIBRARY}" STREQUAL "")
    message(FATAL_ERROR "MPI library not found or empty: ${MPI_LIBRARY}")
endif()

if(NOT DEFINED MPI_INCLUDE_DIR OR "${MPI_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "MPI include directory not found: ${MPI_INCLUDE_DIR}")
endif()


# 确保 `MPI_INCLUDE_DIR` 传递到测试
target_include_directories(${BINARY} PRIVATE ${MPI_INCLUDE_DIR})

# 链接 MPI 库
target_link_libraries(${BINARY} PUBLIC ${MPI_LIBRARY} ${MPI_CXX_LIBRARY})



# 链接 RDMA 库
find_library(IBVERBS_LIBRARY ibverbs HINTS /usr/lib64 /lib64)
find_library(RDMACM_LIBRARY rdmacm HINTS /usr/lib64 /lib64)

if(NOT IBVERBS_LIBRARY OR NOT RDMACM_LIBRARY)
    message(FATAL_ERROR "RDMA libraries not found.")
endif()

target_link_libraries(${BINARY} PUBLIC ${IBVERBS_LIBRARY} ${RDMACM_LIBRARY})

# 添加 glog 支持
find_package(glog REQUIRED)
target_include_directories(${BINARY} PRIVATE ${GLOG_INCLUDE_DIRS})
target_link_libraries(${BINARY} PRIVATE glog)
