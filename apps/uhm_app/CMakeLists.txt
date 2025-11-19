cmake_minimum_required(VERSION 3.10)
project(uhm_app)

set(ENV{ASCEND_RT_VISIBLE_DEVICES} "0,1")

set(RDMA_SERVER_SOURCES server.cpp)
set(RDMA_CLIENT_SOURCES client.cpp)

add_executable(uhm_server ${RDMA_SERVER_SOURCES})
add_executable(uhm_client ${RDMA_CLIENT_SOURCES})

# ==============================
# 1. GPU / AI 加速库检测
# ==============================
if(ROCM_FOUND)
    target_link_libraries(uhm_server PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(uhm_client PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(uhm_server PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(uhm_client PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(uhm_server PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(uhm_client PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
elseif(HUAWEI_FOUND)
    target_link_libraries(uhm_server PRIVATE hmc_shared_lib ${HUAWEI_LIBRARIES})
    target_link_libraries(uhm_client PRIVATE hmc_shared_lib ${HUAWEI_LIBRARIES})
elseif(MUSA_FOUND)
    target_link_libraries(uhm_server PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
    target_link_libraries(uhm_client PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
else()
    message(WARNING "ROCm, CUDA, NEUWARE, HUAWEI, or Moore Threads not found. Defaulting to CPU support.")
endif()

# ====== 关键修复 1: 在链接前设置路径 ======
if(UCX_FOUND)
    # 全局包含目录 (影响所有子项目)
    include_directories(
        ${UCX_INCLUDE_DIRS}      # 公共头文件
    )
    
    # 全局链接目录
    link_directories(${UCX_LIBRARY_DIRS})
endif()

if(UCX_FOUND)
    target_link_libraries(uhm_server PRIVATE 
        ${UCX_LIBRARIES}
        pthread rt dl
    )
    target_link_libraries(uhm_client PRIVATE 
        ${UCX_LIBRARIES}
        pthread rt dl
    )
endif()

# ==============================
# 2. UCX 配置（适配版）
# ==============================
set(UCX_ROOT /usr/local/ucx)

set(UCX_INCLUDE_DIR ${UCX_ROOT}/include)
set(UCX_LIBRARY_DIR ${UCX_ROOT}/lib)

# UCX 组件库
find_library(UCP_LIB ucp HINTS ${UCX_LIBRARY_DIR})
find_library(UCS_LIB ucs HINTS ${UCX_LIBRARY_DIR})
find_library(UCT_LIB uct HINTS ${UCX_LIBRARY_DIR})
find_library(UCM_LIB ucm HINTS ${UCX_LIBRARY_DIR})

if(UCP_LIB AND UCS_LIB AND UCT_LIB)
    message(STATUS "✅ UCX found successfully:")
    message(STATUS "   Include: ${UCX_INCLUDE_DIR}")
    message(STATUS "   Libraries:")
    message(STATUS "      ${UCP_LIB}")
    message(STATUS "      ${UCS_LIB}")
    message(STATUS "      ${UCT_LIB}")
    message(STATUS "      ${UCM_LIB}")
else()
    message(FATAL_ERROR "❌ UCX component libraries not found in ${UCX_LIBRARY_DIR}. Please check installation.")
endif()

# 包含 UCX 头文件
target_include_directories(uhm_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${UCX_INCLUDE_DIR}
)
target_include_directories(uhm_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${UCX_INCLUDE_DIR}
)

# 链接所有 UCX 组件库
target_link_libraries(uhm_server PRIVATE
    ${UCP_LIB}
    ${UCS_LIB}
    ${UCT_LIB}
    ${UCM_LIB}
    glog
)
target_link_libraries(uhm_client PRIVATE
    ${UCP_LIB}
    ${UCS_LIB}
    ${UCT_LIB}
    ${UCM_LIB}
    glog
)


# ==============================
# 3. Include 目录
# ==============================
target_include_directories(uhm_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${UCX_INCLUDE_DIR}
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)

target_include_directories(uhm_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${UCX_INCLUDE_DIR}
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)

# ==============================
# 4. 链接库
# ==============================
target_link_libraries(uhm_server PRIVATE ${UCX_LIB} glog)
target_link_libraries(uhm_client PRIVATE ${UCX_LIB} glog)

# ==============================
# 5. 额外调试信息
# ==============================
message(STATUS "💡 Summary:")
message(STATUS "   Server include dirs: ${CMAKE_SOURCE_DIR}/include, ${UCX_INCLUDE_DIR}")
message(STATUS "   UCX library: ${UCX_LIB}")
message(STATUS "   C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "   Build type: ${CMAKE_BUILD_TYPE}")
