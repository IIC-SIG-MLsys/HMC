project(uhm_app)

set(ENV{ASCEND_RT_VISIBLE_DEVICES} "0,1")

set(RDMA_SERVER_SOURCES server.cpp)
set(RDMA_CLIENT_SOURCES client.cpp)

add_executable(uhm_server ${RDMA_SERVER_SOURCES})
add_executable(uhm_client ${RDMA_CLIENT_SOURCES})

# 新增摩尔线程支持
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
elseif(MUSA_FOUND)  # 新增分支
    target_link_libraries(uhm_server PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
    target_link_libraries(uhm_client PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
else()
    message(WARNING "ROCm, CUDA, NEUWARE, HUAWEI, or Moore Threads not found. Defaulting to CPU support.")
endif()

# 包含头文件目录
target_include_directories(uhm_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>  # 仅在 MUSA 启用时添加头文件
)
target_include_directories(uhm_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)

# 链接glog
target_link_libraries(uhm_server PUBLIC glog)
target_link_libraries(uhm_client PUBLIC glog)
