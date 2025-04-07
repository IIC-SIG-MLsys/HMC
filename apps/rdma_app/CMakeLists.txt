project(rdma_app)

set(ENV{ASCEND_RT_VISIBLE_DEVICES} "0,1")

set(RDMA_SERVER_SOURCES
    server.cpp
)
set(RDMA_CLIENT_SOURCES
    client.cpp
)
set(RDMA_MEM_SOURCES
    mem.cpp
)

add_executable(rdma_server ${RDMA_SERVER_SOURCES})
add_executable(rdma_client ${RDMA_CLIENT_SOURCES})
add_executable(mem ${RDMA_MEM_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(rdma_server PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(rdma_client PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(mem PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(rdma_server PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(rdma_client PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(mem PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(rdma_server PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(rdma_client PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(mem PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
elseif(HUAWEI_FOUND)
    target_link_libraries(rdma_server PRIVATE hmc_shared_lib ${HUAWEI_LIBRARIES})
    target_link_libraries(rdma_client PRIVATE hmc_shared_lib ${HUAWEI_LIBRARIES})
    target_link_libraries(mem PRIVATE hmc_shared_lib ${HUAWEI_LIBRARIES})
else()
    message(WARNING "ROCm or CUDA or NEUWARE or HUAWEI is required but was not found. Only support CPU")
endif()
target_include_directories(rdma_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(rdma_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(mem PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# install(TARGETS rdma_app DESTINATION bin)
target_link_libraries(rdma_server PUBLIC glog)
target_link_libraries(rdma_client PUBLIC glog)
target_link_libraries(mem PUBLIC glog)
