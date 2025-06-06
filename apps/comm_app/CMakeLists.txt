project(comm_app)

set(RDMA_MEM_SOURCES
    mem_test.cpp
)
set(RDMA_COMM_SERVER_SOURCES
    comm_server.cpp
)
set(RDMA_COMM_CLIENT_SOURCES
    comm_client.cpp
)

add_executable(mem_test ${RDMA_MEM_SOURCES})
add_executable(comm_server ${RDMA_COMM_SERVER_SOURCES})
add_executable(comm_client ${RDMA_COMM_CLIENT_SOURCES})

# 根据后端平台链接对应库
if(ROCM_FOUND)
    target_link_libraries(mem_test PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(mem_test PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(mem_test PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
elseif(MUSA_FOUND)
    # 添加 MUSA 支持
    target_link_libraries(mem_test PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hmc_shared_lib ${MUSA_LIBRARIES})
else()
    target_link_libraries(mem_test PRIVATE hmc_shared_lib)
    target_link_libraries(comm_server PRIVATE hmc_shared_lib)
    target_link_libraries(comm_client PRIVATE hmc_shared_lib)
    message(WARNING "No GPU platform found, using CPU only")
endif()

# 包含头文件目录
target_include_directories(mem_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>  # 仅在 MUSA 启用时添加头文件
)
target_include_directories(comm_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)
target_include_directories(comm_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)

# 添加UCX支持 - 如果UCX可用则链接
if(UCX_FOUND)
    target_link_libraries(mem_test PRIVATE ${UCX_LIBRARIES})
    target_link_libraries(comm_server PRIVATE ${UCX_LIBRARIES})
    target_link_libraries(comm_client PRIVATE ${UCX_LIBRARIES})
    
    # 添加UCX包含目录
    target_include_directories(mem_test PRIVATE ${UCX_INCLUDE_DIR})
    target_include_directories(comm_server PRIVATE ${UCX_INCLUDE_DIR})
    target_include_directories(comm_client PRIVATE ${UCX_INCLUDE_DIR})
    
    message(STATUS "UCX libraries linked to comm_app: ${UCX_LIBRARIES}")
endif()

# 链接 GLOG
target_link_libraries(mem_test PUBLIC glog)
target_link_libraries(comm_server PUBLIC glog)
target_link_libraries(comm_client PUBLIC glog)