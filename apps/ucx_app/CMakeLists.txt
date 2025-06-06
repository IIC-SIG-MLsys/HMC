project(ucx_app)

set(UCX_SERVER_SOURCES
    ucx_server.cpp
)
set(UCX_CLIENT_SOURCES
    ucx_client.cpp
)
set(UCX_MEM_SOURCES
    ucx_mem.cpp
)

add_executable(ucx_server ${UCX_SERVER_SOURCES})
add_executable(ucx_client ${UCX_CLIENT_SOURCES})
add_executable(ucx_mem ${UCX_MEM_SOURCES})

# Include directories
target_include_directories(ucx_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(ucx_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(ucx_mem PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Link with main shared library
target_link_libraries(ucx_server PRIVATE hmc_shared_lib)
target_link_libraries(ucx_client PRIVATE hmc_shared_lib)
target_link_libraries(ucx_mem PRIVATE hmc_shared_lib)

# Platform-specific libraries
if(ROCM_FOUND)
    target_link_libraries(ucx_server PRIVATE ${HIP_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${HIP_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(ucx_server PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${CUDA_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(ucx_server PRIVATE ${NEUWARE_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${NEUWARE_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${NEUWARE_LIBRARIES})
elseif(HUAWEI_FOUND)
    target_link_libraries(ucx_server PRIVATE ${HUAWEI_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${HUAWEI_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${HUAWEI_LIBRARIES})
elseif(MUSA_FOUND)
    target_link_libraries(ucx_server PRIVATE ${MUSA_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${MUSA_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${MUSA_LIBRARIES})
endif()

# Add glog dependency
target_link_libraries(ucx_server PUBLIC glog)
target_link_libraries(ucx_client PUBLIC glog)
target_link_libraries(ucx_mem PUBLIC glog)

# Add UCX libraries if found
if(UCX_FOUND)
    target_link_libraries(ucx_server PRIVATE ${UCX_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${UCX_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${UCX_LIBRARIES})
else()
    message(WARNING "UCX not found, ucx_app may not work properly")
endif()