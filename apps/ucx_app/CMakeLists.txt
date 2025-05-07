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

# Link with main shared library instead of platform-specific libraries
target_link_libraries(ucx_server PRIVATE hmc_shared_lib)
target_link_libraries(ucx_client PRIVATE hmc_shared_lib)
target_link_libraries(ucx_mem PRIVATE hmc_shared_lib)

# Add glog dependency
target_link_libraries(ucx_server PUBLIC glog)
target_link_libraries(ucx_client PUBLIC glog)
target_link_libraries(ucx_mem PUBLIC glog)

# Add UCX libraries directly
if(UCX_FOUND)
    target_link_libraries(ucx_server PRIVATE ${UCX_LIBRARIES})
    target_link_libraries(ucx_client PRIVATE ${UCX_LIBRARIES})
    target_link_libraries(ucx_mem PRIVATE ${UCX_LIBRARIES})
endif()

# Add MPI dependency if found
include(FindMPI)
if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_DIR})
    target_link_libraries(ucx_server PUBLIC ${MPI_LIBRARY})
    target_link_libraries(ucx_client PUBLIC ${MPI_LIBRARY})
    target_link_libraries(ucx_mem PUBLIC ${MPI_LIBRARY})
endif()