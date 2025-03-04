# apps/comm_app/CMakeLists.txt
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

add_executable(mem ${RDMA_MEM_SOURCES})
add_executable(comm_server ${RDMA_COMM_SERVER_SOURCES})
add_executable(comm_client ${RDMA_COMM_CLIENT_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(mem PRIVATE hddt_rocm_shared)
    target_link_libraries(mem PRIVATE hddt_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hddt_rocm_shared)
    target_link_libraries(comm_server PRIVATE hddt_shared_lib ${HIP_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hddt_rocm_shared)
    target_link_libraries(comm_client PRIVATE hddt_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(mem PRIVATE hddt_cuda_shared)
    target_link_libraries(mem PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hddt_cuda_shared)
    target_link_libraries(comm_server PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hddt_cuda_shared)
    target_link_libraries(comm_client PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(mem PRIVATE hddt_neuware_shared)
    target_link_libraries(mem PRIVATE hddt_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(comm_server PRIVATE hddt_neuware_shared)
    target_link_libraries(comm_server PRIVATE hddt_shared_lib ${NEUWARE_LIBRARIES})
    target_link_libraries(comm_client PRIVATE hddt_neuware_shared)
    target_link_libraries(comm_client PRIVATE hddt_shared_lib ${NEUWARE_LIBRARIES})
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
endif()

target_include_directories(mem PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(comm_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(comm_client PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# install(TARGETS comm_app DESTINATION bin)
target_link_libraries(mem PUBLIC glog)
target_link_libraries(comm_server PUBLIC glog)
target_link_libraries(comm_client PUBLIC glog)

# use mpicc as compiler
if(ROCM_FOUND)
else()
    find_program(MPICC mpicxx REQUIRED)
    set(CMAKE_C_COMPILER ${MPICC})
    set(CMAKE_CXX_COMPILER ${MPICC})
endif()