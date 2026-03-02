cmake_minimum_required(VERSION 3.10)
project(uhm_app)

set(ENV{ASCEND_RT_VISIBLE_DEVICES} "0,1")

add_executable(uhm_server server.cpp)
add_executable(uhm_client client.cpp)

target_link_libraries(uhm_server PRIVATE hmc_shared_lib)
target_link_libraries(uhm_client PRIVATE hmc_shared_lib)

if(CUDA_FOUND)
  target_link_libraries(uhm_server PRIVATE ${CUDA_LIBRARIES})
  target_link_libraries(uhm_client PRIVATE ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
  target_link_libraries(uhm_server PRIVATE ${NEUWARE_LIBRARIES})
  target_link_libraries(uhm_client PRIVATE ${NEUWARE_LIBRARIES})
elseif(MUSA_FOUND)
  target_link_libraries(uhm_server PRIVATE ${MUSA_LIBRARIES})
  target_link_libraries(uhm_client PRIVATE ${MUSA_LIBRARIES})
endif()

target_include_directories(uhm_server PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)
target_include_directories(uhm_client PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>
)
