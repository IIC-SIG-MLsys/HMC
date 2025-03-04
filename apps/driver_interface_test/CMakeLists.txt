# apps/dit/CMakeLists.txt
project(dit)
set(DIT_SOURCES
    main.cpp
)

add_executable(dit ${DIT_SOURCES})
if(ROCM_FOUND)
    target_link_libraries(dit PRIVATE hddt_rocm_shared)
    target_link_libraries(dit PRIVATE hddt_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(dit PRIVATE hddt_cuda_shared)
    target_link_libraries(dit PRIVATE hddt_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(dit PRIVATE hddt_neuware_shared)
    target_link_libraries(dit PRIVATE hddt_shared_lib ${NEUWARE_LIBRARIES})
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE is required but was not found.")
endif()
target_include_directories(dit PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
