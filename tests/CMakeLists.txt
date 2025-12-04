project(HMC_Test)

# 查找 Google Test 依赖
find_package(GTest REQUIRED)

# 自动收集所有 *_test.cpp 文件
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

set(BINARY run_all_tests)

# 指定生成的可执行文件名称
add_executable(${BINARY} ${TEST_SOURCES})

# 链接核心库和公共依赖
target_link_libraries(${BINARY}
    PRIVATE
    hmc_shared_lib       # 主共享库
    GTest::GTest         # GTest 核心库
    GTest::Main          # GTest 主函数
    ${MPI_LIBRARY} 
    ${MPI_CXX_LIBRARY}   # MPI 库
    ${VERBS_LIBRARIES} 
    ${RDMACM_LIBRARIES}  # RDMA 库
    glog                 # GLOG 日志库
)

# 根据后端平台添加额外链接库
if(ROCM_FOUND)
    target_link_libraries(${BINARY} PRIVATE ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(${BINARY} PRIVATE ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(${BINARY} PRIVATE ${NEUWARE_LIBRARIES})
elseif(MUSA_FOUND)
    target_link_libraries(${BINARY} PRIVATE ${MUSA_LIBRARIES})  # 直接链接 MUSA 运行时库
else()
    message(FATAL_ERROR "ROCm, CUDA, NEUWARE or MUSA is required but was not found.")
endif()

# 添加包含目录
target_include_directories(${BINARY}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_INCLUDE_DIR}
    ${VERBS_INCLUDE_DIR}
    ${RDMACM_INCLUDE_DIR}
    ${GLOG_INCLUDE_DIRS}
    $<$<BOOL:${MUSA_FOUND}>:${MUSA_INCLUDE_DIRS}>  # 仅在 MUSA 启用时添加
)
