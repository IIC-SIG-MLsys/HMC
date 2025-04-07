project(HMC_Test)

# 查找 Google Test 依赖
find_package(GTest REQUIRED)

# 自动收集所有 *_test.cpp 文件
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

set(BINARY run_all_tests)

# 指定生成的可执行文件名称
add_executable(${BINARY} ${TEST_SOURCES})

# 链接依赖库（动态库 + gtest_main）
target_link_libraries(run_all_tests
    PRIVATE
    GTest::GTest        # GTest 核心库
    GTest::Main         # 提供 main 函数（确保只有一个入口）
)
target_link_libraries(${BINARY} PRIVATE ${MPI_LIBRARY} ${MPI_CXX_LIBRARY}) # mpi
target_link_libraries(${BINARY} PRIVATE ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES}) # rdma
target_link_libraries(${BINARY} PRIVATE glog) # glog

if(ROCM_FOUND)
    target_link_libraries(${BINARY} PRIVATE hmc_rocm_shared)
    target_link_libraries(${BINARY} PRIVATE hmc_shared_lib ${HIP_LIBRARIES})
elseif(CUDA_FOUND)
    target_link_libraries(${BINARY} PRIVATE hmc_cuda_shared)
    target_link_libraries(${BINARY} PRIVATE hmc_shared_lib ${CUDA_LIBRARIES})
elseif(NEUWARE_FOUND)
    target_link_libraries(${BINARY} PRIVATE hmc_neuware_shared)
    target_link_libraries(${BINARY} PRIVATE hmc_shared_lib ${NEUWARE_LIBRARIES})
elseif(HUAWEI_FOUND)
    target_link_libraries(${BINARY} PRIVATE hmc_huawei_shared)
    target_link_libraries(${BINARY} PRIVATE hmc_shared_lib ${HUAWEI_LIBRARIES})
else()
    message(FATAL_ERROR "ROCm or CUDA or NEUWARE or HUAWEI is required but was not found.")
endif()

# 添加包含目录
target_include_directories(${BINARY} PRIVATE ${CMAKE_SOURCE_DIR}/include) # source header
target_include_directories(${BINARY} PRIVATE ${MPI_INCLUDE_DIR}) # mpi
target_include_directories(${BINARY} PRIVATE ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR}) # rdma
target_include_directories(${BINARY} PRIVATE ${GLOG_INCLUDE_DIRS}) # glog

if(ROCM_FOUND)
    # skip using mpicc while using hipcc as compiler
else()
    find_program(MPICC mpicxx REQUIRED)
    set(CMAKE_C_COMPILER ${MPICC})
    set(CMAKE_CXX_COMPILER ${MPICC})
endif()