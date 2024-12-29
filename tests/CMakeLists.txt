cmake_minimum_required(VERSION 3.21)
project(HDD_Test)

# 启用 C++ 语言
enable_language(CXX)

include_directories(include)
# 指定生成的可执行文件名称
set(BINARY ${CMAKE_PROJECT_NAME}_test)

# 查找测试源文件
file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/src/mem/*.cpp ${CMAKE_SOURCE_DIR}/src/* .cpp)

# 查找测试文件
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false ${CMAKE_SOURCE_DIR}/tests/hddt_memory_test.cpp)

# 创建可执行文件
add_executable(${BINARY} ${TEST_SOURCES} ${SOURCES})

# 添加测试
enable_testing()
add_test(NAME ${BINARY} COMMAND ${BINARY})


# 添加包含目录
target_include_directories(${BINARY} PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 链接 Google Test 库
find_package(GTest REQUIRED)
target_link_libraries(${BINARY} PRIVATE GTest::GTest GTest::Main)

target_link_libraries(${BINARY} PUBLIC glog)
target_link_libraries(${BINARY} PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})

# 链接 Huawei Ascend 库
if(HUAWEI_FOUND)
    target_link_libraries(${BINARY} PUBLIC ${HUAWEI_LIBRARIES})
endif()