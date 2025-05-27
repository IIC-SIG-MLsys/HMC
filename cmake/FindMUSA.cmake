# ======================================================================
# FindMUSA.cmake
# 查找 MUSA SDK 并设置下列变量：
#   MUSA_FOUND               - 如果找到 MUSA SDK 则为 TRUE
#   MUSA_INCLUDE_DIRS        - MUSA SDK 的头文件目录
#   MUSA_LIBRARIES           - MUSA SDK 的库文件
#   MUSA_VERSION_EXECUTABLE  - 用于查询版本的可执行程序
# ======================================================================

# 若未设置 MUSA_HOME，则尝试设置为默认路径
if(NOT MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
endif()

# 查找 MUSA 的头文件目录
find_path(MUSA_INCLUDE_DIRS
    NAMES musa.h
    HINTS ${MUSA_HOME}/include
)

# 查找版本查询工具（例如 musa_version_query）
find_program(MUSA_VERSION_EXECUTABLE
    NAMES musa_version_query
    HINTS ${MUSA_HOME}/bin
)

# 初始化 MUSA 库列表，并查找运行时库（正确库名为 musa）
set(MUSA_LIBRARIES "")
find_library(MUSA_RT_LIBRARY
    NAMES musart          # 修正后的库名
    HINTS ${MUSA_HOME}/lib
)

if(MUSA_RT_LIBRARY)
    list(APPEND MUSA_LIBRARIES ${MUSA_RT_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUSA
    REQUIRED_VARS MUSA_INCLUDE_DIRS MUSA_LIBRARIES MUSA_VERSION_EXECUTABLE
)

message(STATUS "MUSA_HOME: ${MUSA_HOME}")
message(STATUS "MUSA_INCLUDE_DIRS: ${MUSA_INCLUDE_DIRS}")
message(STATUS "MUSA_LIBRARIES: ${MUSA_LIBRARIES}")
message(STATUS "MUSA_VERSION_EXECUTABLE: ${MUSA_VERSION_EXECUTABLE}")

if(MUSA_FOUND)
    execute_process(
        COMMAND ${MUSA_VERSION_EXECUTABLE} --version
        OUTPUT_VARIABLE MUSA_VERSION_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "[0-9]+\\.[0-9]+" MUSA_VERSION "${MUSA_VERSION_OUTPUT}")
    message(STATUS "MUSA SDK Version: ${MUSA_VERSION}")
endif()