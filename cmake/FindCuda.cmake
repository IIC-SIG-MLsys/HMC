# cmake/FindCUDA.cmake  equal to `find_package(CUDA QUIET)`

# default CUDA path
set(CUDA_PATH "" CACHE PATH "Path to CUDA installation (optional)")

# If the user does not specify the CUDA path, try to find it from environment variables or common locations
if(NOT CUDA_PATH)
    if(DEFINED ENV{CUDA_HOME})
        set(CUDA_PATH $ENV{CUDA_HOME})
    elseif(EXISTS /usr/local/cuda)
        set(CUDA_PATH /usr/local/cuda)
    endif()
endif()

# 查找CUDA的include目录
find_path(CUDA_INCLUDE_DIRS
    NAMES cuda_runtime.h
    HINTS ${CUDA_PATH}/include
)

# 查找CUDA的库目录
find_library(CUDA_CUDART_LIBRARY
    NAMES cudart
    HINTS ${CUDA_PATH}/lib64 ${CUDA_PATH}/lib
)

# 查找nvcc编译器
find_program(CUDA_NVCC_EXECUTABLE
    NAMES nvcc
    HINTS ${CUDA_PATH}/bin
)

# 检查是否找到所有必要的部分
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDA
    REQUIRED_VARS CUDA_INCLUDE_DIRS CUDA_CUDART_LIBRARY CUDA_NVCC_EXECUTABLE
)

# 如果找到了CUDA，设置一些额外的变量
if(CUDA_FOUND)
    # 设置CUDA版本
    execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} --version
        OUTPUT_VARIABLE CUDA_VERSION_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "[0-9]+\\.[0-9]+" CUDA_VERSION "${CUDA_VERSION_OUTPUT}")

    # 导出变量供其他地方使用
    set(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
    set(CUDA_COMPILE_OPTIONS "-Xcompiler -fPIC")
    mark_as_advanced(CUDA_INCLUDE_DIRS CUDA_LIBRARIES CUDA_NVCC_EXECUTABLE)
endif()