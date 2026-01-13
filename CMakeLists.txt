cmake_minimum_required(VERSION 3.21)
project(HMC)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_STATIC_LIB "Build static library" OFF)
option(BUILD_PYTHON_MOD "Build python module" OFF)

if(BUILD_PYTHON_MOD)
  set(BUILD_STATIC_LIB ON)
endif()

set(UCX_ROOT "/usr/local/ucx" CACHE PATH "UCX install prefix")
set(UCX_INCLUDE_DIR "${UCX_ROOT}/include")
set(UCX_LIB_DIR "${UCX_ROOT}/lib")

if(EXISTS "${UCX_INCLUDE_DIR}/ucp/api/ucp.h")
  message(STATUS "UCX include dir detected: ${UCX_INCLUDE_DIR}")
else()
  message(WARNING "UCX headers not found at ${UCX_INCLUDE_DIR}. Update UCX_ROOT if needed.")
endif()

find_library(UCX_UCP_LIB NAMES ucp HINTS "${UCX_LIB_DIR}" "${UCX_ROOT}/lib64")
find_library(UCX_UCS_LIB NAMES ucs HINTS "${UCX_LIB_DIR}" "${UCX_ROOT}/lib64")
find_library(UCX_UCT_LIB NAMES uct HINTS "${UCX_LIB_DIR}" "${UCX_ROOT}/lib64")

set(UCX_LIBRARIES "")
if(UCX_UCP_LIB AND UCX_UCS_LIB AND UCX_UCT_LIB)
  list(APPEND UCX_LIBRARIES ${UCX_UCP_LIB} ${UCX_UCS_LIB} ${UCX_UCT_LIB})
  message(STATUS "UCX libs: ${UCX_LIBRARIES}")
else()
  message(WARNING "UCX libraries not fully found (ucp/ucs/uct). Python module may fail to load.")
endif()

include(FindRocm)
include(FindCuda)
include(FindNeuware)
include(FindMUSA)

include_directories(include)
include_directories(src)

include(FindRdma)
if(NOT RDMA_FOUND)
  message(FATAL_ERROR "RDMA is required but was not found.")
endif()

file(GLOB_RECURSE SRC_RECURSE "src/*.cpp")
if(NOT SRC_RECURSE)
  message(FATAL_ERROR "No source files found in src/")
endif()

if(ROCM_FOUND)
  add_definitions(-DENABLE_ROCM)
  enable_language(HIP)
  add_definitions(-D__HIP_PLATFORM_AMD__)
elseif(CUDA_FOUND)
  add_definitions(-DENABLE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  enable_language(CUDA)
elseif(NEUWARE_FOUND)
  add_definitions(-DENABLE_NEUWARE)
  include_directories(${NEUWARE_INCLUDE_DIRS})
elseif(MUSA_FOUND)
  add_definitions(-DENABLE_MUSA)
  include_directories(${MUSA_INCLUDE_DIRS})
else()
  message(WARNING "No GPU backend detected. Build CPU-only.")
endif()

function(hmc_link_common target_name)
  target_include_directories(${target_name} PUBLIC ${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR})
  target_link_libraries(${target_name} PUBLIC ${VERBS_LIBRARIES} ${RDMACM_LIBRARIES})

  if(EXISTS "${UCX_INCLUDE_DIR}/ucp/api/ucp.h")
    target_include_directories(${target_name} PUBLIC ${UCX_INCLUDE_DIR})
  endif()
  if(UCX_LIBRARIES)
    target_link_libraries(${target_name} PUBLIC ${UCX_LIBRARIES})
    set_target_properties(${target_name} PROPERTIES
      BUILD_RPATH "${UCX_LIB_DIR}"
      INSTALL_RPATH "${UCX_LIB_DIR}"
    )
  endif()

  if(CUDA_FOUND)
    target_link_libraries(${target_name} PUBLIC ${CUDA_LIBRARIES})
  endif()

  if(MUSA_FOUND)
    target_link_libraries(${target_name} PUBLIC ${MUSA_LIBRARIES})
  endif()

  if(NEUWARE_FOUND)
    target_link_libraries(${target_name} PUBLIC ${NEUWARE_LIBRARIES})
  endif()

  if(ROCM_FOUND)
    find_library(ROCM_AMDHIP64
      NAMES amdhip64
      HINTS "${ROCM_PATH}/lib" "${ROCM_PATH}/lib64" "/opt/rocm/lib" "/opt/rocm/lib64"
    )
    if(ROCM_AMDHIP64)
      target_link_libraries(${target_name} PUBLIC ${ROCM_AMDHIP64})
    endif()
  endif()
endfunction()

if(BUILD_STATIC_LIB)
  add_library(hmc_static_lib STATIC ${SRC_RECURSE})
  target_include_directories(hmc_static_lib PUBLIC include)
  hmc_link_common(hmc_static_lib)
  install(TARGETS hmc_static_lib DESTINATION lib)
endif()

add_library(hmc_shared_lib SHARED ${SRC_RECURSE})
target_include_directories(hmc_shared_lib PUBLIC include)
hmc_link_common(hmc_shared_lib)
install(TARGETS hmc_shared_lib DESTINATION lib)

if(BUILD_PYTHON_MOD)
  execute_process(
    COMMAND git submodule update --init
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT
  )
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}")
  endif()

  add_subdirectory(extern/pybind11)
  file(GLOB_RECURSE ALL_PYTHON_CPP "python/*.cpp")
  list(FILTER ALL_PYTHON_CPP EXCLUDE REGEX "vllm-het/") # exclude "vllm-het"
  if(NOT ALL_PYTHON_CPP)
    message(FATAL_ERROR "No python binding sources found in python/ (excluding vllm-het)")
  endif()
  set(PYTHON_SRC_RECURSE ${ALL_PYTHON_CPP})

  pybind11_add_module(hmc ${PYTHON_SRC_RECURSE})
  target_link_libraries(hmc PRIVATE hmc_static_lib)

  if(UCX_LIBRARIES)
    set_target_properties(hmc PROPERTIES
      BUILD_RPATH "${UCX_LIB_DIR}"
      INSTALL_RPATH "${UCX_LIB_DIR}"
    )
  endif()

  install(TARGETS hmc LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/pkg)
endif()

install(DIRECTORY include/ DESTINATION include)

add_subdirectory(apps/uhm_app)
