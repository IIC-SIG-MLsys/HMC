# CNRT

set(NEUWARE_HOME "" CACHE PATH "Path to NEUWARE installation (optional)")

if(NOT NEUWARE_HOME)
    if(DEFINED ENV{NEUWARE_HOME})
        set(NEUWARE_HOME $ENV{NEUWARE_HOME})
    elseif(EXISTS /usr/local/neuware)
        set(NEUWARE_HOME /usr/local/neuware)
    endif()
endif()
message(STATUS "NEUWARE_HOME=${NEUWARE_HOME}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
  )

find_path(NEUWARE_INCLUDE_DIRS
    NAMES cnrt.h
    HINTS ${NEUWARE_HOME}/include
)
find_program(NEUWARE_CNCC_EXECUTABLE
    NAMES cncc
    HINTS ${NEUWARE_HOME}/bin
)

set(NEUWARE_LIBRARIES "")
find_library(NEUWARE_CNRT_LIBRARY
    NAMES cnrt
    HINTS ${NEUWARE_HOME}/lib64 ${NEUWARE_HOME}/lib
)
if(NEUWARE_CNRT_LIBRARY)
    list(APPEND NEUWARE_LIBRARIES ${NEUWARE_CNRT_LIBRARY})
endif()
find_library(NEUWARE_CNNL_LIBRARY
    NAMES cnnl
    HINTS ${NEUWARE_HOME}/lib64 ${NEUWARE_HOME}/lib
)
if(NEUWARE_CNNL_LIBRARY)
    list(APPEND NEUWARE_LIBRARIES ${NEUWARE_CNNL_LIBRARY})
endif()
find_library(NEUWARE_CNDRV_LIBRARY
    NAMES cndrv
    HINTS ${NEUWARE_HOME}/lib64 ${NEUWARE_HOME}/lib
)
if(NEUWARE_CNDRV_LIBRARY)
    list(APPEND NEUWARE_LIBRARIES ${NEUWARE_CNDRV_LIBRARY})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NEUWARE
    REQUIRED_VARS NEUWARE_INCLUDE_DIRS NEUWARE_LIBRARIES NEUWARE_CNCC_EXECUTABLE
)

if(NEUWARE_FOUND)
    execute_process(COMMAND ${NEUWARE_CNCC_EXECUTABLE} --version
            OUTPUT_VARIABLE NEUWARE_VERSION_OUTPUT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    string(REGEX MATCH "[0-9]+\\.[0-9]+" NEUWARE_VERSION "${NEUWARE_VERSION_OUTPUT}")

    set(TARGET_CPU_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -Wno-reorder -fPIC -pthread")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS} -g3 -O0")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${CMAKE_C_FLAGS} -O3")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wno-reorder -fPIC -std=c++11 -pthread")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS} -g3 -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")

    set(CMAKE_EXPORT_COMPILE_COMMANDS on)

    find_package(BANG)
    if(NOT BANG_FOUND)
    message(FATAL_ERROR "Have not found BANG.")
    else()
    message(STATUS "BANG have been found.")
    endif()

    set(BANG_CNCC_FLAGS "-fPIC -std=c++11 -pthread -Wall -Werror -Wno-reorder --target=${TARGET_CPU_ARCH}-linux-gnu")
    if("${BANG_ARCH}-${BANG_MLU_ARCH}" STREQUAL "-")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_220")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_270")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_290")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_372")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_322")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_592")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=tp_520")
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} --bang-mlu-arch=tp_521")
    else()
        foreach(arch ${BANG_ARCH})
            list(APPEND BANG_COMPILE_ARCH --bang-arch=${arch})
        endforeach()

        foreach(arch ${BANG_MLU_ARCH})
            list(APPEND BANG_COMPILE_ARCH --bang-mlu-arch=${arch})
        endforeach()
        
        set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} ${BANG_COMPILE_ARCH}")
    endif()
    set(BANG_EXIST ON)

    mark_as_advanced(NEUWARE_INCLUDE_DIRS NEUWARE_LIBRARIES NEUWARE_CNCC_EXECUTABLE)
endif()
