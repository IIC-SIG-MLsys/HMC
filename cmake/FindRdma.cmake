
message(STATUS "compile with rdma")
find_path(VERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(VERBS_LIBRARIES ibverbs)
find_path(RDMACM_INCLUDE_DIR rdma/rdma_cma.h)
find_library(RDMACM_LIBRARIES rdmacm)
# find_package(rdmacm REQUIRED)

message(STATUS "Found libverbs at ${VERBS_LIBRARIES}")
message(STATUS "Found rdmacm at ${RDMACM_LIBRARIES}")

if((NOT VERBS_LIBRARIES) OR (NOT RDMACM_LIBRARIES))
    message(FATAL_ERROR "Fail to find ibverbs or rdmacm")
else()
    set(RDMA_FOUND TRUE)
endif()
# message(STATUS "${VERBS_INCLUDE_DIR} ${RDMACM_INCLUDE_DIR}")
# message(STATUS "${VERBS_LIBRARIES} ${RDMACM_LIBRARIES}")
mark_as_advanced(RDMA_FOUND VERBS_INCLUDE_DIR RDMACM_INCLUDE_DIR VERBS_LIBRARIES RDMACM_LIBRARIES)