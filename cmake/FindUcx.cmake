message(STATUS "Looking for UCX")
find_path(UCX_INCLUDE_DIR ucp/api/ucp.h
  HINTS
  /usr/local/include
  /usr/include
)

find_library(UCX_UCP_LIBRARIES ucp
  HINTS
  /usr/local/lib
  /usr/lib
  /usr/lib64
)

find_library(UCX_UCT_LIBRARIES uct
  HINTS
  /usr/local/lib
  /usr/lib
  /usr/lib64
)

find_library(UCX_UCS_LIBRARIES ucs
  HINTS
  /usr/local/lib
  /usr/lib
  /usr/lib64
)

if((NOT UCX_INCLUDE_DIR) OR (NOT UCX_UCP_LIBRARIES) OR (NOT UCX_UCT_LIBRARIES) OR (NOT UCX_UCS_LIBRARIES))
  message(FATAL_ERROR "Failed to find UCX. Please install UCX packages")
else()
  set(UCX_FOUND TRUE)
  set(UCX_LIBRARIES ${UCX_UCP_LIBRARIES} ${UCX_UCT_LIBRARIES} ${UCX_UCS_LIBRARIES})
  message(STATUS "Found UCX include at ${UCX_INCLUDE_DIR}")
  message(STATUS "Found UCX libraries: ${UCX_LIBRARIES}")
endif()

mark_as_advanced(UCX_FOUND UCX_INCLUDE_DIR UCX_UCP_LIBRARIES UCX_UCT_LIBRARIES UCX_UCS_LIBRARIES UCX_LIBRARIES)