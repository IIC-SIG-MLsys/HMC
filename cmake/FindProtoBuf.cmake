# 查找protobuf包
find_package(Protobuf REQUIRED)

# 打印出protobuf的版本信息进行验证
if(Protobuf_FOUND)
    message(STATUS "Found protobuf: ${Protobuf_VERSION}")
endif()

# target_link_libraries(MyExecutable PRIVATE ${PROTOBUF_LIBRARIES})