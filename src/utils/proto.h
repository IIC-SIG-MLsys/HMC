/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */
#ifndef HDDT_PROTO_UTILS_H
#define HDDT_PROTO_UTILS_H

#include <iostream>
#include <string>
#include "protobuf/hddt.pb.h"

class ProtoUtils {
public:
    template <typename T>
    static std::string serializeToString(const T& message) {
        std::string output;
        if (!message.SerializeToString(&output)) {
            std::cerr << "Failed to serialize message." << std::endl;
            return "";
        }
        return output;
    }

    template <typename T>
    static bool deserializeFromString(T& message, const std::string& input) {
        return message.ParseFromString(input);
    }
};

#endif // HDDT_PROTO_UTILS_H
