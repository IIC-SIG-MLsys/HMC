/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */
#include <status.h>

namespace hddt {

const char *status_to_string(status_t status) {
  switch (status) {
  case status_t::SUCCESS:
    return "Succeeded";
  case status_t::ERROR:
    return "Error";
  case status_t::UNSUPPORT:
    return "Unsupported";
  case status_t::INVALID_CONFIG:
    return "Invalid Config";
  case status_t::NOT_FOUND:
    return "Not Found";
  default:
    return "Unknown Status";
  }
}

} // namespace hddt
