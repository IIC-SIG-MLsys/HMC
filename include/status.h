/**
 * @file status.h
 * @brief Common status codes and error definitions for the HMC framework.
 *
 * This header defines the unified `status_t` enumeration used throughout
 * the HMC (Heterogeneous Memories Communication) system to represent
 * operation results, including success, error, timeout, and unsupported cases.
 *
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */

#ifndef STATUS_H
#define STATUS_H

namespace hmc {

/**
 * @enum status_t
 * @brief Standardized return status codes for all HMC operations.
 */
enum class status_t {
  SUCCESS,        ///< Operation completed successfully
  ERROR,          ///< Generic or unspecified error
  UNSUPPORT,      ///< Operation not supported on current backend
  INVALID_CONFIG, ///< Invalid configuration or initialization state
  TIMEOUT,        ///< Operation timed out
  NOT_FOUND       ///< Requested resource not found
};

/**
 * @brief Converts a status code into a human-readable string.
 * @param status The status code to convert.
 * @return A const char* representing the description of the given status.
 */
const char *status_to_string(status_t status);

} // namespace hmc

#endif // STATUS_H
