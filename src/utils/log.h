/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */
#ifndef HDDT_LOG_H
#define HDDT_LOG_H

#include <glog/logging.h>
#include <hddt.h>

/* log */
#define logError(fmt, ...)                                                     \
  do {                                                                         \
    char log_buffer[1024];                                                     \
    int len = snprintf(log_buffer, sizeof(log_buffer), fmt, ##__VA_ARGS__);    \
    if (len >= 0) {                                                            \
      LOG(ERROR) << log_buffer;                                                \
    }                                                                          \
  } while (0)
#define logDebug(fmt, ...)                                                     \
  do {                                                                         \
    char log_buffer[1024];                                                     \
    int len = snprintf(log_buffer, sizeof(log_buffer), fmt, ##__VA_ARGS__);    \
    if (len >= 0) {                                                            \
      LOG(WARNING) << log_buffer;                                              \
    }                                                                          \
  } while (0)
#define logInfo(fmt, ...)                                                      \
  do {                                                                         \
    char log_buffer[1024];                                                     \
    int len = snprintf(log_buffer, sizeof(log_buffer), fmt, ##__VA_ARGS__);    \
    if (len >= 0) {                                                            \
      LOG(INFO) << log_buffer;                                                 \
    }                                                                          \
  } while (0)

#endif