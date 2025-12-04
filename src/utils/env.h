/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */
#ifndef HMC_ENV_CONFIG_H
#define HMC_ENV_CONFIG_H

static uint16_t env_u16_or_default(const char* name, uint16_t defv) {
  const char* s = std::getenv(name);
  if (!s || !*s) return defv;

  errno = 0;
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);

  if (errno != 0 || end == s || *end != '\0') return defv;
  if (v <= 0 || v > 65535) return defv;

  return static_cast<uint16_t>(v);
}

/* 
 * HMC_CTRL_PORT
 */

#endif