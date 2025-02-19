/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef STATUS_H
#define STATUS_H

namespace hddt {
/* status */
enum class status_t { SUCCESS, ERROR, UNSUPPORT, INVALID_CONFIG, NOT_FOUND };

const char *status_to_string(status_t status);

}
#endif