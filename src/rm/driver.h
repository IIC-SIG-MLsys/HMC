/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_DRIVER_H
#define HDDT_DRIVER_H

#include "../utils/log.h"
#include <hddt.h>

namespace hddt {
/*
gpu resource manager
*/
status_t init_gpu_driver(int device_id);
status_t free_gpu_driver();

/*
rdma resource manager
*/
bool support_rdma();

// TODO: resource manager

} // namespace hddt

#endif