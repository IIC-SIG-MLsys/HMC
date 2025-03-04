/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_GPU_H
#define HDDT_GPU_H

#include <hddt.h>
#include <status.h>

#include "../utils/log.h"

namespace hddt {

status_t gpuInit();

template<typename T>
status_t gpuGetDevice(T *dev, int mlu_ordinal = 0);
status_t gpuGetPcieBusId(std::string *bus_id, int device_id); // char [16]
status_t gpuGetDeviceCount(int *device_count);
status_t gpuGetDeviceMemory(uint64_t *free_size, uint64_t *total_size);

status_t gpuSetDevice(int device_id); // Cuda and Rocm


template<typename T> // Cambricon
status_t gpuSetCtx(T &ctx); 
template<typename T>
status_t gpuCreateContext(T *ctx, int device_id);
template<typename T>
status_t gpuFreeContext(T ctx);

/* TODO: CUmodule */

}

#endif