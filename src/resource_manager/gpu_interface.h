/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_GPU_H
#define HMC_GPU_H

#include <hmc.h>
#include <status.h>

#include "../utils/log.h"

namespace hmc {

status_t gpuInit();
status_t gpuGetPcieBusId(std::string *bus_id, int device_id); // char [16]
status_t gpuGetDeviceCount(int *device_count);
status_t gpuGetDeviceMemory(uint64_t *free_size, uint64_t *total_size);
status_t gpuSetDevice(int device_id); // Cuda and Rocm

/* 模板函数的完整定义放在头文件,编译每个使用该模板函数的源文件时才会生成相应的实例代码
 */
template <typename T>
inline status_t gpuGetDevice(T *dev, int mlu_ordinal = 0) {
#ifdef ENABLE_CUDA
  int device; // 使用 Runtime API 获取当前设备，注意 T 应该为 int 类型
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) {
    logError("cudaGetDevice failed with error code %d\n", err);
    return status_t::ERROR;
  }
  *dev = device;
  return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
  int device;
  hipError_t res = hipGetDevice(&device);
  if (res != hipSuccess) {
    logError("hipGetDevice failed with error code %d\n", res);
    return status_t::ERROR;
  }
  *dev = device;
  return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
  CNdev device;
  CNresult res = cnDeviceGet(&device, mlu_ordinal);
  if (res != CN_SUCCESS) {
    logError("cnDeviceGet failed with error code %d", res);
    return status_t::ERROR;
  }
  *dev = device;
  return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
  // TODO
  return status_t::SUCCESS;
#elif defined(ENABLE_MUSA)
  int device; // 使用 Runtime API 获取当前设备，注意 T 应该为 int 类型
  musaError_t err = musaGetDevice(&device);
  if (err != musaSuccess) {
    logError("musaGetDevice failed with error code %d\n", err);
    return status_t::ERROR;
  }
  *dev = device;
  return status_t::SUCCESS;
#else
  return status_t::UNSUPPORT;
#endif
};

template <typename T> // Cambricon
inline status_t gpuSetCtx(T &ctx) {
#ifdef ENABLE_CUDA
  // CUDA Runtime 不支持显式上下文管理，这里给出空实现
  return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
  CNresult ret = cnCtxSetCurrent(ctx);
  if (ret != CN_SUCCESS) {
    logError("failed to set cnCtx %d.", ret);
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
#else
  return status_t::UNSUPPORT;
#endif
};

// only for neuware: CNContext. CNDev
template <typename T, typename D>
inline status_t gpuCreateContext(T *ctx, D dev) {
#ifdef ENABLE_NEUWARE
  CNresult res = cnCtxCreate(ctx, 0, dev);
  if (res != CN_SUCCESS) {
    logError("cnCtxCreate failed with error code %d", res);
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
#endif
  return status_t::UNSUPPORT;
};

template <typename T> inline status_t gpuFreeContext(T ctx) {
#ifdef ENABLE_CUDA
  // CUDA Runtime 的上下文由系统自动管理，不需要手动释放
  return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
  return status_t::UNSUPPORT;
#elif defined(ENABLE_NEUWARE)
  CNresult res = cnCtxDestroy(ctx);
  if (res != CN_SUCCESS) {
    logError("cnCtxDestroy failed with error code %d", res);
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
#elif defined(ENABLE_MUSA)
  // TODO
  return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
  // TODO
  return status_t::UNSUPPORT;
#else
  return status_t::UNSUPPORT;
#endif
};

} // namespace hmc

#endif