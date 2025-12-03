/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "../resource_manager/gpu_interface.h"
#include "./mem_type.h"
#include <mem.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cstring>
#endif

namespace hmc {

#ifdef ENABLE_CUDA

status_t CudaMemory::init() { return gpuInit(); }

status_t CudaMemory::free() { return status_t::SUCCESS; }

status_t CudaMemory::allocateBuffer(void **addr, size_t size) {
  if (this->mem_type != MemoryType::NVIDIA_GPU) {
    return status_t::UNSUPPORT;
  }

  logInfo("Allocate memory using cudaMalloc.");
  cudaError_t ret = cudaMalloc(addr, size);
  if (ret != cudaSuccess) {
    logError("failed to allocate memory: %s", cudaGetErrorString(ret));
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t CudaMemory::allocatePeerableBuffer(void **addr, size_t size) {
  size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
  return this->allocateBuffer(addr, buf_size);
}

static inline bool is_exit_stage_cuda_error(cudaError_t e) {
  return (e == cudaErrorCudartUnloading) || (e == cudaErrorContextIsDestroyed);
}

status_t CudaMemory::freeBuffer(void *addr) {
  if (addr == nullptr) return status_t::SUCCESS;

  int prev_dev = -1;
  (void)cudaGetDevice(&prev_dev);

#if CUDART_VERSION >= 10000
  cudaPointerAttributes attr;
  std::memset(&attr, 0, sizeof(attr));
  cudaError_t aerr = cudaPointerGetAttributes(&attr, addr);
  if (aerr == cudaSuccess) {
#if CUDART_VERSION >= 11000
    if (attr.type == cudaMemoryTypeDevice) {
      (void)cudaSetDevice(attr.device);
    }
#else
    if (attr.memoryType == cudaMemoryTypeDevice) {
      (void)cudaSetDevice(attr.device);
    }
#endif
  } else {
    // clear sticky error if any
    (void)cudaGetLastError();
  }
#endif

  cudaError_t ret = cudaFree(addr);

  if (prev_dev >= 0) (void)cudaSetDevice(prev_dev);

  if (ret == cudaSuccess) return status_t::SUCCESS;

  if (is_exit_stage_cuda_error(ret)) {
    return status_t::SUCCESS;
  }

  logError("failed to free memory: %s", cudaGetErrorString(ret));
  return status_t::ERROR;
}

status_t CudaMemory::copyHostToDevice(void *dest, const void *src, size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("CudaMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }

  cudaError_t ret = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
    if (is_exit_stage_cuda_error(ret)) return status_t::SUCCESS;
    logError("failed to copy memory from host to device: %s", cudaGetErrorString(ret));
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t CudaMemory::copyDeviceToHost(void *dest, const void *src, size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("CudaMemory::copyDeviceToHost Error.");
    return status_t::ERROR;
  }

  cudaError_t ret = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
  if (ret != cudaSuccess) {
    if (is_exit_stage_cuda_error(ret)) return status_t::SUCCESS;
    logError("failed to copy memory from device to host: %s", cudaGetErrorString(ret));
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t CudaMemory::copyDeviceToDevice(void *dest, const void *src, size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("CudaMemory::copyDeviceToDevice Error.");
    return status_t::ERROR;
  }

  cudaError_t ret = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
  if (ret != cudaSuccess) {
    if (is_exit_stage_cuda_error(ret)) return status_t::SUCCESS;
    logError("failed to copy memory from device to device: %s", cudaGetErrorString(ret));
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

#else

status_t CudaMemory::init() { return status_t::UNSUPPORT; }
status_t CudaMemory::free() { return status_t::UNSUPPORT; }

status_t CudaMemory::allocateBuffer(void **addr, size_t size) {
  (void)addr; (void)size;
  return status_t::UNSUPPORT;
}

status_t CudaMemory::allocatePeerableBuffer(void **addr, size_t size) {
  (void)addr; (void)size;
  return status_t::UNSUPPORT;
}

status_t CudaMemory::freeBuffer(void *addr) {
  (void)addr;
  return status_t::UNSUPPORT;
}

status_t CudaMemory::copyHostToDevice(void *dest, const void *src, size_t size) {
  (void)dest; (void)src; (void)size;
  return status_t::UNSUPPORT;
}

status_t CudaMemory::copyDeviceToHost(void *dest, const void *src, size_t size) {
  (void)dest; (void)src; (void)size;
  return status_t::UNSUPPORT;
}

status_t CudaMemory::copyDeviceToDevice(void *dest, const void *src, size_t size) {
  (void)dest; (void)src; (void)size;
  return status_t::UNSUPPORT;
}

#endif

} // namespace hmc
