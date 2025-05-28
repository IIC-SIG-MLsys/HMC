/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "../resource_manager/gpu_interface.h"
#include "./mem_type.h"
#include <mem.h>

namespace hmc {
#ifdef ENABLE_CUDA
/*
 * nvidia gpu memory
 */
status_t CudaMemory::init() { return gpuInit(); }

status_t CudaMemory::free() { return status_t::SUCCESS; }

status_t CudaMemory::allocateBuffer(void **addr, size_t size) {
  cudaError_t ret;

  if (this->mem_type != MemoryType::NVIDIA_GPU) {
    return status_t::UNSUPPORT;
  }
  logInfo("Allocate memory using cudaMalloc.");
  ret = cudaMalloc(addr, size);
  if (ret != cudaSuccess) {
    logError("failed to allocate memory.");
    return status_t::ERROR;
  }

  // todo : dmabuf support : cuMemGetHandleForAddressRange()
  return status_t::SUCCESS;
}

status_t CudaMemory::allocatePeerableBuffer(void **addr, size_t size) {
  size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
  return this->allocateBuffer(addr, buf_size);
}

status_t CudaMemory::freeBuffer(void *addr) {
  cudaError_t ret;

  ret = cudaFree(addr);
  if (ret != cudaSuccess) {
    logError("failed to free memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t CudaMemory::copyHostToDevice(void *dest, const void *src,
                                      size_t size) {
  cudaError_t ret;

  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }

  ret = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
  if (ret != cudaSuccess) {
    logError("failed to copy memory from host to device");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t CudaMemory::copyDeviceToHost(void *dest, const void *src,
                                      size_t size) {
  cudaError_t ret;

  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyDeviceToHost Error.");
    return status_t::ERROR;
  }

  ret = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
  if (ret != cudaSuccess) {
    logError("failed to copy memory from device to host");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t CudaMemory::copyDeviceToDevice(void *dest, const void *src,
                                        size_t size) {
  cudaError_t ret;

  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyDeviceToDevice Error.");
    return status_t::ERROR;
  }

  ret = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
  if (ret != cudaSuccess) {
    logError("failed to copy memory from device to device");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

#else
status_t CudaMemory::init() { return status_t::UNSUPPORT; }
status_t CudaMemory::free() { return status_t::UNSUPPORT; }
status_t CudaMemory::allocateBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t CudaMemory::allocatePeerableBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t CudaMemory::freeBuffer(void *addr) { return status_t::UNSUPPORT; }

status_t CudaMemory::copyHostToDevice(void *dest, const void *src,
                                      size_t size) {
  return status_t::UNSUPPORT;
}
status_t CudaMemory::copyDeviceToHost(void *dest, const void *src,
                                      size_t size) {
  return status_t::UNSUPPORT;
}
status_t CudaMemory::copyDeviceToDevice(void *dest, const void *src,
                                        size_t size) {
  return status_t::UNSUPPORT;
}

#endif

} // namespace hmc
