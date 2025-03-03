/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "../rm/driver.h"
#include "mem_type.h"
#include <mem.h>

namespace hddt {
#ifdef ENABLE_ROCM
// todo : ref
// https://github1s.com/linux-rdma/perftest/blob/master/src/rocm_memory.c
/*
 * amd gpu memory
 */
status_t RocmMemory::init() { return init_gpu_driver(this->device_id); }

status_t RocmMemory::free() { return free_gpu_driver(); }

status_t RocmMemory::allocateBuffer(void **addr, size_t size) {
  hipError_t ret;

  if (this->mem_type != MemoryType::AMD_GPU) {
    return status_t::UNSUPPORT;
  }

  logInfo("Allocate memory using hipMalloc.");
  ret = hipMalloc(addr, size);
  if (ret != hipSuccess) {
    logError("failed to allocate memory");
    return status_t::ERROR;
  }
  // todo : dmabuf support :pfn_hsa_amd_portable_export_dmabuf
  return status_t::SUCCESS;
}

status_t RocmMemory::allocatePeerableBuffer(void **addr, size_t size) {
  size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
  return this->allocateBuffer(addr, buf_size);
}

status_t RocmMemory::freeBuffer(void *addr) {
  hipError_t ret;
  ret = hipFree(addr);
  if (ret != hipSuccess) {
    logError("failed to free memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t RocmMemory::copyHostToDevice(void *dest, const void *src,
                                      size_t size) {
  hipError_t ret;

  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }
  ret = hipMemcpy(dest, src, size, hipMemcpyDeviceToHost);
  if (ret != hipSuccess) {
    logError("failed to copy memory from host to memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t RocmMemory::copyDeviceToHost(void *dest, const void *src,
                                      size_t size) {
  hipError_t ret;

  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }
  ret = hipMemcpy(dest, src, size, hipMemcpyHostToDevice);
  if (ret != hipSuccess) {
    logError("failed to copy memory from host to memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t RocmMemory::copyDeviceToDevice(void *dest, const void *src,
                                        size_t size) {
  hipError_t ret;

  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }
  ret = hipMemcpy(dest, src, size, hipMemcpyDeviceToDevice);
  if (ret != hipSuccess) {
    logError("failed to copy memory from host to memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

#else
status_t RocmMemory::init() { return status_t::UNSUPPORT; }
status_t RocmMemory::free() { return status_t::UNSUPPORT; }
status_t RocmMemory::allocateBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t RocmMemory::allocatePeerableBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t RocmMemory::freeBuffer(void *addr) { return status_t::UNSUPPORT; }

status_t RocmMemory::copyHostToDevice(void *dest, const void *src,
                                      size_t size) {
  return status_t::UNSUPPORT;
}
status_t RocmMemory::copyDeviceToHost(void *dest, const void *src,
                                      size_t size) {
  return status_t::UNSUPPORT;
}
status_t RocmMemory::copyDeviceToDevice(void *dest, const void *src,
                                        size_t size) {
  return status_t::UNSUPPORT;
}
#endif

} // namespace hddt
