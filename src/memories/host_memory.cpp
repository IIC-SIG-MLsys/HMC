/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "./mem_type.h"
#include <mem.h>

namespace hmc {
MemoryType memory_supported() {
#ifdef ENABLE_CUDA
  return MemoryType::NVIDIA_GPU;
#endif

#ifdef ENABLE_ROCM
  return MemoryType::AMD_GPU;
#endif

  return MemoryType::CPU;
}

bool memory_dmabuf_supported() {
#ifdef ENABLE_CUDA_DMABUF
  return true;
#else
  return false;
#endif
}

/*
 * host memory
 */
status_t HostMemory::init() { return status_t::SUCCESS; }

status_t HostMemory::free() { return status_t::SUCCESS; }

status_t HostMemory::allocateBuffer(void **addr, size_t size) {
  logInfo("Allocate memory using new malloc.");

  void *buffer = new (std::nothrow) char[size];
  if (buffer == nullptr) {
    logError("HostMemory::allocateBuffer Error.");
    return status_t::ERROR;
  } else {
    *addr = buffer;
    return status_t::SUCCESS;
  }
}

status_t HostMemory::allocatePeerableBuffer(void **addr, size_t size) {
  return this->allocateBuffer(addr, size);
}

status_t HostMemory::freeBuffer(void *addr) {
  if (addr == nullptr) {
    return status_t::ERROR;
  }

  delete[] static_cast<char *>(addr);

  addr = nullptr;

  return status_t::SUCCESS;
}

status_t HostMemory::copyHostToDevice(void *dest, const void *src,
                                      size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }

  try {
    memcpy(dest, src, size);
  } catch (const std::exception &e) {
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t HostMemory::copyDeviceToHost(void *dest, const void *src,
                                      size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyDeviceToHost Error.");
    return status_t::ERROR;
  }

  try {
    memcpy(dest, src, size);
  } catch (const std::exception &e) {
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t HostMemory::copyDeviceToDevice(void *dest, const void *src,
                                        size_t size) {
  if (dest == nullptr || src == nullptr) {
    logError("HostMemory::copyDeviceToDevice Error.");
    return status_t::ERROR;
  }

  try {
    memcpy(dest, src, size);
  } catch (const std::exception &e) {
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}
} // namespace hmc
