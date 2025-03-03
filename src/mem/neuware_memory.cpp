/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "../rm/driver.h"
#include "mem_type.h"
#include <mem.h>

namespace hddt {

#ifdef ENABLE_NEUWARE
/*
 * nvidia gpu memory
 */
status_t NeuwareMemory::init() { return init_gpu_driver(this->device_id); }

status_t NeuwareMemory::free() { return free_gpu_driver(); }

status_t NeuwareMemory::allocateBuffer(void **addr, size_t size) {
  CNresult ret;

  if (this->mem_type != MemoryType::CAMBRICON_MLU) {
    return status_t::UNSUPPORT;
  }
  // logInfo("Allocate memory using cnMalloc.");
  ret = cnMalloc(&this->mlu_addr, size);
  if (ret != CN_SUCCESS) {
    logError("failed to allocate memory %d.", ret);
    return status_t::ERROR;
  }
  *addr = (void *)this->mlu_addr;
  // todo : dmabuf support : cuMemGetHandleForAddressRange()
  return status_t::SUCCESS;
}

status_t NeuwareMemory::allocatePeerableBuffer(void **addr, size_t size) {
  CNresult ret;
  cn_uint64_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);

  if (this->mem_type != MemoryType::CAMBRICON_MLU) {
    return status_t::UNSUPPORT;
  }
  // logInfo("Allocate memory using cnMallocPeerAble.");
  ret = cnMallocPeerAble(&this->mlu_addr, buf_size);
  if (ret != CN_SUCCESS) {
    logError("failed to allocate memory %d.", ret);
    return status_t::ERROR;
  }
  *addr = (void *)this->mlu_addr;
  // todo : dmabuf support : cuMemGetHandleForAddressRange()
  return status_t::SUCCESS;
}

status_t NeuwareMemory::freeBuffer(void *addr) {
  CNresult ret;
  CNaddr addr_mlu = (CNaddr)addr;
  ret = cnFree(addr_mlu);
  if (ret != CN_SUCCESS) {
    logError("failed to free memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t NeuwareMemory::copyHostToDevice(void *dest, const void *src,
                                         size_t size) {
  CNresult ret;

  if (dest == nullptr || src == nullptr) {
    logError("NeuwareMemory::copyHostToDevice Error.");
    return status_t::ERROR;
  }

  CNaddr dest_mlu = (CNaddr)dest;
  CNaddr src_mlu = (CNaddr)src;
  ret = cnMemcpy(dest_mlu, src_mlu, size);
  if (ret != CN_SUCCESS) {
    logError("failed to copy memory from host to memory");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t NeuwareMemory::copyDeviceToHost(void *dest, const void *src,
                                         size_t size) {
  CNresult ret;

  if (dest == nullptr || src == nullptr) {
    logError("NeuwareMemory::copyDeviceToHost Error.");
    return status_t::ERROR;
  }

  CNaddr dest_mlu = (CNaddr)dest;
  CNaddr src_mlu = (CNaddr)src;
  ret = cnMemcpy(dest_mlu, src_mlu, size);
  if (ret != CN_SUCCESS) {
    logError("failed to copy memory from device to host");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

status_t NeuwareMemory::copyDeviceToDevice(void *dest, const void *src,
                                           size_t size) {
  CNresult ret;

  if (dest == nullptr || src == nullptr) {
    logError("NeuwareMemory::copyDeviceToDevice Error.");
    return status_t::ERROR;
  }

  CNaddr dest_mlu = (CNaddr)dest;
  CNaddr src_mlu = (CNaddr)src;
  ret = cnMemcpy(dest_mlu, src_mlu, size);
  if (ret != CN_SUCCESS) {
    logError("failed to copy memory from device to device");
    return status_t::ERROR;
  }

  return status_t::SUCCESS;
}

#else
status_t NeuwareMemory::init() { return status_t::UNSUPPORT; }
status_t NeuwareMemory::free() { return status_t::UNSUPPORT; }
status_t NeuwareMemory::allocateBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::allocatePeerableBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::freeBuffer(void *addr) { return status_t::UNSUPPORT; }

status_t NeuwareMemory::copyHostToDevice(void *dest, const void *src,
                                         size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::copyDeviceToHost(void *dest, const void *src,
                                         size_t size) {
  return status_t::UNSUPPORT;
}
status_t NeuwareMemory::copyDeviceToDevice(void *dest, const void *src,
                                           size_t size) {
  return status_t::UNSUPPORT;
}

#endif
} // namespace hddt
