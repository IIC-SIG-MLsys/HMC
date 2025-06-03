/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "../resource_manager/gpu_interface.h"
#include "./mem_type.h"
#include <mem.h>

namespace hmc {

#ifdef ENABLE_NEUWARE
/*
 * nvidia gpu memory
 */
status_t NeuwareMemory::init() { 
  status_t sret = gpuInit();
  if (sret != status_t::SUCCESS) {
    logError("NeuwareMemory init Neuware err %s.", status_to_string(sret));
    return sret;
  }
  sret = gpuGetDevice(&mlu_dev, device_id);
  if (sret != status_t::SUCCESS) {
    logError("NeuwareMemory get device %d err %s.", device_id, status_to_string(sret));
    return sret;
  }
  sret = gpuCreateContext(&mlu_ctx, mlu_dev);
  if (sret != status_t::SUCCESS) {
    logError("NeuwareMemory creating context err %s.", status_to_string(sret));
    return sret;
  }
  return sret;
}

status_t NeuwareMemory::free() { 
  return gpuFreeContext(mlu_ctx);
}

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
    logError("failed to copy memory from host to device, err code: %d", ret);
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
} // namespace hmc
