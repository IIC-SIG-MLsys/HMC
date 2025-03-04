/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "./mem_type.h"

#include <iostream>
#include <mem.h>
#include <stdexcept>

namespace hddt {
#ifdef ENABLE_HUAWEI

status_t HuaweiMemory::init() {

  logInfo("HuaweiMemory initialization completed successfully.");
  return status_t::SUCCESS;
}

status_t HuaweiMemory::free() {
  if (!is_initialized_) {
    logDebug("ACL is not initialized or already cleaned up. Skipping redundant "
             "free().");
    return status_t::SUCCESS;
  }

  if (stream_) {
    aclrtDestroyStream(stream_);
    stream_ = nullptr;
    logInfo("Stream destroyed.");
  }

  if (context_) {
    aclrtDestroyContext(context_);
    context_ = nullptr;
    logInfo("Context destroyed.");
  }

  aclrtResetDevice(device_id);
  logInfo("Device reset completed.");

  aclFinalize();
  logInfo("ACL finalized successfully.");

  is_initialized_ = false;
  logInfo("ACL cleanup completed successfully.");
  return status_t::SUCCESS;
}

status_t HuaweiMemory::allocateBuffer(void **addr, size_t size) {
  size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
  aclError ret = aclrtMalloc(addr, buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t HuaweiMemory::freeBuffer(void *addr) {
  if (!addr) {
    return status_t::ERROR;
  }

  aclError ret = aclrtFree(addr);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t HuaweiMemory::copyHostToDevice(void *dest, const void *src,
                                        size_t size) {
  if (!dest || !src) {
    return status_t::ERROR;
  }

  aclError ret = aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t HuaweiMemory::copyDeviceToHost(void *dest, const void *src,
                                        size_t size) {
  if (!dest || !src) {
    return status_t::ERROR;
  }

  aclError ret = aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t HuaweiMemory::copyDeviceToDevice(void *dest, const void *src,
                                          size_t size) {
  if (!dest || !src) {
    return status_t::ERROR;
  }

  aclError ret =
      aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

#else
status_t HuaweiMemory::init() { return status_t::UNSUPPORT; }
status_t HuaweiMemory::free() { return status_t::UNSUPPORT; }
status_t HuaweiMemory::allocateBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t HuaweiMemory::allocatePeerableBuffer(void **addr, size_t size) {
  return status_t::UNSUPPORT;
}
status_t HuaweiMemory::freeBuffer(void *addr) { return status_t::UNSUPPORT; }

status_t HuaweiMemory::copyHostToDevice(void *dest, const void *src,
                                        size_t size) {
  return status_t::UNSUPPORT;
}
status_t HuaweiMemory::copyDeviceToHost(void *dest, const void *src,
                                        size_t size) {
  return status_t::UNSUPPORT;
}
status_t HuaweiMemory::copyDeviceToDevice(void *dest, const void *src,
                                          size_t size) {
  return status_t::UNSUPPORT;
}

#endif

} // namespace hddt
