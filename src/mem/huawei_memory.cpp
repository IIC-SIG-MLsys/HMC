#include <acl/acl.h>
#include <iostream>
#include <mem.h>
#include <stdexcept>

namespace hddt {

HuaweiMemory::HuaweiMemory(int device_id, MemoryType mem_type)
    : MemoryBase(device_id, mem_type), is_initialized_(false),
      context_(nullptr), stream_(nullptr) {}

HuaweiMemory::~HuaweiMemory() { this->free(); }

status_t HuaweiMemory::init() {
  if (is_initialized_) {
    logInfo("ACL already initialized, skipping initialization.");
    return status_t::SUCCESS;
  }

  aclError ret;
  //  const char *aclConfigPath = "/home/sdu/acl.json";

  ret = aclInit(NULL);
  if (ret != ACL_SUCCESS) {
    logError("aclInit failed, error code: %d", ret);
    return status_t::ERROR;
  }

  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    logError("aclrtSetDevice failed, error code: %d", ret);
    aclFinalize();
    return status_t::ERROR;
  }

  ret = aclrtCreateContext(&context_, device_id);
  if (ret != ACL_SUCCESS) {
    logError("aclrtCreateContext failed, error code: %d", ret);
    aclrtResetDevice(device_id);
    aclFinalize();
    return status_t::ERROR;
  }

  ret = aclrtCreateStream(&stream_);
  if (ret != ACL_SUCCESS) {
    logError("aclrtCreateStream failed, error code: %d", ret);
    aclrtDestroyContext(context_);
    aclrtResetDevice(device_id);
    aclFinalize();
    return status_t::ERROR;
  }

  is_initialized_ = true;
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

status_t HuaweiMemory::allocate_buffer(void **addr, size_t size) {
  size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
  aclError ret = aclrtMalloc(addr, buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t HuaweiMemory::free_buffer(void *addr) {
  if (!addr) {
    return status_t::ERROR;
  }

  aclError ret = aclrtFree(addr);
  if (ret != ACL_SUCCESS) {
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t HuaweiMemory::copy_host_to_device(void *dest, const void *src,
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

status_t HuaweiMemory::copy_device_to_host(void *dest, const void *src,
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

status_t HuaweiMemory::copy_device_to_device(void *dest, const void *src,
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

} // namespace hddt
