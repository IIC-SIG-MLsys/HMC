#include <mem.h>

namespace hddt {
#ifdef ENABLE_HUAWEI
/*
 * 华为 Ascend NPU 显存管理
 */
status_t HuaweiMemory::init() { 
    logInfo("Initializing Huawei Ascend NPU driver.");
    return init_gpu_driver(this->device_id); 
}

status_t HuaweiMemory::free() { 
    logInfo("Releasing Huawei Ascend NPU resources.");
    return free_gpu_driver(); 
}

status_t HuaweiMemory::allocate_buffer(void **addr, size_t size) {
    size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
    logInfo("Allocating memory on Huawei Ascend NPU.");
    
    // 使用 CANN 的 malloc 接口
    aclError ret = aclrtMalloc(addr, buf_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        logError("Failed to allocate memory on Huawei Ascend NPU.");
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
}

status_t HuaweiMemory::free_buffer(void *addr) {
    logInfo("Freeing memory on Huawei Ascend NPU.");
    aclError ret = aclrtFree(addr);
    if (ret != ACL_SUCCESS) {
        logError("Failed to free memory on Huawei Ascend NPU.");
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
}

status_t HuaweiMemory::copy_host_to_device(void *dest, const void *src, size_t size) {
    logInfo("Copying data from host to Huawei Ascend NPU.");
    aclError ret = aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        logError("Failed to copy data from host to Huawei Ascend NPU.");
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
}

status_t HuaweiMemory::copy_device_to_host(void *dest, const void *src, size_t size) {
    logInfo("Copying data from Huawei Ascend NPU to host.");
    aclError ret = aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        logError("Failed to copy data from Huawei Ascend NPU to host.");
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
}

status_t HuaweiMemory::copy_device_to_device(void *dest, const void *src, size_t size) {
    logInfo("Copying data between Huawei Ascend NPUs.");
    aclError ret = aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        logError("Failed to copy data between Huawei Ascend NPUs.");
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
}

#else
status_t HuaweiMemory::init() { return status_t::UNSUPPORT; }
status_t HuaweiMemory::free() { return status_t::UNSUPPORT; }
status_t HuaweiMemory::allocate_buffer(void **addr, size_t size) { return status_t::UNSUPPORT; }
status_t HuaweiMemory::free_buffer(void *addr) { return status_t::UNSUPPORT; }
status_t HuaweiMemory::copy_host_to_device(void *dest, const void *src, size_t size) { return status_t::UNSUPPORT; }
status_t HuaweiMemory::copy_device_to_host(void *dest, const void *src, size_t size) { return status_t::UNSUPPORT; }
status_t HuaweiMemory::copy_device_to_device(void *dest, const void *src, size_t size) { return status_t::UNSUPPORT; }
#endif
} // namespace hddt
