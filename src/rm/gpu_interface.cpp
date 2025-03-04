/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */

#include "gpu_interface.h"
#include <string>

namespace hddt {

status_t gpuInit() {
#ifdef ENABLE_CUDA
    cudaError_t err = cudaFree(0); // 使用 CUDA Runtime API，不需要显式调用 cuInit，使用 cudaFree(0) 来触发初始化
    if (err != cudaSuccess) {
        logError("cudaFree(0) failed with error code %d\n", err);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    hipError_t res = hipInit(0);
    if (res != hipSuccess) {
        logError("hipInit failed with error code %d\n", res);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
    CNresult res = cnInit(0);
    if (res != CN_SUCCESS) {
        logError("cnInit failed with error code %d", res);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    // TODO
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
}

// 获取设备的 PCIe 总线 ID
status_t gpuGetPcieBusId(std::string *bus_id, int device_id) {
#ifdef ENABLE_CUDA
    char pciBusId[16];
    cudaError_t err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_id);
    if (err != cudaSuccess) {
         logError("cudaDeviceGetPCIBusId failed with error code %d\n", err);
         return status_t::ERROR;
    }
    *bus_id = std::string(pciBusId);
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    char pciBusId[16];
    hipError_t res = hipDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_id);
    if (res != hipSuccess) {
         logError("hipDeviceGetPCIBusId failed with error code %d\n", res);
         return status_t::ERROR;
    }
    *bus_id = std::string(pciBusId);
    return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
    char pciBusId[16];
    CNdev dev;
    if (gpuGetDevice(&dev, device_id) != status_t::SUCCESS)
        return status_t::ERROR;
    cnDeviceGetPCIBusId(pciBusId, 16, dev);
    *bus_id = std::string(pciBusId);
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    // TODO
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
}

status_t gpuGetDeviceCount(int *device_count) {
#ifdef ENABLE_CUDA
    cudaError_t err = cudaGetDeviceCount(device_count);
    if (err != cudaSuccess) {
         logError("cudaGetDeviceCount failed with error code %d\n", err);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    hipError_t res = hipGetDeviceCount(device_count);
    if (res != hipSuccess) {
         logError("hipGetDeviceCount failed with error code %d\n", res);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
    CNresult res = cnDeviceGetCount(device_count);
    if (res != CN_SUCCESS) {
         logError("cnDeviceGetCount failed with error code %d\n", res);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    // TODO
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
}

status_t gpuGetDeviceMemory(uint64_t *free_size, uint64_t *total_size) {
#ifdef ENABLE_CUDA
    cudaError_t err = cudaMemGetInfo(free_size, total_size);
    if (err != cudaSuccess) {
         logError("cudaMemGetInfo failed with error code %d\n", err);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    hipError_t res = hipMemGetInfo(free_size, total_size);
    if (res != hipSuccess) {
         logError("hipMemGetInfo failed with error code %d\n", res);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
    CNresult ret = cnMemGetInfo((cn_uint64_t *)free_size, (cn_uint64_t *)total_size);
    if (ret != CN_SUCCESS) {
        logError("failed to get cnMemGetInfo %d.", ret);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    // TODO
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
}

status_t gpuSetDevice(int device_id) {
#ifdef ENABLE_CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
         logError("cudaSetDevice failed with error code %d\n", err);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    hipError_t res = hipSetDevice(device_id);
    if (res != hipSuccess) {
         logError("hipSetDevice failed with error code %d\n", res);
         return status_t::ERROR;
    }
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
}

} // namespace hddt
