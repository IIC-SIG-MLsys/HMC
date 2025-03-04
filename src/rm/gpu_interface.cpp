/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */

#include "gpu_interface.h"

namespace hddt {

status_t gpuInit() {
#ifdef ENABLE_CUDA
    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        logError("cuInit failed with error code %d\n", res);
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

template<typename T>
status_t gpuGetDevice(T *dev, int mlu_ordinal) {
#ifdef ENABLE_CUDA
    CUdevice device;
    CUresult res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        logError("cuDeviceGet failed with error code %d\n", res);
        return status_t::ERROR;
    }
    *dev = device;
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    int device;
    hipError_t res = hipGetDevice(&device);
    if (res != hipSuccess) {
        logError("hipGetDevice failed with error code %d\n", res);
        return status_t::ERROR;
    }
    *dev = device;
    return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
    CNdev device;
    CNresult res = cnDeviceGet(&device, ordinal);
    if (res != CN_SUCCESS) {
        logError("cnDeviceGet failed with error code %d", res);
        return status_t::ERROR;
    }
    *dev = device;
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    // TODO
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
}

status_t gpuGetPcieBusId(std::string *bus_id, int device_id) {
#ifdef ENABLE_CUDA
    char pciBusId[16];
    CUresult res = cuDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_id);
    if (res != CUDA_SUCCESS) {
        logError("cuDeviceGetPCIBusId failed with error code %d\n", res);
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
    if (gpuGetDevice(&dev, device_id)!=status_t::SUCCESS) return status_t::ERROR;
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
    CUresult res = cuDeviceGetCount(device_count);
    if (res != CUDA_SUCCESS) {
        logError("cuDeviceGetCount failed with error code %d\n", res);
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
    CNresult ret = cnDeviceGetCount(device_count);
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
    cudaMemGetInfo(free_size, total_size);
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    hipMemGetInfo(free_size, total_size);
    return status_t::SUCCESS;
#elif defined(ENABLE_NEUWARE)
    cnMemGetInfo((cn_uint64_t *)free_size, (cn_uint64_t *)total_size);
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
    cudaSetDevice(device_id);
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    hipSetDevice(device_id);
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
};

template<typename T>
status_t gpuSetCtx(T &ctx) {
#ifdef ENABLE_NEUWARE
    CNresult ret = cnCtxSetCurrent(ctx); // CNcontext
    if (ret != CN_SUCCESS) {
      logError("failed to set cnCtx %d.", ret);
      return status_t::ERROR;
    }
    return status_t::SUCCESS;
#else
    return status_t::UNSUPPORT;
#endif
};

template<typename T>
status_t gpuCreateContext(T *ctx, int device_id) {
#ifdef ENABLE_CUDA
    CUdevice device;
    cuDeviceGet(&device, device_id);
    CUcontext context;
    CUresult res = cuCtxCreate(&context, 0, device);
    if (res != CUDA_SUCCESS) {
        logError("cuCtxCreate failed with error code %d\n", res);
        return status_t::ERROR;
    }
    *ctx = context;
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    return status_t::UNSUPPORT;
#elif defined(ENABLE_NEUWARE)
    CNresult res = cnCtxCreate(ctx, 0, device_id);
    if (res != CN_SUCCESS) {
        logError("cnCtxCreate failed with error code %d", res);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    return status_t::UNSUPPORT;
#else
    return status_t::UNSUPPORT;
#endif
}

template<typename T>
status_t gpuFreeContext(T ctx) {
#ifdef ENABLE_CUDA
    CUresult res = cuCtxDestroy(ctx);
    if (res != CUDA_SUCCESS) {
        logError("cuCtxDestroy failed with error code %d\n", res);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_ROCM)
    return status_t::UNSUPPORT;
#elif defined(ENABLE_NEUWARE)
    CNresult res = cnCtxDestroy(ctx);
    if (res != CN_SUCCESS) {
        logError("cnCtxDestroy failed with error code %d", res);
        return status_t::ERROR;
    }
    return status_t::SUCCESS;
#elif defined(ENABLE_HUAWEI)
    // TODO
    return status_t::UNSUPPORT;
#else
    return status_t::UNSUPPORT;
#endif
}

}
