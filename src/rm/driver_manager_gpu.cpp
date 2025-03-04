/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */

#include "driver_manager.h"

namespace hddt {

int GPUManager::getDeviceCount() const {
  return devices.size();
};

GPUDeviceInfo GPUManager::getDeviceInfo(int device_id) const {
  if (device_id < 0 || device_id >= static_cast<int>(devices.size())) {
    return {-1, 0, 0, 0, MemoryType::DEFAULT};
  }
  return devices[device_id];
};

status_t GPUManager::init() {
  // 初始化驱动
  status_t ret = gpuInit();
  if (ret != status_t::SUCCESS) return ret;

  // 获取设备数量
  int device_count = 0;
  if (gpuGetDeviceCount(&device_count) != status_t::SUCCESS || device_count <= 0) {
    logError("No available GPU devices");
    return status_t::ERROR;
  }

  devices.resize(device_count);
#if ENABLE_NEUWARE
  contexts.resize(device_count);
#endif

  // 收集设备信息
  devices.clear();
  for (int dev_id = 0; dev_id < device_count; ++dev_id) {
    GPUDeviceInfo info;
    
    // 获取PCIe总线ID
    if (gpuGetPcieBusId(&info.pcieBusId, dev_id) != status_t::SUCCESS) {
        continue;
    }

    // 获取设备类型和内存
    info.deviceId = dev_id;
#ifdef ENABLE_CUDA
    gpuSetDevice(dev_id);
    info.vendor = MemoryType::NVIDIA_GPU;
#elif defined(ENABLE_ROCM)
    gpuSetDevice(dev_id);
    info.vendor = MemoryType::AMD_GPU;
#elif defined(ENABLE_NEUWARE)
    gpuSetCtx(contexts[dev_id]);
    info.vendor = MemoryType::CAMBRICON_MLU;
#endif
    if (gpuGetDeviceMemory(&info.freeMemory, &info.totalMemory) != status_t::SUCCESS) {
        info.freeMemory = 0;
        info.totalMemory = 0;
    }
    devices.emplace_back(info);
  }

  return status_t::SUCCESS;
}

} // namespace hddt
