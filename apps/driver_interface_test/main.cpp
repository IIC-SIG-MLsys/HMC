#include <coll.h>
#include <iostream>

#include "rm/gpu_interface.h"
#include "rm/driver_manager.h"

using namespace hddt;

int main() {
  // 创建 GPUManager 实例
  hddt::GPUManager gpuManager;

  // 初始化 GPUManager
  if (gpuManager.init() != hddt::status_t::SUCCESS) {
      std::cerr << "GPUManager 初始化失败！" << std::endl;
      return -1;
  }

  // 获取设备数量
  int deviceCount = gpuManager.getDeviceCount();
  std::cout << "检测到 " << deviceCount << " 个 GPU 设备。" << std::endl;

  // 遍历每个设备，获取并显示其信息
  for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
      hddt::GPUDeviceInfo deviceInfo = gpuManager.getDeviceInfo(deviceId);
      std::cout << "设备 ID: " << deviceInfo.deviceId << std::endl;
      std::cout << "PCIe 总线 ID: " << deviceInfo.pcieBusId << std::endl;
      std::cout << "总内存: " << deviceInfo.totalMemory/1024/1024/1024 << " GB" << std::endl;
      std::cout << "可用内存: " << deviceInfo.freeMemory/1024/1024/1024 << " GB" << std::endl;
      switch (deviceInfo.vendor) {
        case MemoryType::NVIDIA_GPU:
            std::cout << "供应商: NVIDIA" << std::endl;
            break;
        case MemoryType::AMD_GPU:
            std::cout << "供应商: AMD" << std::endl;
            break;
        case MemoryType::CAMBRICON_MLU:
            std::cout << "供应商: 寒武纪" << std::endl;
            break;
        case MemoryType::HUAWEI_ASCEND_NPU:
            std::cout << "供应商: 华为" << std::endl;
            break;
        case MemoryType::CPU:
            std::cout << "CPU" << std::endl;
            break;
        default:
            std::cout << "未知" << std::endl;
            break;
    }
      std::cout << "-------------------------" << std::endl;
  }

  /* gpu interface 接口测试 */
  std::cout << "\n开始测试 GPU 接口函数:" << std::endl;

  // 测试 gpuInit 接口
  if (hddt::gpuInit() != hddt::status_t::SUCCESS) {
      std::cerr << "gpuInit() 失败！" << std::endl;
  } else {
      std::cout << "gpuInit() 成功！" << std::endl;
  }

  // 测试获取设备数量（通过 gpu_interface 接口）
  int gpuCount = 0;
  if (hddt::gpuGetDeviceCount(&gpuCount) != hddt::status_t::SUCCESS) {
      std::cerr << "gpuGetDeviceCount() 失败！" << std::endl;
  } else {
      std::cout << "通过 gpuGetDeviceCount() 检测到 " << gpuCount << " 个 GPU 设备。" << std::endl;
  }

  // 测试获取每个设备的 PCIe 总线 ID和内存信息
  for (int i = 0; i < gpuCount; ++i) {
      std::string busId;
      if (hddt::gpuGetPcieBusId(&busId, i) != hddt::status_t::SUCCESS) {
          std::cerr << "gpuGetPcieBusId() 失败，设备 " << i << std::endl;
      } else {
          std::cout << "设备 " << i << " 的 PCIe 总线 ID: " << busId << std::endl;
      }

      uint64_t freeMem = 0, totalMem = 0;
      if (hddt::gpuGetDeviceMemory(&freeMem, &totalMem) != hddt::status_t::SUCCESS) {
          std::cerr << "gpuGetDeviceMemory() 失败，设备 " << i << std::endl;
      } else {
          std::cout << "设备 " << i << " 内存信息: 总内存 " << totalMem/(1024*1024*1024)
                    << " GB, 可用内存 " << freeMem/(1024*1024*1024) << " GB" << std::endl;
      }
  }

  // 测试设置设备（仅测试第一个设备）
  if (gpuCount > 0) {
#ifndef ENABLE_NEUWARE // skip, neuware 不支持
      if (hddt::gpuSetDevice(0) != hddt::status_t::SUCCESS) {
          std::cerr << "gpuSetDevice(0) 失败！" << std::endl;
      } else {
          std::cout << "gpuSetDevice(0) 成功！" << std::endl;
      }
#endif
  }

#ifdef ENABLE_NEUWARE
  // 如果启用 ENABLE_NEUWARE，测试创建、设置和销毁上下文
  CNcontext ctx;
  if (hddt::gpuCreateContext(&ctx, 0) != hddt::status_t::SUCCESS) {
      std::cerr << "gpuCreateContext() 失败！" << std::endl;
  } else {
      std::cout << "gpuCreateContext() 成功！" << std::endl;

      if (hddt::gpuSetCtx(ctx) != hddt::status_t::SUCCESS) {
          std::cerr << "gpuSetCtx() 失败！" << std::endl;
      } else {
          std::cout << "gpuSetCtx() 成功！" << std::endl;
      }

      if (hddt::gpuFreeContext(ctx) != hddt::status_t::SUCCESS) {
          std::cerr << "gpuFreeContext() 失败！" << std::endl;
      } else {
          std::cout << "gpuFreeContext() 成功！" << std::endl;
      }
  }
#endif

  // 释放 GPUManager 资源
  gpuManager.free();

  return 0;
}

// sudo mpirun -np 2 -host ip1,ip2 ./coll_app