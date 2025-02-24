/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "mem_type.h"
#include <mem.h>

namespace hddt {

status_t Memory::init() { return this->memoryClass->init(); }

status_t Memory::free() { return this->memoryClass->free(); }

// create memory class according to memory type
std::unique_ptr<MemoryBase> Memory::createMemoryClass(MemoryType mem_type) {
  logError("[DEBUG] createMemoryClass() called with mem_type={}",
           static_cast<int>(mem_type));
  switch (mem_type) {
  case MemoryType::CPU:
    return std::make_unique<HostMemory>(this->hddtDeviceId,
                                        this->hddtMemoryType);
  case MemoryType::NVIDIA_GPU:
    return std::make_unique<CudaMemory>(this->hddtDeviceId,
                                        this->hddtMemoryType);
  case MemoryType::AMD_GPU:
    return std::make_unique<RocmMemory>(this->hddtDeviceId,
                                        this->hddtMemoryType);
  case MemoryType::CAMBRICON_MLU:
    return std::make_unique<NeuwareMemory>(this->hddtDeviceId,
                                           this->hddtMemoryType);
  case MemoryType::HUAWEI_ASCEND_NPU:
    return std::make_unique<HuaweiMemory>(this->hddtDeviceId,
                                          this->hddtMemoryType);
  default:
    logError("[ERROR] Invalid memory type: {}", static_cast<int>(mem_type));
    return nullptr;
  }
}

// copy data from host to device
status_t Memory::copy_host_to_device(void *dest, const void *src, size_t size) {
  return this->memoryClass->copy_host_to_device(dest, src, size);
}

// copy data from device to host
status_t Memory::copy_device_to_host(void *dest, const void *src, size_t size) {
  return this->memoryClass->copy_device_to_host(dest, src, size);
}

// copy data from device to device
status_t Memory::copy_device_to_device(void *dest, const void *src,
                                       size_t size) {
  return this->memoryClass->copy_device_to_device(dest, src, size);
}

status_t Memory::allocate_buffer(void **addr, size_t size) {
  return this->memoryClass->allocate_buffer(addr, size);
}

status_t Memory::allocate_peerable_buffer(void **addr, size_t size) {
  return this->memoryClass->allocate_peerable_buffer(addr, size);
}

status_t Memory::free_buffer(void *addr) {
  return this->memoryClass->free_buffer(addr);
}

// get memory type
MemoryType Memory::get_MemoryType() { return this->hddtMemoryType; }

// get init status
status_t Memory::get_init_Status() { return this->initStatus; }

// get device id
int Memory::get_DeviceId() { return this->hddtDeviceId; }

// reset device id and memory type
status_t Memory::set_DeviceId_and_MemoryType(int device_id,
                                             MemoryType mem_type) {
  logError("[DEBUG] Setting device_id={}, memory_type={}", this->hddtDeviceId,
           static_cast<int>(this->hddtMemoryType));

  if (mem_type == MemoryType::DEFAULT) { // 未指定mem_type, 则根据系统决定
    this->hddtMemoryType = MemoryType::CPU;

#ifdef ENABLE_CUDA
    this->hddtMemoryType = MemoryType::NVIDIA_GPU;
#endif

#ifdef ENABLE_ROCM
    this->hddtMemoryType = MemoryType::AMD_GPU;
#endif

#ifdef ENABLE_NEUWARE
    this->hddtMemoryType = MemoryType::CAMBRICON_MLU;
#endif

#ifdef ENABLE_HUAWEI
    this->hddtMemoryType = MemoryType::HUAWEI_ASCEND_NPU;
#endif

    this->initStatus = status_t::SUCCESS;
  } else {
    this->initStatus = status_t::SUCCESS;
    if (mem_type == MemoryType::NVIDIA_GPU) {
#ifndef ENABLE_CUDA
      throw std::runtime_error("NVIDIA GPU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    } else if (mem_type == MemoryType::AMD_GPU) {
#ifndef ENABLE_ROCM
      throw std::runtime_error("AMD GPU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    } else if (mem_type == MemoryType::CAMBRICON_MLU) {
#ifndef ENABLE_NEUWARE
      throw std::runtime_error("Cambricon MLU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    } else if (mem_type == MemoryType::HUAWEI_ASCEND_NPU) {
#ifndef ENABLE_HUAWEI
      throw std::runtime_error("Huawei GPU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    }

    this->hddtMemoryType = mem_type;
  }
  this->hddtDeviceId = device_id;
  logError("[DEBUG] Setting device_id %d, memory_type %d", this->hddtDeviceId,
           static_cast<int>(this->hddtMemoryType));
  this->memoryClass = this->createMemoryClass(this->hddtMemoryType);
  this->memoryClass->init();

  return this->initStatus;
}

} // namespace hddt
