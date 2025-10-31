/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "./mem_type.h"
#include <mem.h>

namespace hmc {

status_t Memory::init() { return this->memoryClass->init(); }

status_t Memory::free() { return this->memoryClass->free(); }

// create memory class according to device type
std::unique_ptr<MemoryBase> Memory::createMemoryClass(MemoryType mem_type) {
  switch (mem_type) {
  case MemoryType::CPU:
    return std::make_unique<HostMemory>(this->hmcDeviceId, this->hmcMemoryType);
  case MemoryType::NVIDIA_GPU:
    return std::make_unique<CudaMemory>(this->hmcDeviceId, this->hmcMemoryType);
  case MemoryType::AMD_GPU:
    return std::make_unique<RocmMemory>(this->hmcDeviceId, this->hmcMemoryType);
  case MemoryType::CAMBRICON_MLU:
    return std::make_unique<NeuwareMemory>(this->hmcDeviceId,
                                           this->hmcMemoryType);
  case MemoryType::MOORE_GPU:
    return std::make_unique<MusaMemory>(this->hmcDeviceId, this->hmcMemoryType);
  default:
    return nullptr;
  }
}

// copy data from host to device
status_t Memory::copyHostToDevice(void *dest, const void *src, size_t size) {
  return this->memoryClass->copyHostToDevice(dest, src, size);
}

// copy data from device to host
status_t Memory::copyDeviceToHost(void *dest, const void *src, size_t size) {
  return this->memoryClass->copyDeviceToHost(dest, src, size);
}

// copy data from device to device
status_t Memory::copyDeviceToDevice(void *dest, const void *src, size_t size) {
  return this->memoryClass->copyDeviceToDevice(dest, src, size);
}

status_t Memory::allocateBuffer(void **addr, size_t size) {
  return this->memoryClass->allocateBuffer(addr, size);
}

status_t Memory::allocatePeerableBuffer(void **addr, size_t size) {
  return this->memoryClass->allocatePeerableBuffer(addr, size);
}

status_t Memory::freeBuffer(void *addr) {
  return this->memoryClass->freeBuffer(addr);
}

// get memory type
MemoryType Memory::getMemoryType() { return this->hmcMemoryType; }

// get init status
status_t Memory::getInitStatus() { return this->initStatus; }

// get device id
int Memory::getDeviceId() { return this->hmcDeviceId; }

// reset device id and memory type
status_t Memory::setDeviceIdAndMemoryType(int device_id, MemoryType mem_type) {
  if (mem_type == MemoryType::DEFAULT) { // 未指定mem_type, 则根据系统决定
    this->hmcMemoryType = MemoryType::CPU;

#ifdef ENABLE_CUDA
    this->hmcMemoryType = MemoryType::NVIDIA_GPU;
#endif

#ifdef ENABLE_ROCM
    this->hmcMemoryType = MemoryType::AMD_GPU;
#endif

#ifdef ENABLE_NEUWARE
    this->hmcMemoryType = MemoryType::CAMBRICON_MLU;
#endif

#ifdef ENABLE_MUSA
    this->hmcMemoryType = MemoryType::MOORE_GPU;
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
    } else if (mem_type == MemoryType::MOORE_GPU) {
#ifdef ENABLE_MUSA
      throw std::runtime_error("Moore GPU is not supported");
      this->initStatus = status_t::UNSUPPORT;
#endif
    }

    this->hmcMemoryType = mem_type;
  }
  this->hmcDeviceId = device_id;
  this->memoryClass = this->createMemoryClass(this->hmcMemoryType);
  this->memoryClass->init();

  return this->initStatus;
}

} // namespace hmc
