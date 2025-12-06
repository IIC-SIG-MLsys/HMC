/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "../utils/log.h"
#include <hmc.h>

namespace hmc {

ConnBuffer::ConnBuffer(int device_id, size_t buffer_size, MemoryType mem_type)
    : device_id(device_id), buffer_size(buffer_size), mem_type(mem_type) {
  mem_ops = new Memory(device_id, mem_type);
  mem_ops->allocatePeerableBuffer(&ptr, buffer_size);
}

ConnBuffer::~ConnBuffer() {
  if (mem_ops) {
    mem_ops->freeBuffer(ptr);
    mem_ops->free();
    // delete mem_ops;
    mem_ops = nullptr;
  }
  ptr = nullptr;
}

// 从CPU向ConnBuffer写入数据
status_t ConnBuffer::writeFromCpu(void *src, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("writeFromCpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  return mem_ops->copyHostToDevice(static_cast<char *>(ptr) + bias, src, size);
}

// 从ConnBuffer读取数据到CPU
status_t ConnBuffer::readToCpu(void *dest, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("readToCpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    memcpy(dest, static_cast<char *>(ptr) + bias, size);
    return status_t::SUCCESS;
  }
  return mem_ops->copyDeviceToHost(dest, static_cast<char *>(ptr) + bias, size);
}

// 从GPU向ConnBuffer写入数据
status_t ConnBuffer::writeFromGpu(void *src, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("writeFromGpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    logError("Error write data from GPU to CPU using CPU ConnBuffer, Please use gpu mem_ops.");
    return status_t::ERROR;
  }
  return mem_ops->copyDeviceToDevice(static_cast<char *>(ptr) + bias, src, size);
}

// 从ConnBuffer读取数据到GPU
status_t ConnBuffer::readToGpu(void *dest, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("readToGpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    logError("Error read data from CPU to GPU using CPU ConnBuffer, Please use gpu mem_ops.");
    return status_t::ERROR;
  }
  return mem_ops->copyDeviceToDevice(dest, static_cast<char *>(ptr) + bias, size);
}

status_t ConnBuffer::copyWithin(size_t dst_bias, size_t src_bias, size_t size) {
  if (dst_bias + size > buffer_size || src_bias + size > buffer_size) {
    logError("copyWithin: Invalid bias/size");
    return status_t::ERROR;
  }
  if (size == 0 || dst_bias == src_bias) {
    return status_t::SUCCESS;
  }

  void *dst = static_cast<char *>(ptr) + dst_bias;
  void *src = static_cast<char *>(ptr) + src_bias;

  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    memmove(dst, src, size);
    return status_t::SUCCESS;
  }

  return mem_ops->copyDeviceToDevice(dst, src, size);
}
}
