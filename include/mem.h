/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef MEMORY_H
#define MEMORY_H

#include "status.h"

#include <cstring>
#include <memory>

namespace hmc {

#define ACCEL_PAGE_SIZE (64 * 1024)
typedef uint64_t CNaddr;

enum class MemoryType {
  DEFAULT, // 默认情况, 系统决定
  CPU,
  NVIDIA_GPU,
  AMD_GPU,
  CAMBRICON_MLU,
  HUAWEI_ASCEND_NPU
}; // todo: NVIDIA_GPU_MANAGED, AMD_GPU_MANAGED

MemoryType memory_supported();
bool memory_dmabuf_supported();

class MemoryBase {
protected:
  int device_id;
  MemoryType mem_type;

public:
  MemoryBase(int device_id, MemoryType mem_type)
      : device_id(device_id), mem_type(mem_type){};
  virtual ~MemoryBase(){};

  virtual status_t init() = 0;
  virtual status_t free() = 0;
  virtual status_t allocateBuffer(void **addr, size_t size) = 0;
  virtual status_t allocatePeerableBuffer(void **addr, size_t size) = 0;
  virtual status_t freeBuffer(void *addr) = 0;

  virtual status_t copyHostToDevice(void *dest, const void *src,
                                    size_t size) = 0;
  virtual status_t copyDeviceToHost(void *dest, const void *src,
                                    size_t size) = 0;
  virtual status_t copyDeviceToDevice(void *dest, const void *src,
                                      size_t size) = 0;
};

/*
 * 可由用户指定设备类型和设备号，并自动创建相应的Memory类实例
 * 也可由系统自动识别支持device的类型
 */
class Memory {
private:
  int hmcDeviceId;
  MemoryType hmcMemoryType;
  std::unique_ptr<MemoryBase> memoryClass;
  status_t initStatus;

public:
  Memory(int device_id, MemoryType mem_type = MemoryType::DEFAULT) {
    this->setDeviceIdAndMemoryType(device_id, mem_type);
  }

  ~Memory() { this->free(); }

  std::unique_ptr<MemoryBase> createMemoryClass(MemoryType mem_type);
  status_t init();
  status_t free();

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);

  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  status_t setDeviceIdAndMemoryType(int device_id,
                                    MemoryType mem_type = MemoryType::DEFAULT);

  MemoryType getMemoryType();
  status_t getInitStatus();
  int getDeviceId();
};

} // namespace hmc
#endif
