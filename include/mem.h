/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef MEMORY_H
#define MEMORY_H

#include "status.h"

#include <cstring>
#include <memory>

namespace hddt {

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
  virtual status_t allocate_buffer(void **addr, size_t size) = 0;
  virtual status_t allocate_peerable_buffer(void **addr, size_t size) = 0;
  virtual status_t free_buffer(void *addr) = 0;

  virtual status_t copy_host_to_device(void *dest, const void *src,
                                       size_t size) = 0;
  virtual status_t copy_device_to_host(void *dest, const void *src,
                                       size_t size) = 0;
  virtual status_t copy_device_to_device(void *dest, const void *src,
                                         size_t size) = 0;
};

/*
 * 新增HddtMemory类，可由用户指定设备类型和设备号，并自动创建相应的Memory类实例
 * 也可由系统自动识别支持device的类型
 *
 */
class Memory {
private:
  int hddtDeviceId;
  MemoryType hddtMemoryType;
  std::unique_ptr<MemoryBase> memoryClass;
  status_t initStatus;

public:
  Memory(int device_id, MemoryType mem_type = MemoryType::DEFAULT) {
    this->set_DeviceId_and_MemoryType(device_id, mem_type);
  }

  ~Memory() { this->free(); }

  std::unique_ptr<MemoryBase> createMemoryClass(MemoryType mem_type);
  status_t init();
  status_t free();

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);

  status_t allocate_buffer(void **addr, size_t size);
  status_t allocate_peerable_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t
  set_DeviceId_and_MemoryType(int device_id,
                              MemoryType mem_type = MemoryType::DEFAULT);

  MemoryType get_MemoryType();
  status_t get_init_Status();
  int get_DeviceId();
};

} // namespace hddt
#endif
