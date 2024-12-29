#ifndef MEMORY_H
#define MEMORY_H

#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <hddt.h>
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
  status_t free_buffer(void *addr);

  status_t
  set_DeviceId_and_MemoryType(int device_id,
                              MemoryType mem_type = MemoryType::DEFAULT);

  MemoryType get_MemoryType();
  status_t get_init_Status();
  int get_DeviceId();
};

class HostMemory : public MemoryBase {
public:
  HostMemory(int device_id, MemoryType mem_type)
      : MemoryBase(device_id, mem_type) {
    status_t sret;
    sret = this->init();
    if (sret != status_t::SUCCESS) {
      logError("HostMemory init mem_ops err %s.", status_to_string(sret));
      exit(1);
    }
  };
  ~HostMemory() { this->free(); }
  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);
};

class CudaMemory : public MemoryBase {
public:
  CudaMemory(int device_id, MemoryType mem_type)
      : MemoryBase(device_id, mem_type) {
    status_t sret;
    sret = this->init();
    if (sret != status_t::SUCCESS) {
      logError("CudaMemory init mem_ops err %s.", status_to_string(sret));
      exit(1);
    }
  };
  ~CudaMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);
};

class RocmMemory : public MemoryBase {
public:
  RocmMemory(int device_id, MemoryType mem_type)
      : MemoryBase(device_id, mem_type) {
    status_t sret;
    sret = this->init();
    if (sret != status_t::SUCCESS) {
      logError("RocmMemory init mem_ops err %s.", status_to_string(sret));
      exit(1);
    }
  };
  ~RocmMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);
};

class NeuwareMemory : public MemoryBase {
public:
  CNaddr mlu_addr;

public:
  NeuwareMemory(int device_id, MemoryType mem_type)
      : MemoryBase(device_id, mem_type) {
    status_t sret;
    sret = this->init();
    if (sret != status_t::SUCCESS) {
      logError("NeuwareMemory init mem_ops err %s.", status_to_string(sret));
      exit(1);
    }
  };
  ~NeuwareMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);
};

class HuaweiMemory : public MemoryBase {
public:
  HuaweiMemory(int device_id, MemoryType mem_type);
  ~HuaweiMemory();

  status_t init();
  status_t free();
  status_t allocate_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);

private:
  aclrtContext context_; // 上下文
  aclrtStream stream_;   // 流
  bool is_initialized_ = false;
};

} // namespace hddt
#endif
