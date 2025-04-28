/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_MEM_TYPE_H
#define HMC_MEM_TYPE_H

#include <log.h>
#include <mem.h>

// #include <acl/acl.h>  // -> TODO: ResourceManager

namespace hmc {

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
  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);
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
  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);
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
  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);
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
  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);
};

class HuaweiMemory : public MemoryBase {
public:
  HuaweiMemory(int device_id, MemoryType mem_type)
      : MemoryBase(device_id, mem_type){};
  ~HuaweiMemory() { this->free(); };

  status_t init();
  status_t free();
  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);

private:
  // aclrtContext context_; // 上下文 -> TODO: ResourceManager
  // aclrtStream stream_;   // 流
  // bool is_initialized_ = false;
};

} // namespace hmc

#endif