/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_MEM_TYPE_H
#define HDDT_MEM_TYPE_H

#include "../utils/log.h"
#include <mem.h>

namespace hddt {

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
  status_t allocate_peerable_buffer(void **addr, size_t size);
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
  status_t allocate_peerable_buffer(void **addr, size_t size);
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
  status_t allocate_peerable_buffer(void **addr, size_t size);
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
  status_t allocate_peerable_buffer(void **addr, size_t size);
  status_t free_buffer(void *addr);

  status_t copy_host_to_device(void *dest, const void *src, size_t size);
  status_t copy_device_to_host(void *dest, const void *src, size_t size);
  status_t copy_device_to_device(void *dest, const void *src, size_t size);
};

} // namespace hddt

#endif