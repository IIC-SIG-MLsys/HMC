/**
 * @file memory.h
 * @brief Memory management interface for heterogeneous device environments.
 *
 * This header defines unified APIs for allocating, freeing, and copying memory
 * across CPUs, GPUs, MLUs, NPUs, and other accelerators. The abstraction allows
 * developers to work seamlessly with different device backends using a single
 * unified interface.
 *
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */

#ifndef MEMORY_H
#define MEMORY_H

#include "status.h"
#include <cstring>
#include <memory>

namespace hmc {

#define ACCEL_PAGE_SIZE (64 * 1024) ///< Default accelerator page size (64KB)
typedef uint64_t CNaddr;            ///< Generic device address type

/**
 * @enum MemoryType
 * @brief Supported memory types across different platforms.
 */
enum class MemoryType {
  DEFAULT,       ///< Automatically determined by the system
  CPU,           ///< Host system memory
  NVIDIA_GPU,    ///< NVIDIA CUDA GPU memory
  AMD_GPU,       ///< AMD ROCm GPU memory
  CAMBRICON_MLU, ///< Cambricon MLU accelerator memory
  MOORE_GPU,      ///< Moore Threads GPU memory
  HUAWEI_ASCEND_NPU
};
// Future extensions: NVIDIA_GPU_MANAGED, AMD_GPU_MANAGED

/**
 * @brief Returns the memory type supported by the current system.
 */
MemoryType memory_supported();

/**
 * @brief Checks whether DMA-BUF sharing is supported on this platform.
 * @return True if DMA-BUF interop is available.
 */
bool memory_dmabuf_supported();

/**
 * @class MemoryBase
 * @brief Abstract base class for all memory backends.
 *
 * Each subclass (e.g., CUDA, ROCm, CNRT) must implement device-specific
 * memory allocation and copy operations. Provides the foundation for the
 * high-level `Memory` abstraction.
 */
class MemoryBase {
protected:
  int device_id;       ///< Device index
  MemoryType mem_type; ///< Memory type

public:
  MemoryBase(int device_id, MemoryType mem_type)
      : device_id(device_id), mem_type(mem_type) {}
  virtual ~MemoryBase() {}

  // ----- Lifecycle -----
  virtual status_t init() = 0;
  virtual status_t free() = 0;

  // ----- Allocation -----
  virtual status_t allocateBuffer(void **addr, size_t size) = 0;
  virtual status_t allocatePeerableBuffer(void **addr, size_t size) = 0;
  virtual status_t freeBuffer(void *addr) = 0;

  // ----- Copy Operations -----
  virtual status_t copyHostToDevice(void *dest, const void *src,
                                    size_t size) = 0;
  virtual status_t copyDeviceToHost(void *dest, const void *src,
                                    size_t size) = 0;
  virtual status_t copyDeviceToDevice(void *dest, const void *src,
                                      size_t size) = 0;
};

/**
 * @class Memory
 * @brief Unified high-level interface for memory management across devices.
 *
 * The `Memory` class provides a device-agnostic way to allocate and transfer
 * memory. It automatically selects the appropriate backend implementation
 * (CUDA, ROCm, MLU, etc.) based on the target device and `MemoryType`.
 */
class Memory {
private:
  int hmcDeviceId;                         ///< Target device ID
  MemoryType hmcMemoryType;                ///< Selected memory type
  std::unique_ptr<MemoryBase> memoryClass; ///< Backend implementation
  status_t initStatus;                     ///< Initialization status

public:
  /**
   * @brief Constructor that sets the device ID and memory type.
   * @param device_id Device index to use.
   * @param mem_type Memory type (defaults to system-detected type).
   */
  Memory(int device_id, MemoryType mem_type = MemoryType::DEFAULT) {
    this->setDeviceIdAndMemoryType(device_id, mem_type);
  }

  /**
   * @brief Destructor. Automatically frees allocated memory.
   */
  ~Memory() { this->free(); }

  /**
   * @brief Creates a platform-specific memory backend (e.g., CUDA, ROCm).
   */
  std::unique_ptr<MemoryBase> createMemoryClass(MemoryType mem_type);

  /**
   * @brief Initializes the selected memory backend.
   */
  status_t init();

  /**
   * @brief Frees all allocated resources for this memory instance.
   */
  status_t free();

  // ----- Data Transfer -----

  status_t copyHostToDevice(void *dest, const void *src, size_t size);
  status_t copyDeviceToHost(void *dest, const void *src, size_t size);
  status_t copyDeviceToDevice(void *dest, const void *src, size_t size);

  // ----- Buffer Allocation -----

  status_t allocateBuffer(void **addr, size_t size);
  status_t allocatePeerableBuffer(void **addr, size_t size);
  status_t freeBuffer(void *addr);

  // ----- Configuration & Info -----

  status_t setDeviceIdAndMemoryType(int device_id,
                                    MemoryType mem_type = MemoryType::DEFAULT);

  MemoryType getMemoryType();
  status_t getInitStatus();
  int getDeviceId();
};

} // namespace hmc

#endif // MEMORY_H
