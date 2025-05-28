/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
 #include "../resource_manager/gpu_interface.h"
 #include "./mem_type.h"
 #include <mem.h>
 #include <musa_runtime.h>  // 必须包含 MUSA 运行时头文件
 namespace hmc {
 #ifdef ENABLE_MUSA
  #include <musa_runtime.h>  // 必须包含 MUSA 运行时头文件
 /*
  * moore gpu memory
  */
 status_t MusaMemory::init() {
   // 如果有初始化操作，可在此添加，例如 musaInit()（假设存在）
   return gpuInit();
 }
 
 status_t MusaMemory::free() {
   return status_t::SUCCESS;
 }
 /*
status_t MusaMemory::allocateBuffer(void **addr, size_t size) {
   musaError_t ret;
 
   if (this->mem_type != MemoryType::MOORE_GPU) {
     return status_t::UNSUPPORT;
   }
 
   logInfo("Allocate memory using musaMalloc.");
   ret = musaMalloc(addr, size);
   if (ret != musaSuccess) {
     logError("failed to allocate memory.");
     return status_t::ERROR;
   }
 
   return status_t::SUCCESS;
}*/

 status_t MusaMemory::allocateBuffer(void **addr, size_t size) {
  if (this->mem_type != MemoryType::MOORE_GPU) {
    return status_t::UNSUPPORT;
  }

  logInfo("Allocate aligned host memory + register for pin/persist.");

  // 1) 在主机上对齐分配一块内存 (128 字节对齐)，长度就是 size
  void *host_ptr = nullptr;
  if (posix_memalign(&host_ptr, /*alignment=*/128, /*length=*/size) != 0 || !host_ptr) {
    logError("posix_memalign failed");
    return status_t::ERROR;
  }

  // 2) 锁页并映射到 GPU
  musaError_t mret = musaHostRegister(
      host_ptr, size,
      musaHostRegisterMapped    // 映射到 GPU 地址空间
    | musaHostRegisterPortable // 多卡可见
  );
  if (mret != musaSuccess) {
    logError("musaHostRegister failed: %d", mret);
    return status_t::ERROR;
  }

  // 3) 拿到 device pointer
  void *dev_ptr = nullptr;
  mret = musaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
  if (mret != musaSuccess) {
    logError("musaHostGetDevicePointer failed: %d", mret);
    musaHostUnregister(host_ptr);
    return status_t::ERROR;
  }
  
  *addr = host_ptr;
  logInfo("Buffer ready: host_ptr=%p, dev_ptr=%p, size=%zu",
          host_ptr, dev_ptr, size);
  return status_t::SUCCESS;
}

 
 status_t MusaMemory::allocatePeerableBuffer(void **addr, size_t size) {
   size_t buf_size = (size + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);
   return this->allocateBuffer(addr, buf_size);
 }
 
 status_t MusaMemory::freeBuffer(void *addr) {
   musaError_t ret;
 
   ret = musaFree(addr);
   if (ret != musaSuccess) {
     logError("failed to free memory");
     return status_t::ERROR;
   }
 
   return status_t::SUCCESS;
 }
 
 status_t MusaMemory::copyHostToDevice(void *dest, const void *src,
                                       size_t size) {
   musaError_t ret;
 
   if (dest == nullptr || src == nullptr) {
     logError("MusaMemory::copyHostToDevice Error.");
     return status_t::ERROR;
   }
 
   ret = musaMemcpy(dest, src, size, musaMemcpyHostToDevice);
   if (ret != musaSuccess) {
     logError("failed to copy memory from host to device");
     return status_t::ERROR;
   }
 
   return status_t::SUCCESS;
 }
 
 status_t MusaMemory::copyDeviceToHost(void *dest, const void *src,
                                       size_t size) {
   musaError_t ret;
 
   if (dest == nullptr || src == nullptr) {
     logError("MusaMemory::copyDeviceToHost Error.");
     return status_t::ERROR;
   }
 
   ret = musaMemcpy(dest, src, size, musaMemcpyDeviceToHost);
   if (ret != musaSuccess) {
     logError("failed to copy memory from device to host");
     return status_t::ERROR;
   }
 
   return status_t::SUCCESS;
 }
 
 status_t MusaMemory::copyDeviceToDevice(void *dest, const void *src,
                                         size_t size) {
   musaError_t ret;
 
   if (dest == nullptr || src == nullptr) {
     logError("MusaMemory::copyDeviceToDevice Error.");
     return status_t::ERROR;
   }
 
   ret = musaMemcpy(dest, src, size, musaMemcpyDeviceToDevice);
   if (ret != musaSuccess) {
     logError("failed to copy memory from device to device");
     return status_t::ERROR;
   }
 
   return status_t::SUCCESS;
 }
 
 #else
 status_t MusaMemory::init() { return status_t::UNSUPPORT; }
 status_t MusaMemory::free() { return status_t::UNSUPPORT; }
 status_t MusaMemory::allocateBuffer(void **addr, size_t size) {
   return status_t::UNSUPPORT;
 }
 status_t MusaMemory::allocatePeerableBuffer(void **addr, size_t size) {
   return status_t::UNSUPPORT;
 }
 status_t MusaMemory::freeBuffer(void *addr) { return status_t::UNSUPPORT; }
 
 status_t MusaMemory::copyHostToDevice(void *dest, const void *src,
                                       size_t size) {
   return status_t::UNSUPPORT;
 }
 status_t MusaMemory::copyDeviceToHost(void *dest, const void *src,
                                       size_t size) {
   return status_t::UNSUPPORT;
 }
 status_t MusaMemory::copyDeviceToDevice(void *dest, const void *src,
                                         size_t size) {
   return status_t::UNSUPPORT;
 }
 #endif
 
 } // namespace hmc
 