#include "p2p_transport_rocm.h"

#include "utils/log.h"

namespace hmc {

#ifdef ENABLE_ROCM

RocmTransport::~RocmTransport() { destroy(); }

status_t RocmTransport::init(int device, void *buf, size_t size) {
  (void)size;
  device_ = device;
  buf_ = buf;
  hipError_t err = hipSetDevice(device);
  if (err != hipSuccess) {
    logError("RocmTransport hipSetDevice(%d) failed: %d", device, err);
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RocmTransport::exportHandle(void *handle_buf) {
  hipIpcMemHandle_t ipc_handle;
  hipError_t err = hipIpcGetMemHandle(&ipc_handle, buf_);
  if (err != hipSuccess) {
    logError("RocmTransport hipIpcGetMemHandle failed: %d", err);
    return status_t::ERROR;
  }
  memcpy(handle_buf, &ipc_handle, sizeof(ipc_handle));
  return status_t::SUCCESS;
}

status_t RocmTransport::importHandle(const void *handle_buf, void *&ptr,
                                     int peer_device) {
  if (peer_ptr_) { hipIpcCloseMemHandle(peer_ptr_); peer_ptr_ = nullptr; }
  peer_device_ = peer_device;
  hipIpcMemHandle_t ipc_handle;
  memcpy(&ipc_handle, handle_buf, sizeof(ipc_handle));

  unsigned int flags = (peer_device >= 0 && peer_device != device_)
                           ? hipIpcMemLazyEnablePeerAccess : 0;
  hipError_t err = hipIpcOpenMemHandle(&ptr, ipc_handle, flags);
  if (err != hipSuccess) {
    logError("RocmTransport hipIpcOpenMemHandle failed: %d", err);
    return status_t::ERROR;
  }
  peer_ptr_ = ptr;
  return status_t::SUCCESS;
}

status_t RocmTransport::copy(void *dst, void *src, size_t sz) {
  hipError_t err = hipMemcpy(dst, src, sz, hipMemcpyDeviceToDevice);
  if (err != hipSuccess) {
    logError("RocmTransport hipMemcpy(D2D) failed: %d", err);
    return status_t::ERROR;
  }
  err = hipDeviceSynchronize();
  if (err != hipSuccess) { logError("RocmTransport hipDeviceSynchronize failed: %d", err); return status_t::ERROR; }
  return status_t::SUCCESS;
}

status_t RocmTransport::destroy() {
  if (peer_ptr_) {
    hipIpcCloseMemHandle(peer_ptr_);
    peer_ptr_ = nullptr;
  }
  return status_t::SUCCESS;
}

#else

RocmTransport::~RocmTransport() = default;
status_t RocmTransport::init(int, void *, size_t) { return status_t::UNSUPPORT; }
status_t RocmTransport::exportHandle(void *) { return status_t::UNSUPPORT; }
status_t RocmTransport::importHandle(const void *, void *&, int) {
  return status_t::UNSUPPORT;
}
status_t RocmTransport::copy(void *, void *, size_t) { return status_t::UNSUPPORT; }
status_t RocmTransport::destroy() { return status_t::UNSUPPORT; }

#endif

} // namespace hmc
