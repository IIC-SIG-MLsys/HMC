#include "p2p_transport_cambricon.h"

#include "utils/log.h"

namespace hmc {

#ifdef ENABLE_NEUWARE

CambriconTransport::~CambriconTransport() { destroy(); }

status_t CambriconTransport::init(int device, void *buf, size_t size) {
  device_ = device;
  buf_ = buf;
  cnrtRet_t ret = cnrtSetDevice(device);
  if (ret != cnrtSuccess) {
    logError("CambriconTransport cnrtSetDevice(%d) failed: %d", device, ret);
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t CambriconTransport::exportHandle(void *handle_buf) {
  cnrtIpcMemHandle ipc_handle;
  cnrtRet_t ret = cnrtAcquireMemHandle(&ipc_handle, buf_);
  if (ret != cnrtSuccess) {
    logError("CambriconTransport cnrtAcquireMemHandle failed: %d", ret);
    return status_t::ERROR;
  }
  *reinterpret_cast<uint64_t *>(handle_buf) = reinterpret_cast<uint64_t>(ipc_handle);
  return status_t::SUCCESS;
}

status_t CambriconTransport::importHandle(const void *handle_buf, void *&ptr,
                                          int peer_device) {
  if (peer_ptr_) { cnrtUnMapMemHandle(peer_ptr_); peer_ptr_ = nullptr; }
  uint64_t handle_val = *reinterpret_cast<const uint64_t *>(handle_buf);
  int flags = (peer_device >= 0 && peer_device != device_) ? 0x2 : 0;
  cnrtRet_t ret = cnrtMapMemHandle(&ptr, reinterpret_cast<cnrtIpcMemHandle>(handle_val), flags);
  if (ret != cnrtSuccess) {
    logError("CambriconTransport cnrtMapMemHandle failed: %d", ret);
    return status_t::ERROR;
  }
  peer_ptr_ = ptr;
  return status_t::SUCCESS;
}

status_t CambriconTransport::copy(void *dst, void *src, size_t sz) {
  cnrtRet_t ret = cnrtMemcpy(dst, src, sz, cnrtMemcpyDevToDev);
  if (ret != cnrtSuccess) {
    logError("CambriconTransport cnrtMemcpy(DevToDev) failed: %d", ret);
    return status_t::ERROR;
  }
  ret = cnrtQueueSync(nullptr);
  if (ret != cnrtSuccess) return status_t::ERROR;
  return status_t::SUCCESS;
}

status_t CambriconTransport::destroy() {
  if (peer_ptr_) {
    cnrtUnMapMemHandle(peer_ptr_);
    peer_ptr_ = nullptr;
  }
  return status_t::SUCCESS;
}

#else

CambriconTransport::~CambriconTransport() = default;

status_t CambriconTransport::init(int, void *, size_t) {
  return status_t::UNSUPPORT;
}
status_t CambriconTransport::exportHandle(void *) {
  return status_t::UNSUPPORT;
}
status_t CambriconTransport::importHandle(const void *, void *&, int) {
  return status_t::UNSUPPORT;
}
status_t CambriconTransport::copy(void *, void *, size_t) {
  return status_t::UNSUPPORT;
}
status_t CambriconTransport::destroy() {
  return status_t::UNSUPPORT;
}

#endif

} // namespace hmc
