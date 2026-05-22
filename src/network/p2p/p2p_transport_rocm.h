#ifndef HMC_P2P_TRANSPORT_ROCM_H
#define HMC_P2P_TRANSPORT_ROCM_H

#include "p2p_transport.h"

#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace hmc {

class RocmTransport : public P2pTransport {
public:
  RocmTransport() = default;
  ~RocmTransport() override;

  status_t init(int device, void *buf, size_t size) override;
  status_t exportHandle(void *handle_buf) override;
  status_t importHandle(const void *handle_buf, void *&ptr,
                        int peer_device = -1) override;
  status_t copy(void *dst, void *src, size_t sz) override;
  status_t destroy() override;

private:
#ifdef ENABLE_ROCM
  int device_ = -1;
  int peer_device_ = -1;
  void *buf_ = nullptr;
  void *peer_ptr_ = nullptr;
#endif
};

} // namespace hmc

#endif
