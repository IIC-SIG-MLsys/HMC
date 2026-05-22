#ifndef HMC_P2P_TRANSPORT_CAMBRICON_H
#define HMC_P2P_TRANSPORT_CAMBRICON_H

#include "p2p_transport.h"

#ifdef ENABLE_NEUWARE
#include <cnrt.h>
#endif

namespace hmc {

class CambriconTransport : public P2pTransport {
public:
  CambriconTransport() = default;
  ~CambriconTransport() override;

  status_t init(int device, void *buf, size_t size) override;
  status_t exportHandle(void *handle_buf) override;
  status_t importHandle(const void *handle_buf, void *&ptr,
                        int peer_device = -1) override;
  status_t copy(void *dst, void *src, size_t sz) override;
  status_t destroy() override;

private:
#ifdef ENABLE_NEUWARE
  int device_ = -1;
  void *buf_ = nullptr;
  void *peer_ptr_ = nullptr;
#endif
};

} // namespace hmc

#endif
