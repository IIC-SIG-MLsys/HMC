#ifndef HMC_P2P_TRANSPORT_H
#define HMC_P2P_TRANSPORT_H

#include <cstdint>
#include <memory>
#include <mem.h>

namespace hmc {

static const size_t P2P_HANDLE_SIZE = 64;

class P2pTransport {
public:
  P2pTransport() = default;
  virtual ~P2pTransport() = default;

  virtual status_t init(int device, void *buf, size_t size) = 0;
  virtual status_t exportHandle(void *handle_buf) = 0;
  virtual status_t importHandle(const void *handle_buf, void *&ptr,
                                int peer_device = -1) = 0;
  virtual status_t copy(void *dst, void *src, size_t sz) = 0;
  virtual status_t destroy() = 0;

  static std::unique_ptr<P2pTransport> create(MemoryType type);
};

} // namespace hmc

#endif
