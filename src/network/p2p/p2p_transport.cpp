#include "p2p_transport.h"
#include "p2p_transport_cambricon.h"
#include "p2p_transport_rocm.h"

namespace hmc {

std::unique_ptr<P2pTransport> P2pTransport::create(MemoryType type) {
  switch (type) {
  case MemoryType::CAMBRICON_MLU:
    return std::make_unique<CambriconTransport>();
  case MemoryType::AMD_GPU:
    return std::make_unique<RocmTransport>();
  default:
    return nullptr;
  }
}

} // namespace hmc
