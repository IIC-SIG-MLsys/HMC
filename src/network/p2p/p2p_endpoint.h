#ifndef HMC_P2P_ENDPOINT_H
#define HMC_P2P_ENDPOINT_H

#include "../net.h"
#include "p2p_transport.h"
#include <memory>

namespace hmc {

class P2pEndpoint : public Endpoint {
public:
  P2pEndpoint(std::unique_ptr<P2pTransport> transport, void *local_ptr,
              void *peer_ptr, int peer_rank);

  status_t writeData(size_t local_off, size_t remote_off, size_t size) override;
  status_t readData(size_t local_off, size_t remote_off, size_t size) override;

  status_t writeDataNB(size_t local_off, size_t remote_off, size_t size,
                       uint64_t *wr_id) override;
  status_t readDataNB(size_t local_off, size_t remote_off, size_t size,
                      uint64_t *wr_id) override;
  status_t waitWrId(uint64_t wr_id) override;
  status_t waitWrIdMulti(const std::vector<uint64_t> &target_wr_ids,
                         std::chrono::milliseconds timeout) override;

  status_t uhm_send(void *input_buffer, const size_t send_flags,
                    MemoryType mem_type) override;
  status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                    size_t *recv_flags, MemoryType mem_type) override;

  status_t closeEndpoint() override;

private:
  std::unique_ptr<P2pTransport> transport_;
  void *local_ptr_;
  void *peer_ptr_;
  int peer_rank_;
};

} // namespace hmc

#endif
