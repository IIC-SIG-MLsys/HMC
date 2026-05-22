#include "p2p_endpoint.h"

namespace hmc {

P2pEndpoint::P2pEndpoint(std::unique_ptr<P2pTransport> transport,
                         void *local_ptr, void *peer_ptr, int peer_rank)
    : transport_(std::move(transport)), local_ptr_(local_ptr),
      peer_ptr_(peer_ptr), peer_rank_(peer_rank) {
  role = EndpointType::Client;
}

status_t P2pEndpoint::writeData(size_t local_off, size_t remote_off,
                                size_t size) {
  if (!transport_ || !local_ptr_ || !peer_ptr_) return status_t::ERROR;
  void *src = static_cast<char *>(local_ptr_) + local_off;
  void *dst = static_cast<char *>(peer_ptr_) + remote_off;
  return transport_->copy(dst, src, size);
}

status_t P2pEndpoint::readData(size_t local_off, size_t remote_off,
                               size_t size) {
  if (!transport_ || !local_ptr_ || !peer_ptr_) return status_t::ERROR;
  void *dst = static_cast<char *>(local_ptr_) + local_off;
  void *src = static_cast<char *>(peer_ptr_) + remote_off;
  return transport_->copy(dst, src, size);
}

status_t P2pEndpoint::writeDataNB(size_t lo, size_t ro, size_t sz,
                                  uint64_t *wr_id) {
  if (wr_id) *wr_id = 0;
  return writeData(lo, ro, sz);
}
status_t P2pEndpoint::readDataNB(size_t lo, size_t ro, size_t sz,
                                 uint64_t *wr_id) {
  if (wr_id) *wr_id = 0;
  return readData(lo, ro, sz);
}
status_t P2pEndpoint::waitWrId(uint64_t) { return status_t::SUCCESS; }
status_t P2pEndpoint::waitWrIdMulti(const std::vector<uint64_t> &,
                                    std::chrono::milliseconds) {
  return status_t::SUCCESS;
}
status_t P2pEndpoint::uhm_send(void *, const size_t, MemoryType) {
  return status_t::UNSUPPORT;
}
status_t P2pEndpoint::uhm_recv(void *, const size_t, size_t *, MemoryType) {
  return status_t::UNSUPPORT;
}
status_t P2pEndpoint::closeEndpoint() {
  if (transport_) { transport_->destroy(); transport_.reset(); }
  return status_t::SUCCESS;
}

} // namespace hmc
