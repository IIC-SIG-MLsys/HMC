#include "net_ucx.h"

#include "../utils/log.h"

#include <arpa/inet.h>
#include <cstring>
#include <thread>
#include <chrono>

namespace hmc {

static constexpr std::uint16_t kHandshakeChannel = 1;
static constexpr std::uint32_t kSeqHdr  = 1;
static constexpr std::uint32_t kSeqRkey = 2;

static bool peer_ip_from_conn_request(ucp_conn_request_h cr, std::string &out_ip) {
  ucp_conn_request_attr_t attr;
  std::memset(&attr, 0, sizeof(attr));
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  ucs_status_t st = ucp_conn_request_query(cr, &attr);
  if (st != UCS_OK) {
    logError("ucp_conn_request_query failed: %d", (int)st);
    return false;
  }

  const struct sockaddr *sa =
      reinterpret_cast<const struct sockaddr *>(&attr.client_address);

  char buf[INET6_ADDRSTRLEN] = {0};

  if (sa->sa_family == AF_INET) {
    const struct sockaddr_in *sin =
        reinterpret_cast<const struct sockaddr_in *>(sa);
    if (!inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf))) {
      logError("inet_ntop(AF_INET) failed");
      return false;
    }
    out_ip = buf;
    return true;
  }

  if (sa->sa_family == AF_INET6) {
    const struct sockaddr_in6 *sin6 =
        reinterpret_cast<const struct sockaddr_in6 *>(sa);
    if (!inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf))) {
      logError("inet_ntop(AF_INET6) failed");
      return false;
    }
    out_ip = buf;
    return true;
  }

  logError("peer sockaddr family unsupported: %d", (int)sa->sa_family);
  return false;
}

UCXServer::UCXServer(std::shared_ptr<ConnManager> conn_manager,
                     std::shared_ptr<ConnBuffer> buffer)
    : Server(std::move(conn_manager)),
      buffer_(std::move(buffer)),
      ctx_(std::make_shared<UCXContext>()) {}

UCXServer::~UCXServer() { (void)stopListen(); }

status_t UCXServer::listen(std::string ip, uint16_t port) {
  ip_ = std::move(ip);
  port_ = port;

  if (running_) return status_t::SUCCESS;

  if (!ctx_) ctx_ = std::make_shared<UCXContext>();
  status_t st = ctx_->init();
  if (st != status_t::SUCCESS) {
    logError("UCXServer::listen: UCXContext init failed");
    return st;
  }

  st = ctx_->startListener(ip_, port_);
  if (st != status_t::SUCCESS) {
    logError("UCXServer::listen: startListener failed on %s:%u",
             ip_.c_str(), port_);
    return st;
  }

  running_ = true;
  logInfo("UCXServer listening (UCX listener) on %s:%u", ip_.c_str(), port_);

  accept_running_.store(true, std::memory_order_release);
  accept_th_ = std::thread([this] {
    while (accept_running_.load(std::memory_order_acquire)) {
      ucp_conn_request_h cr = ctx_->popConnRequest();
      if (!cr) {
        ctx_->progress();
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }

      std::string peer_ip;
      if (!peer_ip_from_conn_request(cr, peer_ip)) {
        logError("UCXServer accept: cannot query peer ip, drop conn_request");
        continue;
      }

      ucp_ep_params_t ep_params;
      std::memset(&ep_params, 0, sizeof(ep_params));
      ep_params.field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST;
      ep_params.conn_request = cr;

      ucp_ep_h ep = nullptr;
      ucs_status_t ust = ucp_ep_create(ctx_->worker(), &ep_params, &ep);
      if (ust != UCS_OK) {
        logError("UCXServer accept: ucp_ep_create failed: %d", (int)ust);
        continue;
      }

      std::unique_ptr<UCXEndpoint> endpoint(
          new UCXEndpoint(ctx_, ep, buffer_));
      endpoint->role = EndpointType::Server;

      UcxRemoteMemInfo local_hdr{};
      std::vector<std::uint8_t> local_rkey;
      if (endpoint->exportLocalMemInfo(local_hdr, local_rkey) != status_t::SUCCESS) {
        logError("UCXServer accept: exportLocalMemInfo failed");
        endpoint->closeEndpoint();
        continue;
      }

      // server 协议：先 recv 再 send（避免双方同时先 send 对撞）
      UcxRemoteMemInfo remote_hdr{};
      size_t got = 0;
      if (endpoint->tagRecv(&remote_hdr, sizeof(remote_hdr), &got,
                            kHandshakeChannel, kSeqHdr) != status_t::SUCCESS ||
          got != sizeof(remote_hdr) || remote_hdr.rkey_len == 0) {
        logError("UCXServer accept: tagRecv(remote_hdr) failed got=%zu rkey_len=%u",
                 got, remote_hdr.rkey_len);
        endpoint->closeEndpoint();
        continue;
      }

      if (endpoint->tagSend(&local_hdr, sizeof(local_hdr),
                            kHandshakeChannel, kSeqHdr) != status_t::SUCCESS) {
        logError("UCXServer accept: tagSend(local_hdr) failed");
        endpoint->closeEndpoint();
        continue;
      }

      std::vector<std::uint8_t> remote_rkey(remote_hdr.rkey_len);
      got = 0;
      if (endpoint->tagRecv(remote_rkey.data(), remote_rkey.size(), &got,
                            kHandshakeChannel, kSeqRkey) != status_t::SUCCESS ||
          got != remote_rkey.size()) {
        logError("UCXServer accept: tagRecv(remote_rkey) failed got=%zu expect=%zu",
                 got, remote_rkey.size());
        endpoint->closeEndpoint();
        continue;
      }

      if (endpoint->tagSend(local_rkey.data(), local_rkey.size(),
                            kHandshakeChannel, kSeqRkey) != status_t::SUCCESS) {
        logError("UCXServer accept: tagSend(local_rkey) failed");
        endpoint->closeEndpoint();
        continue;
      }

      if (endpoint->setRemoteMemInfo(remote_hdr, remote_rkey) != status_t::SUCCESS) {
        logError("UCXServer accept: setRemoteMemInfo failed");
        endpoint->closeEndpoint();
        continue;
      }

      conn_manager->_addEndpoint(peer_ip, std::unique_ptr<Endpoint>(endpoint.release()), ConnType::UCX);

      logInfo("UCXServer established with %s remote_base=0x%llx remote_size=%llu rkey_len=%u",
              peer_ip.c_str(),
              (unsigned long long)remote_hdr.base_addr,
              (unsigned long long)remote_hdr.size,
              remote_hdr.rkey_len);
    }
  });

  return status_t::SUCCESS;
}

status_t UCXServer::stopListen() {
  if (!running_) return status_t::SUCCESS;
  running_ = false;

  accept_running_.store(false, std::memory_order_release);
  if (accept_th_.joinable()) accept_th_.join();

  if (ctx_) ctx_->stopListener();

  logInfo("UCXServer stopped");
  return status_t::SUCCESS;
}

} // namespace hmc
