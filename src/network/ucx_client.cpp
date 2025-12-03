#include "net_ucx.h"

#include "../utils/log.h"

#include <arpa/inet.h>
#include <cstring>
#include <thread>
#include <chrono>

namespace hmc {

// 同 server：保证一致
static std::uint16_t channel_from_ip(const std::string &ip) {
  std::uint32_t h = 2166136261u;
  for (unsigned char c : ip) {
    h ^= c;
    h *= 16777619u;
  }
  std::uint16_t ch = static_cast<std::uint16_t>((h ^ (h >> 16)) & 0xFFFFu);
  return (ch == 0) ? 1 : ch;
}

static bool make_sockaddr_v4(const std::string &ip, uint16_t port,
                             sockaddr_storage &ss, socklen_t &ss_len) {
  std::memset(&ss, 0, sizeof(ss));
  sockaddr_in *sin = reinterpret_cast<sockaddr_in *>(&ss);
  sin->sin_family = AF_INET;
  sin->sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &sin->sin_addr) != 1) {
    return false;
  }
  ss_len = sizeof(sockaddr_in);
  return true;
}

// 静态成员
std::weak_ptr<UCXContext> UCXClient::global_ctx_;

UCXClient::UCXClient(std::shared_ptr<ConnBuffer> buffer)
    : buffer_(std::move(buffer)) {
  auto shared = global_ctx_.lock();
  if (!shared) {
    shared = std::make_shared<UCXContext>();
    global_ctx_ = shared;
  }
  ctx_ = std::move(shared);
}

UCXClient::~UCXClient() = default;

std::unique_ptr<Endpoint> UCXClient::connect(std::string ip, uint16_t port) {
  if (!ctx_) {
    logError("UCXClient::connect: null UCXContext");
    return nullptr;
  }
  if (ctx_->init() != status_t::SUCCESS) {
    logError("UCXClient::connect: UCXContext init failed");
    return nullptr;
  }

  // 1) 创建 client endpoint：sockaddr connect 到 server listener
  sockaddr_storage ss;
  socklen_t ss_len = 0;
  if (!make_sockaddr_v4(ip, port, ss, ss_len)) {
    logError("UCXClient::connect: invalid ip %s", ip.c_str());
    return nullptr;
  }

  ucp_ep_params_t ep_params;
  std::memset(&ep_params, 0, sizeof(ep_params));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS |
                         UCP_EP_PARAM_FIELD_SOCK_ADDR;
  ep_params.flags      = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  ep_params.sockaddr.addr    = reinterpret_cast<const sockaddr *>(&ss);
  ep_params.sockaddr.addrlen = ss_len;

  ucp_ep_h ep = nullptr;
  ucs_status_t ust = ucp_ep_create(ctx_->worker(), &ep_params, &ep);
  if (ust != UCS_OK) {
    logError("UCXClient::connect: ucp_ep_create(connect) failed: %d", ust);
    return nullptr;
  }

  auto endpoint = std::make_unique<UCXEndpoint>(ctx_, ep, buffer_);
  endpoint->role = EndpointType::Client;

  // 2) UCX TAG 握手：交换 remote base/size/rkey（与 server 完全一致）
  const std::uint16_t ch = 1;

  UcxRemoteMemInfo local_hdr{};
  std::vector<std::uint8_t> local_rkey;
  if (endpoint->exportLocalMemInfo(local_hdr, local_rkey) != status_t::SUCCESS) {
    logError("UCXClient::connect: exportLocalMemInfo failed");
    endpoint->closeEndpoint();
    return nullptr;
  }

  // send hdr -> recv hdr
  if (endpoint->tagSend(&local_hdr, sizeof(local_hdr), ch, /*seq=*/1) != status_t::SUCCESS) {
    logError("UCXClient::connect: tagSend(local_hdr) failed");
    endpoint->closeEndpoint();
    return nullptr;
  }

  UcxRemoteMemInfo remote_hdr{};
  size_t got = 0;
  if (endpoint->tagRecv(&remote_hdr, sizeof(remote_hdr), &got, ch, /*seq=*/1) != status_t::SUCCESS) {
    logError("UCXClient::connect: tagRecv(remote_hdr) failed");
    endpoint->closeEndpoint();
    return nullptr;
  }
  if (got != sizeof(remote_hdr)) {
    logError("UCXClient::connect: remote_hdr size mismatch got=%zu expect=%zu",
             got, sizeof(remote_hdr));
    endpoint->closeEndpoint();
    return nullptr;
  }
  if (remote_hdr.rkey_len == 0) {
    logError("UCXClient::connect: remote_hdr.rkey_len == 0");
    endpoint->closeEndpoint();
    return nullptr;
  }

  // send rkey -> recv rkey
  if (endpoint->tagSend(local_rkey.data(), local_rkey.size(), ch, /*seq=*/2) != status_t::SUCCESS) {
    logError("UCXClient::connect: tagSend(local_rkey) failed");
    endpoint->closeEndpoint();
    return nullptr;
  }

  std::vector<std::uint8_t> remote_rkey(remote_hdr.rkey_len);
  got = 0;
  if (endpoint->tagRecv(remote_rkey.data(), remote_rkey.size(), &got, ch, /*seq=*/2) != status_t::SUCCESS) {
    logError("UCXClient::connect: tagRecv(remote_rkey) failed");
    endpoint->closeEndpoint();
    return nullptr;
  }
  if (got != remote_rkey.size()) {
    logError("UCXClient::connect: remote_rkey size mismatch got=%zu expect=%zu",
             got, remote_rkey.size());
    endpoint->closeEndpoint();
    return nullptr;
  }

  if (endpoint->setRemoteMemInfo(remote_hdr, remote_rkey) != status_t::SUCCESS) {
    logError("UCXClient::connect: setRemoteMemInfo failed");
    endpoint->closeEndpoint();
    return nullptr;
  }

  logInfo("UCXClient connected to %s:%u: remote_base=0x%llx remote_size=%llu rkey_len=%u (channel=%u)",
          ip.c_str(), port,
          (unsigned long long)remote_hdr.base_addr,
          (unsigned long long)remote_hdr.size,
          remote_hdr.rkey_len,
          (unsigned)ch);

  return endpoint;
}

} // namespace hmc
