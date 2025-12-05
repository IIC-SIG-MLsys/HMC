/**
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */
#include "net_rdma.h"
#include <hmc.h>

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <deque>
#include <mutex>
#include <sys/socket.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {

bool sendAll(int fd, const void *buf, size_t len) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
  while (len > 0) {
    ssize_t n = ::send(fd, p, len, 0);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool recvAll(int fd, void *buf, size_t len) {
  uint8_t *p = reinterpret_cast<uint8_t *>(buf);
  while (len > 0) {
    ssize_t n = ::recv(fd, p, len, 0);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool sendU16(int fd, uint16_t v) { return sendAll(fd, &v, sizeof(v)); }
bool recvU16(int fd, uint16_t &v) { return recvAll(fd, &v, sizeof(v)); }

bool sendMsg(int fd, hmc::CtrlMsgType type, const void *payload, size_t len,
             uint16_t flags = 0) {
  hmc::CtrlMsgHeader hdr{static_cast<uint16_t>(type), flags,
                         static_cast<uint32_t>(len)};
  if (!sendAll(fd, &hdr, sizeof(hdr))) return false;
  if (payload && len > 0 && !sendAll(fd, payload, len)) return false;
  return true;
}

bool recvMsg(int fd, hmc::CtrlMsgHeader &hdr, std::vector<uint8_t> &payload) {
  if (!recvAll(fd, &hdr, sizeof(hdr))) return false;
  payload.resize(hdr.length);
  if (hdr.length > 0 && !recvAll(fd, payload.data(), hdr.length)) return false;
  return true;
}

int dialOnce(const std::string &ip, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    ::close(fd);
    return -1;
  }
  if (::connect(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) <
      0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int acceptOnce(int listen_fd, std::string *out_ip = nullptr) {
  struct sockaddr_in client_addr;
  std::memset(&client_addr, 0, sizeof(client_addr));
  socklen_t len = sizeof(client_addr);
  int fd = ::accept(listen_fd, reinterpret_cast<struct sockaddr *>(&client_addr),
                    &len);
  if (fd < 0) return -1;

  if (out_ip) {
    char ipbuf[64]{};
    ::inet_ntop(AF_INET, &client_addr.sin_addr, ipbuf, sizeof(ipbuf));
    *out_ip = ipbuf;
  }
  return fd;
}

} // namespace

namespace hmc {

namespace {
std::mutex g_meta_mu;
std::unordered_map<uint16_t, std::deque<int>> g_pending_meta_fd;

static void pushMetaFd_(uint16_t peer_data_port, int fd) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  g_pending_meta_fd[peer_data_port].push_back(fd);
}

static int popMetaFd_(uint16_t peer_data_port) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  auto it = g_pending_meta_fd.find(peer_data_port);
  if (it == g_pending_meta_fd.end() || it->second.empty()) return -1;
  int fd = it->second.front();
  it->second.pop_front();
  return fd;
}
} // namespace

RDMAServer::RDMAServer(std::shared_ptr<ConnBuffer> buffer,
                       std::shared_ptr<ConnManager> conn_manager,
                       size_t num_qps)
    : buffer(buffer), Server(conn_manager), num_qps_(num_qps) {}

RDMAServer::~RDMAServer() {}

status_t RDMAServer::listen(std::string ip, uint16_t port) {
  setup_signal_handler();

  int ret = -1;
  struct sockaddr_in sockaddr;
  struct rdma_cm_event *cm_event = nullptr;
  struct sockaddr_in *client_addr = nullptr;
  char recv_ip[INET_ADDRSTRLEN];
  uint16_t recv_port = 0;

  cm_event_channel = rdma_create_event_channel();
  if (!cm_event_channel) {
    logError("Server::Start: Failed to create event channel");
    return status_t::ERROR;
  }

  ret = rdma_create_id(this->cm_event_channel, &this->server_cm_id, nullptr,
                       RDMA_PS_TCP);
  if (ret) {
    logError("Server::Start: Failed to create cm id");
    goto cleanup;
  }

  std::memset(&sockaddr, 0, sizeof(sockaddr));
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_port = htons(port);
  inet_pton(AF_INET, ip.c_str(), &sockaddr.sin_addr);

  ret = rdma_bind_addr(this->server_cm_id,
                       reinterpret_cast<struct sockaddr *>(&sockaddr));
  if (ret) {
    logError("Server::Start: Failed to bind address");
    goto cleanup;
  }

  ret = rdma_listen(this->server_cm_id, 16);
  if (ret) {
    logError("Server::Start: Failed to listen");
    goto cleanup;
  }

  logInfo("Server::Start: Listening on %s:%d (max QPs per conn: %zu)", ip.c_str(),
          port, num_qps_);

  {
    meta_port_ = static_cast<uint16_t>(port + 10000);
    meta_listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (meta_listen_fd_ < 0) {
      logError("Server::meta: socket() failed: %s", strerror(errno));
      goto cleanup;
    }

    int opt = 1;
    ::setsockopt(meta_listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in meta_addr;
    std::memset(&meta_addr, 0, sizeof(meta_addr));
    meta_addr.sin_family = AF_INET;
    meta_addr.sin_port = htons(meta_port_);
    inet_pton(AF_INET, ip.c_str(), &meta_addr.sin_addr);

    if (::bind(meta_listen_fd_,
               reinterpret_cast<struct sockaddr *>(&meta_addr),
               sizeof(meta_addr)) < 0) {
      logError("Server::meta: bind %s:%u failed: %s", ip.c_str(), meta_port_,
               strerror(errno));
      ::close(meta_listen_fd_);
      meta_listen_fd_ = -1;
      goto cleanup;
    }

    if (::listen(meta_listen_fd_, 64) < 0) {
      logError("Server::meta: listen() failed: %s", strerror(errno));
      ::close(meta_listen_fd_);
      meta_listen_fd_ = -1;
      goto cleanup;
    }

    logInfo("Server::meta: Listening on %s:%u", ip.c_str(), meta_port_);
  }

  while (!should_exit()) {
    if (rdma_get_cm_event(cm_event_channel, &cm_event)) continue;

    struct rdma_cm_event event_copy;
    std::memcpy(&event_copy, cm_event, sizeof(*cm_event));
    rdma_ack_cm_event(cm_event);

    if (event_copy.id->route.addr.src_addr.sa_family == AF_INET) {
      client_addr = reinterpret_cast<struct sockaddr_in *>(
          &event_copy.id->route.addr.src_addr);
      recv_port = ntohs(client_addr->sin_port);
      if (!inet_ntop(AF_INET, &(client_addr->sin_addr), recv_ip,
                     INET_ADDRSTRLEN)) {
        logError("Failed to parse client address");
        continue;
      }
    } else {
      logError("Unsupported address family (non-IPv4)");
      continue;
    }

    if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST) {
      logInfo("Recv Connect msg from %s:%d", recv_ip, recv_port);

      std::unique_ptr<hmc::RDMAEndpoint> ep = handleConnection(event_copy.id);
      if (!ep) {
        logError("Server::handleConnection failed for %s", recv_ip);
        continue;
      }

      if (exchangeMetaData(recv_ip, recv_port, ep) != status_t::SUCCESS) {
        logError("Server::exchangeMetaData failed for %s:%u", recv_ip, recv_port);
        continue;
      }

      ep->transitionExtraQPsToRTS();
      conn_manager->_addEndpoint(recv_ip, 0, std::move(ep), ConnType::RDMA);
      logInfo("Server:: connection established with %s:%d", recv_ip, recv_port);

    } else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED) {
      logInfo("Recv Disconnect msg from %s:%d", recv_ip, recv_port);
      rdma_disconnect(event_copy.id);
      conn_manager->_removeEndpoint(recv_ip, 0, ConnType::RDMA);
      logInfo("Disconnect success for %s", recv_ip);
    }
  }

cleanup:
  if (cm_event_channel) {
    rdma_destroy_event_channel(cm_event_channel);
    cm_event_channel = nullptr;
  }
  if (server_cm_id) {
    rdma_destroy_id(server_cm_id);
    server_cm_id = nullptr;
  }
  if (meta_listen_fd_ >= 0) {
    ::shutdown(meta_listen_fd_, SHUT_RDWR);
    ::close(meta_listen_fd_);
    meta_listen_fd_ = -1;
  }

  return status_t::SUCCESS;
}

std::unique_ptr<RDMAEndpoint> RDMAServer::handleConnection(rdma_cm_id *id) {
  struct rdma_cm_event *cm_event = nullptr;

  logDebug("buffer: buffer->ptr %p, buffer->size %zu", buffer->ptr,
           buffer->buffer_size);

  auto endpoint = std::make_unique<RDMAEndpoint>(buffer, num_qps_);
  endpoint->role = EndpointType::Server;
  endpoint->cm_id = id;

  endpoint->completion_channel = ibv_create_comp_channel(endpoint->cm_id->verbs);
  if (!endpoint->completion_channel) {
    logError("Server::setup: Failed to create completion channel");
    goto failed;
  }

  endpoint->cq = ibv_create_cq(endpoint->cm_id->verbs, endpoint->cq_capacity,
                              nullptr, endpoint->completion_channel, 0);
  if (!endpoint->cq) {
    logError("Server::setup: Failed to create completion queue");
    goto failed;
  }

  endpoint->pd = ibv_alloc_pd(endpoint->cm_id->verbs);
  if (!endpoint->pd) {
    logError("Server::setup: Failed to allocate protection domain");
    goto failed;
  }

  if (endpoint->setupQPs() != status_t::SUCCESS) {
    logError("Server::setup: Failed to create QPs");
    goto failed;
  }

  if (endpoint->setupBuffers() != status_t::SUCCESS) {
    logError("Server::setup: Failed to setup buffers");
    goto failed;
  }

  struct rdma_conn_param conn_param;
  std::memset(&conn_param, 0, sizeof(conn_param));
  conn_param.responder_resources = endpoint->responder_resources;
  conn_param.initiator_depth = endpoint->initiator_depth;

  if (rdma_accept(endpoint->cm_id, &conn_param)) {
    logError("Server::accept: Failed to accept connection");
    goto failed;
  }

  if (rdma_get_cm_event(this->cm_event_channel, &cm_event)) {
    logError("Server::accept: Failed to get RDMA_CM_EVENT_ESTABLISHED");
    goto failed;
  }

  if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {
    logError("Server::accept: Unexpected event %s",
             rdma_event_str(cm_event->event));
    rdma_ack_cm_event(cm_event);
    goto failed;
  }

  rdma_ack_cm_event(cm_event);
  endpoint->connStatus = status_t::SUCCESS;
  logInfo("Server:: connection established (QPs: %zu)", num_qps_);
  return endpoint;

failed:
  endpoint.reset();
  return nullptr;
}

status_t RDMAServer::exchangeMetaData(std::string ip,
                                      uint16_t expected_peer_data_port,
                                      std::unique_ptr<RDMAEndpoint> &endpoint) {
  if (meta_listen_fd_ < 0) {
    logError("[RDMAServer] exchangeMetaData: meta_listen_fd_ not ready");
    return status_t::ERROR;
  }

  int fd = popMetaFd_(expected_peer_data_port);

  while (fd < 0) {
    std::string peer_ip;
    int newfd = acceptOnce(meta_listen_fd_, &peer_ip);
    if (newfd < 0) {
      logError("[RDMAServer] exchangeMetaData: acceptOnce failed: %s",
               strerror(errno));
      return status_t::ERROR;
    }

    uint16_t peer_data_port = 0;
    if (!recvU16(newfd, peer_data_port)) {
      ::shutdown(newfd, SHUT_RDWR);
      ::close(newfd);
      logError("[RDMAServer] exchangeMetaData: recv peer_data_port failed");
      continue;
    }

    pushMetaFd_(peer_data_port, newfd);
    fd = popMetaFd_(expected_peer_data_port);
  }

  decltype(endpoint->remote_metadata_attr) client_meta{};
  {
    hmc::CtrlMsgHeader hdr{};
    std::vector<uint8_t> payload;
    if (!recvMsg(fd, hdr, payload) ||
        hdr.type != static_cast<uint16_t>(hmc::CtrlMsgType::CTRL_STRUCT) ||
        payload.size() != sizeof(client_meta)) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      logError("[RDMAServer] exchangeMetaData: recv client metadata failed");
      return status_t::ERROR;
    }
    std::memcpy(&client_meta, payload.data(), sizeof(client_meta));
  }
  endpoint->remote_metadata_attr = client_meta;

  if (!sendMsg(fd, hmc::CtrlMsgType::CTRL_STRUCT,
               &endpoint->local_metadata_attr,
               sizeof(endpoint->local_metadata_attr))) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    logError("[RDMAServer] exchangeMetaData: send local metadata failed");
    return status_t::ERROR;
  }

  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);

  if (endpoint->remote_metadata_attr.address == 0 ||
      endpoint->remote_metadata_attr.length == 0 ||
      endpoint->remote_metadata_attr.key == 0) {
    logError("[RDMAServer] exchangeMetaData: invalid remote metadata");
    return status_t::ERROR;
  }

  logInfo("[RDMAServer] exchangeMetaData: local_qps=%u, remote_qps=%u",
          endpoint->local_metadata_attr.qp_nums,
          endpoint->remote_metadata_attr.qp_nums);

  for (uint32_t i = 0; i < endpoint->local_metadata_attr.qp_nums; ++i)
    logDebug("  local QP[%u] num = %u", i,
             endpoint->local_metadata_attr.qp_num_list[i]);
  for (uint32_t i = 0; i < endpoint->remote_metadata_attr.qp_nums; ++i)
    logDebug("  remote QP[%u] num = %u", i,
             endpoint->remote_metadata_attr.qp_num_list[i]);

  return status_t::SUCCESS;
}

status_t RDMAServer::stopListen() {
  std::raise(SIGINT);
  rdma_cm_id *dummy_id = nullptr;
  if (rdma_create_id(cm_event_channel, &dummy_id, nullptr, RDMA_PS_TCP) == 0) {
    rdma_destroy_id(dummy_id);
  }
  return status_t::SUCCESS;
}

} // namespace hmc
