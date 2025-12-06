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
#include <endian.h>

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

// cm_id -> (ip,port,conn_id)
struct BoundKey {
  std::string ip;
  uint16_t port = 0;
  uint64_t conn_id = 0;
};

std::mutex g_bind_mu;
std::unordered_map<rdma_cm_id*, BoundKey> g_bound_by_cmid;

static void bindCmid_(rdma_cm_id* id, const std::string& ip, uint16_t port, uint64_t conn_id) {
  std::lock_guard<std::mutex> lk(g_bind_mu);
  g_bound_by_cmid[id] = BoundKey{ip, port, conn_id};
}

static bool takeBind_(rdma_cm_id* id, BoundKey* out) {
  std::lock_guard<std::mutex> lk(g_bind_mu);
  auto it = g_bound_by_cmid.find(id);
  if (it == g_bound_by_cmid.end()) return false;
  if (out) *out = it->second;
  g_bound_by_cmid.erase(it);
  return true;
}
} // namespace

namespace hmc {

uint16_t g_rdma_listen_port = 0;

namespace {
std::mutex g_meta_mu;
std::unordered_map<uint64_t, std::deque<int>> g_pending_meta_fd;
std::unordered_map<uint64_t, std::string> g_peer_ip_by_conn;
std::unordered_map<uint64_t, uint16_t> g_peer_port_by_conn;

static void pushMetaFd_(uint64_t conn_id, int fd,
                        const std::string &peer_ip,
                        uint16_t client_rdma_port) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  g_pending_meta_fd[conn_id].push_back(fd);
  g_peer_ip_by_conn[conn_id] = peer_ip;
  g_peer_port_by_conn[conn_id] = client_rdma_port;
}

static int popMetaFd_(uint64_t conn_id,
                      std::string *out_peer_ip,
                      uint16_t *out_client_port) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  auto it = g_pending_meta_fd.find(conn_id);
  if (it == g_pending_meta_fd.end() || it->second.empty()) return -1;
  int fd = it->second.front();
  it->second.pop_front();
  if (out_peer_ip) *out_peer_ip = g_peer_ip_by_conn[conn_id];
  if (out_client_port) *out_client_port = g_peer_port_by_conn[conn_id];
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

  g_rdma_listen_port = port;
  meta_port_ = static_cast<uint16_t>(port + 10000);

  int opt = 1;
  struct sockaddr_in meta_addr;
  std::memset(&meta_addr, 0, sizeof(meta_addr));

  cm_event_channel = rdma_create_event_channel();
  if (!cm_event_channel) {
    logError("Server::listen: Failed to create event channel");
    return status_t::ERROR;
  }

  ret = rdma_create_id(this->cm_event_channel, &this->server_cm_id, nullptr,
                       RDMA_PS_TCP);
  if (ret) {
    logError("Server::listen: Failed to create cm id");
    goto cleanup;
  }

  std::memset(&sockaddr, 0, sizeof(sockaddr));
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_port = htons(port);
  inet_pton(AF_INET, ip.c_str(), &sockaddr.sin_addr);

  ret = rdma_bind_addr(this->server_cm_id,
                       reinterpret_cast<struct sockaddr *>(&sockaddr));
  if (ret) {
    logError("Server::listen: Failed to bind address");
    goto cleanup;
  }

  ret = rdma_listen(this->server_cm_id, 16);
  if (ret) {
    logError("Server::listen: Failed to rdma_listen");
    goto cleanup;
  }

  logInfo("Server::listen: Listening on %s:%u (max QPs per conn: %zu)",
          ip.c_str(), port, num_qps_);

  // meta listen socket
  meta_listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (meta_listen_fd_ < 0) {
    logError("Server::meta: socket() failed: %s", strerror(errno));
    goto cleanup;
  }

  ::setsockopt(meta_listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  meta_addr.sin_family = AF_INET;
  meta_addr.sin_port = htons(meta_port_);
  inet_pton(AF_INET, ip.c_str(), &meta_addr.sin_addr);

  if (::bind(meta_listen_fd_,
             reinterpret_cast<struct sockaddr *>(&meta_addr),
             sizeof(meta_addr)) < 0) {
    logError("Server::meta: bind %s:%u failed: %s", ip.c_str(), meta_port_,
             strerror(errno));
    goto cleanup;
  }

  if (::listen(meta_listen_fd_, 64) < 0) {
    logError("Server::meta: listen() failed: %s", strerror(errno));
    goto cleanup;
  }

  logInfo("Server::meta: Listening on %s:%u", ip.c_str(), meta_port_);

  // main loop
  while (!should_exit()) {
    if (rdma_get_cm_event(cm_event_channel, &cm_event)) continue;

    struct rdma_cm_event event_copy;
    std::memcpy(&event_copy, cm_event, sizeof(*cm_event));
    rdma_ack_cm_event(cm_event);

    if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST) {
      uint64_t conn_id = 0;
      if (event_copy.param.conn.private_data &&
          event_copy.param.conn.private_data_len >= sizeof(uint64_t)) {
        uint64_t be = 0;
        std::memcpy(&be, event_copy.param.conn.private_data, sizeof(be));
        conn_id = be64toh(be);
      }
      if (conn_id == 0) {
        logError("Server::CONNECT_REQUEST: missing/invalid conn_id in private_data");
        rdma_reject(event_copy.id, nullptr, 0);
        continue;
      }

      logInfo("Server::CONNECT_REQUEST: conn_id=%lu", conn_id);

      std::unique_ptr<hmc::RDMAEndpoint> ep = handleConnection(event_copy.id, conn_id);
      if (!ep) {
        logError("Server::handleConnection failed (conn_id=%lu)", conn_id);
        continue;
      }

      std::string peer_ip;
      uint16_t client_rdma_port = 0;
      uint16_t server_rdma_port = 0;

      if (exchangeMetaData(conn_id, ep, &peer_ip, &client_rdma_port, &server_rdma_port) != status_t::SUCCESS) {
        logError("Server::exchangeMetaData failed (conn_id=%lu)", conn_id);
        continue;
      }

      ep->transitionExtraQPsToRTS();
      bindCmid_(event_copy.id, peer_ip, client_rdma_port, conn_id);

      conn_manager->_removeEndpoint(peer_ip, client_rdma_port, ConnType::RDMA); // replace
      conn_manager->_addEndpoint(peer_ip, client_rdma_port, std::move(ep), ConnType::RDMA);

      logInfo("Server:: established: peer=%s client_rdma_port=%u server_rdma_port=%u conn_id=%lu",
              peer_ip.c_str(), client_rdma_port, server_rdma_port, conn_id);
    } else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED) {
      logInfo("Server::DISCONNECTED cm_id=%p", (void*)event_copy.id);
      BoundKey key;
      if (!takeBind_(event_copy.id, &key)) {
        logError("Server::DISCONNECTED: cm_id not bound, ignore remove");
        continue;
      }
      conn_manager->_removeEndpointIfMatch(key.ip, key.port, ConnType::RDMA, key.conn_id);
    }
  }

cleanup:
  if (meta_listen_fd_ >= 0) {
    ::shutdown(meta_listen_fd_, SHUT_RDWR);
    ::close(meta_listen_fd_);
    meta_listen_fd_ = -1;
  }
  if (server_cm_id) {
    rdma_destroy_id(server_cm_id);
    server_cm_id = nullptr;
  }
  if (cm_event_channel) {
    rdma_destroy_event_channel(cm_event_channel);
    cm_event_channel = nullptr;
  }
  return status_t::SUCCESS;
}

std::unique_ptr<RDMAEndpoint> RDMAServer::handleConnection(rdma_cm_id *id,
                                                          uint64_t conn_id) {
  struct rdma_cm_event *cm_event = nullptr;

  auto endpoint = std::make_unique<RDMAEndpoint>(buffer, num_qps_);
  endpoint->role = EndpointType::Server;
  endpoint->cm_id = id;
  endpoint->conn_id_ = conn_id;

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
  logInfo("Server:: connection established (QPs: %zu, conn_id=%lu)", num_qps_, conn_id);
  return endpoint;

failed:
  endpoint.reset();
  return nullptr;
}

status_t RDMAServer::exchangeMetaData(uint64_t expected_conn_id,
                                      std::unique_ptr<RDMAEndpoint> &endpoint,
                                      std::string *out_peer_ip,
                                      uint16_t *out_client_rdma_port,
                                      uint16_t *out_server_rdma_port) {
  if (meta_listen_fd_ < 0) {
    logError("[RDMAServer] exchangeMetaData: meta_listen_fd_ not ready");
    return status_t::ERROR;
  }

  std::string peer_ip;
  uint16_t client_rdma_port = 0;
  int fd = popMetaFd_(expected_conn_id, &peer_ip, &client_rdma_port);

  while (fd < 0) {
    std::string ip;
    int newfd = acceptOnce(meta_listen_fd_, &ip);
    if (newfd < 0) {
      logError("[RDMAServer] exchangeMetaData: acceptOnce failed: %s",
               strerror(errno));
      return status_t::ERROR;
    }

    MetaHello hello{};
    if (!recvAll(newfd, &hello, sizeof(hello))) {
      ::shutdown(newfd, SHUT_RDWR);
      ::close(newfd);
      logError("[RDMAServer] exchangeMetaData: recv MetaHello failed");
      continue;
    }

    const uint64_t conn_id = be64toh(hello.conn_id_be);
    const uint16_t cport = ntohs(hello.client_port_be);

    pushMetaFd_(conn_id, newfd, ip, cport);

    fd = popMetaFd_(expected_conn_id, &peer_ip, &client_rdma_port);
  }

  // recv client metadata struct
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

  // send MetaReply(conn_id + server_rdma_port)
  MetaReply reply{};
  reply.conn_id_be = htobe64(expected_conn_id);
  reply.server_port_be = htons(g_rdma_listen_port);
  if (!sendAll(fd, &reply, sizeof(reply))) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    logError("[RDMAServer] exchangeMetaData: send MetaReply failed");
    return status_t::ERROR;
  }

  // send server metadata struct
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

  if (out_peer_ip) *out_peer_ip = peer_ip;
  if (out_client_rdma_port) *out_client_rdma_port = client_rdma_port;
  if (out_server_rdma_port) *out_server_rdma_port = g_rdma_listen_port;

  logInfo("[RDMAServer] exchangeMetaData: conn_id=%lu peer=%s client_rdma_port=%u server_rdma_port=%u local_qps=%u remote_qps=%u",
          expected_conn_id, peer_ip.c_str(), client_rdma_port, g_rdma_listen_port,
          endpoint->local_metadata_attr.qp_nums,
          endpoint->remote_metadata_attr.qp_nums);

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
