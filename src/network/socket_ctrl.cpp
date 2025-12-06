#include "net.h"
#include "utils/env.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace hmc {

namespace {
// HELLO: use CTRL_STRUCT + a reserved flag bit.
// Pick a high bit to avoid conflict with user flags.
constexpr uint16_t kHelloFlag = 0x8000;
} // namespace

CtrlSocketManager& CtrlSocketManager::instance() {
  static CtrlSocketManager inst;
  return inst;
}

CtrlSocketManager::CtrlSocketManager() {}

CtrlSocketManager::~CtrlSocketManager() {
  stop();
  closeAll();
}

// ----------------------------- Server -----------------------------

bool CtrlSocketManager::start(const std::string& bind_ip, uint16_t tcp_port,
                              const std::string& uds_path) {
  if (running_) return true;

  bind_ip_ = bind_ip;
  tcp_port_ = tcp_port;
  uds_path_ = uds_path;

  running_ = true;
  is_server_ = true;

  bool ok = true;

  // ----- TCP listener -----
  if (tcp_port_ != 0) {
    tcp_listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (tcp_listen_fd_ < 0) {
      std::cerr << "[CtrlSocketManager] TCP socket() failed: " << strerror(errno) << "\n";
      ok = false;
    } else {
      int opt = 1;
      ::setsockopt(tcp_listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(tcp_port_);
      if (::inet_pton(AF_INET, bind_ip_.c_str(), &addr.sin_addr) <= 0) {
        std::cerr << "[CtrlSocketManager] Invalid bind IP " << bind_ip_ << "\n";
        ::close(tcp_listen_fd_);
        tcp_listen_fd_ = -1;
        ok = false;
      } else if (::bind(tcp_listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[CtrlSocketManager] TCP bind() failed: " << strerror(errno) << "\n";
        ::close(tcp_listen_fd_);
        tcp_listen_fd_ = -1;
        ok = false;
      } else if (::listen(tcp_listen_fd_, 64) < 0) {
        std::cerr << "[CtrlSocketManager] TCP listen() failed: " << strerror(errno) << "\n";
        ::close(tcp_listen_fd_);
        tcp_listen_fd_ = -1;
        ok = false;
      } else {
        tcp_thread_ = std::thread(&CtrlSocketManager::tcpAcceptLoop_, this);
        std::cout << "[CtrlSocketManager] TCP listening on " << bind_ip_ << ":" << tcp_port_ << "\n";
      }
    }
  }

  // ----- UDS listener -----
  if (!uds_path_.empty()) {
    uds_listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (uds_listen_fd_ < 0) {
      std::cerr << "[CtrlSocketManager] UDS socket() failed: " << strerror(errno) << "\n";
      ok = false;
    } else {
      // Remove existing path (ignore errors).
      ::unlink(uds_path_.c_str());

      sockaddr_un addr{};
      addr.sun_family = AF_UNIX;
      // sun_path must be null-terminated, and limited length.
      std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", uds_path_.c_str());

      if (::bind(uds_listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "[CtrlSocketManager] UDS bind() failed: " << strerror(errno) << "\n";
        ::close(uds_listen_fd_);
        uds_listen_fd_ = -1;
        ok = false;
      } else if (::listen(uds_listen_fd_, 64) < 0) {
        std::cerr << "[CtrlSocketManager] UDS listen() failed: " << strerror(errno) << "\n";
        ::close(uds_listen_fd_);
        uds_listen_fd_ = -1;
        ok = false;
      } else {
        uds_thread_ = std::thread(&CtrlSocketManager::udsAcceptLoop_, this);
        std::cout << "[CtrlSocketManager] UDS listening on " << uds_path_ << "\n";
      }
    }
  }

  if (!ok) {
    // If anything failed, stop what we started.
    stop();
    return false;
  }
  return true;
}

void CtrlSocketManager::stop() {
  if (!running_) return;
  running_ = false;

  // Stop TCP listener.
  if (tcp_listen_fd_ >= 0) {
    ::shutdown(tcp_listen_fd_, SHUT_RDWR);
    ::close(tcp_listen_fd_);
    tcp_listen_fd_ = -1;
  }
  if (tcp_thread_.joinable()) tcp_thread_.join();

  // Stop UDS listener.
  if (uds_listen_fd_ >= 0) {
    ::shutdown(uds_listen_fd_, SHUT_RDWR);
    ::close(uds_listen_fd_);
    uds_listen_fd_ = -1;
  }
  if (uds_thread_.joinable()) uds_thread_.join();

  // Best-effort remove UDS path when server stops.
  if (!uds_path_.empty()) {
    ::unlink(uds_path_.c_str());
  }

  is_server_ = false;
  std::cout << "[CtrlSocketManager] Server stopped\n";
}

void CtrlSocketManager::tcpAcceptLoop_() {
  while (running_) {
    sockaddr_in client_addr{};
    socklen_t len = sizeof(client_addr);
    int fd = ::accept(tcp_listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &len);
    if (fd < 0) {
      if (running_) std::cerr << "[CtrlSocketManager] TCP accept() failed: " << strerror(errno) << "\n";
      continue;
    }

    char ip_buf[64]{};
    ::inet_ntop(AF_INET, &client_addr.sin_addr, ip_buf, sizeof(ip_buf));
    std::string hint = std::string(ip_buf) + ":" + std::to_string(ntohs(client_addr.sin_port));

    if (!recvHelloAndBind_(fd, hint)) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      continue;
    }

    std::cout << "[CtrlSocketManager] TCP accepted & registered from " << hint << "\n";
  }
}

void CtrlSocketManager::udsAcceptLoop_() {
  while (running_) {
    sockaddr_un client_addr{};
    socklen_t len = sizeof(client_addr);
    int fd = ::accept(uds_listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &len);
    if (fd < 0) {
      if (running_) std::cerr << "[CtrlSocketManager] UDS accept() failed: " << strerror(errno) << "\n";
      continue;
    }

    // client_addr.sun_path may be empty/undefined for some clients. Use fixed hint.
    std::string hint = "uds";

    if (!recvHelloAndBind_(fd, hint)) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      continue;
    }

    std::cout << "[CtrlSocketManager] UDS accepted & registered\n";
  }
}

// ----------------------------- Client / Dial -----------------------------

std::string CtrlSocketManager::udsPathFor(const std::string& dir, CtrlId peer_id) {
  // Keep it simple & deterministic.
  // Example: /tmp/hmc_ctrl_rank_7.sock
  std::string d = dir;
  if (!d.empty() && d.back() == '/') d.pop_back();
  return d + "/hmc_ctrl_rank_" + std::to_string(peer_id) + ".sock";
}

bool CtrlSocketManager::connectTcp(CtrlId peer_id, const std::string& ip, uint16_t port, CtrlId self_id) {
  int fd = dialTcp_(ip, port);
  if (fd < 0) {
    std::cerr << "[CtrlSocketManager] connectTcp failed to " << ip << ":" << port << "\n";
    return false;
  }

  if (!sendHello_(fd, self_id)) {
    std::cerr << "[CtrlSocketManager] connectTcp sendHello failed\n";
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    return false;
  }

  // Bind immediately by known peer_id for outgoing usage.
  {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = id_to_conn_.find(peer_id);
    if (it != id_to_conn_.end() && it->second.fd >= 0 && it->second.fd != fd) {
      ::shutdown(it->second.fd, SHUT_RDWR);
      ::close(it->second.fd);
    }
    id_to_conn_[peer_id].fd = fd;
  }
  return true;
}

bool CtrlSocketManager::connectUds(CtrlId peer_id, const std::string& uds_path, CtrlId self_id) {
  int fd = dialUds_(uds_path);
  if (fd < 0) {
    std::cerr << "[CtrlSocketManager] connectUds failed to " << uds_path << "\n";
    return false;
  }

  if (!sendHello_(fd, self_id)) {
    std::cerr << "[CtrlSocketManager] connectUds sendHello failed\n";
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    return false;
  }

  {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = id_to_conn_.find(peer_id);
    if (it != id_to_conn_.end() && it->second.fd >= 0 && it->second.fd != fd) {
      ::shutdown(it->second.fd, SHUT_RDWR);
      ::close(it->second.fd);
    }
    id_to_conn_[peer_id].fd = fd;
  }
  return true;
}

// bool CtrlSocketManager::waitPeer(CtrlId peer_id, int timeout_ms) {
//   std::unique_lock<std::mutex> lk(mu_);
//   auto pred = [&]{ 
//     auto it = id_to_conn_.find(peer_id);
//     return it != id_to_conn_.end() && it->second.fd >= 0;
//   };
//   if (timeout_ms <= 0) {
//     cv_.wait(lk, pred);
//     return true;
//   }
//   return cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred);
// }

int CtrlSocketManager::dialTcp_(const std::string& ip, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    ::close(fd);
    return -1;
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

int CtrlSocketManager::dialUds_(const std::string& uds_path) {
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", uds_path.c_str());

  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

// ----------------------------- HELLO -----------------------------

bool CtrlSocketManager::sendHello_(int fd, CtrlId self_id) {
  // HELLO is a CTRL_STRUCT with hello-flag and payload = CtrlId
  CtrlMsgHeader hdr{};
  hdr.type = static_cast<uint16_t>(CtrlMsgType::CTRL_STRUCT);
  hdr.flags = kHelloFlag;
  hdr.length = static_cast<uint32_t>(sizeof(CtrlId));

  if (!sendAll_(fd, &hdr, sizeof(hdr))) return false;
  return sendAll_(fd, &self_id, sizeof(self_id));
}

bool CtrlSocketManager::recvHelloAndBind_(int fd, const std::string& from_hint) {
  CtrlMsgHeader hdr{};
  if (!recvAll_(fd, &hdr, sizeof(hdr))) {
    std::cerr << "[CtrlSocketManager] HELLO recv header failed from " << from_hint << "\n";
    return false;
  }

  if (hdr.type != static_cast<uint16_t>(CtrlMsgType::CTRL_STRUCT) ||
      (hdr.flags & kHelloFlag) == 0 ||
      hdr.length != sizeof(CtrlId)) {
    std::cerr << "[CtrlSocketManager] Invalid HELLO from " << from_hint
              << " (type=" << hdr.type << ", flags=" << hdr.flags
              << ", len=" << hdr.length << ")\n";
    // Drain payload if any (best-effort) to keep stream consistent is not needed since we close.
    return false;
  }

  CtrlId peer_id{};
  if (!recvAll_(fd, &peer_id, sizeof(peer_id))) {
    std::cerr << "[CtrlSocketManager] HELLO recv payload failed from " << from_hint << "\n";
    return false;
  }

  {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = id_to_conn_.find(peer_id);
    if (it != id_to_conn_.end() && it->second.fd >= 0 && it->second.fd != fd) {
      std::cerr << "[CtrlSocketManager] WARNING replace peer=" << peer_id
              << " oldfd=" << it->second.fd << " newfd=" << fd
              << " from=" << from_hint << "\n";
      // Replace old connection for this peer_id.
      ::shutdown(it->second.fd, SHUT_RDWR);
      ::close(it->second.fd);
    }
    id_to_conn_[peer_id].fd = fd;
  }
  // cv_.notify_all();

  return true;
}

// ----------------------------- Message I/O -----------------------------

int CtrlSocketManager::fdOf_(CtrlId peer_id) const {
  std::unique_lock<std::mutex> lk(mu_);
  auto it = id_to_conn_.find(peer_id);
  if (it == id_to_conn_.end()) return -1;
  return it->second.fd;
}

bool CtrlSocketManager::send(CtrlId peer_id, CtrlMsgType type,
                             const void* payload, size_t len, uint16_t flags) {
  int fd = fdOf_(peer_id);
  if (fd < 0) return false;

  CtrlMsgHeader hdr{};
  hdr.type = static_cast<uint16_t>(type);
  hdr.flags = flags;
  hdr.length = static_cast<uint32_t>(len);

  if (!sendAll_(fd, &hdr, sizeof(hdr))) return false;
  if (payload && len > 0 && !sendAll_(fd, payload, len)) return false;
  return true;
}

bool CtrlSocketManager::recv(CtrlId peer_id, CtrlMsgHeader& hdr,
                             std::vector<uint8_t>& payload) {
  int fd = fdOf_(peer_id);
  if (fd < 0) return false;

  if (!recvAll_(fd, &hdr, sizeof(hdr))) return false;

  // Disallow receiving HELLO via normal recv (HELLO should be handled only at accept/bootstrap).
  if (hdr.type == static_cast<uint16_t>(CtrlMsgType::CTRL_STRUCT) && (hdr.flags & kHelloFlag)) {
    std::cerr << "[CtrlSocketManager] Unexpected HELLO on recv(peer_id=" << peer_id << ")\n";
    return false;
  }

  payload.resize(hdr.length);
  if (hdr.length > 0 && !recvAll_(fd, payload.data(), hdr.length)) return false;
  return true;
}

bool CtrlSocketManager::sendInt(CtrlId peer_id, int v) {
  return send(peer_id, CtrlMsgType::CTRL_INT, &v, sizeof(v));
}

bool CtrlSocketManager::recvInt(CtrlId peer_id, int& v) {
  CtrlMsgHeader hdr{};
  std::vector<uint8_t> payload;
  if (!recv(peer_id, hdr, payload)) return false;
  if (hdr.type != static_cast<uint16_t>(CtrlMsgType::CTRL_INT) || payload.size() != sizeof(int))
    return false;
  std::memcpy(&v, payload.data(), sizeof(int));
  return true;
}

bool CtrlSocketManager::sendU64(CtrlId peer_id, uint64_t v) {
  // Keep type CTRL_INT for historical compatibility; discriminate by payload size.
  return send(peer_id, CtrlMsgType::CTRL_INT, &v, sizeof(v));
}

bool CtrlSocketManager::recvU64(CtrlId peer_id, uint64_t& v) {
  CtrlMsgHeader hdr{};
  std::vector<uint8_t> payload;
  if (!recv(peer_id, hdr, payload)) return false;
  if (hdr.type != static_cast<uint16_t>(CtrlMsgType::CTRL_INT) || payload.size() != sizeof(uint64_t))
    return false;
  std::memcpy(&v, payload.data(), sizeof(uint64_t));
  return true;
}

// ----------------------------- Cleanup -----------------------------

void CtrlSocketManager::close(CtrlId peer_id) {
  std::unique_lock<std::mutex> lk(mu_);
  auto it = id_to_conn_.find(peer_id);
  if (it == id_to_conn_.end()) return;

  int fd = it->second.fd;
  if (fd >= 0) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
  }
  id_to_conn_.erase(it);
}

void CtrlSocketManager::closeAll() {
  std::unique_lock<std::mutex> lk(mu_);
  for (auto& kv : id_to_conn_) {
    int fd = kv.second.fd;
    if (fd >= 0) ::close(fd);
  }
  id_to_conn_.clear();
}

// ----------------------------- Raw IO -----------------------------

bool CtrlSocketManager::sendAll_(int fd, const void* buf, size_t len) {
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buf);
  while (len > 0) {
    ssize_t sent = ::send(fd, ptr, len, 0);
    if (sent <= 0) return false;
    ptr += sent;
    len -= static_cast<size_t>(sent);
  }
  return true;
}

bool CtrlSocketManager::recvAll_(int fd, void* buf, size_t len) {
  uint8_t* ptr = reinterpret_cast<uint8_t*>(buf);
  while (len > 0) {
    ssize_t recvd = ::recv(fd, ptr, len, 0);
    if (recvd <= 0) return false;
    ptr += recvd;
    len -= static_cast<size_t>(recvd);
  }
  return true;
}

} // namespace hmc
