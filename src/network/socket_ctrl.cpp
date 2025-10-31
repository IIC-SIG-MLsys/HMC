#include "net.h"
#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

namespace hmc {

CtrlSocketManager& CtrlSocketManager::instance() {
  static CtrlSocketManager inst;
  return inst;
}

CtrlSocketManager::CtrlSocketManager() = default;

CtrlSocketManager::~CtrlSocketManager() {
  stopServer();
  closeAll();
}

// ================= Server Side =================

bool CtrlSocketManager::startServer(const std::string& bindIp, uint16_t port) {
  if (running_) return true; // already running

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    std::cerr << "[CtrlSocketManager] socket() failed\n";
    return false;
  }

  int opt = 1;
  setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, bindIp.c_str(), &addr.sin_addr) <= 0) {
    std::cerr << "[CtrlSocketManager] Invalid bind IP " << bindIp << "\n";
    ::close(listen_fd_);
    return false;
  }

  // 尝试 bind
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    if (errno == EADDRINUSE) {
      std::cerr << "[CtrlSocketManager] Port " << port << " already in use, switching to CLIENT mode.\n";
      ::close(listen_fd_);
      listen_fd_ = -1;
      is_server_ = false;
      return false;  // 不报错，只表示自己不是 server
    } else {
      std::cerr << "[CtrlSocketManager] bind() failed: " << strerror(errno) << "\n";
      ::close(listen_fd_);
      return false;
    }
  }

  if (::listen(listen_fd_, 64) < 0) {
    std::cerr << "[CtrlSocketManager] listen() failed\n";
    ::close(listen_fd_);
    return false;
  }

  running_ = true;
  is_server_ = true;
  listener_thread_ = std::thread(&CtrlSocketManager::acceptLoop, this);
  std::cout << "[CtrlSocketManager] Listening on " << bindIp << ":" << port << "\n";
  return true;
}


void CtrlSocketManager::stopServer() {
  if (!running_) return;
  running_ = false;
  ::shutdown(listen_fd_, SHUT_RDWR);
  ::close(listen_fd_);
  if (listener_thread_.joinable()) listener_thread_.join();
  std::cout << "[CtrlSocketManager] Server stopped\n";
}

void CtrlSocketManager::acceptLoop() {
  while (running_) {
    sockaddr_in client_addr{};
    socklen_t len = sizeof(client_addr);
    int client_fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&client_addr), &len);
    if (client_fd < 0) {
      if (running_) std::cerr << "[CtrlSocketManager] accept() failed\n";
      continue;
    }

    char ip_buf[64];
    inet_ntop(AF_INET, &client_addr.sin_addr, ip_buf, sizeof(ip_buf));
    std::string client_ip = ip_buf;

    {
      std::unique_lock<std::mutex> lk(mu_);
      ip_to_fd_[client_ip] = client_fd;
    }

    std::cout << "[CtrlSocketManager] Accepted connection from " << client_ip << "\n";
  }
}

// ================= Client Side =================

int CtrlSocketManager::getCtrlSockFd(const std::string& ip, uint16_t port) {
  {
    std::unique_lock<std::mutex> lk(mu_);
    auto it = ip_to_fd_.find(ip);
    if (it != ip_to_fd_.end())
        return it->second;
  }

  std::unique_lock<std::mutex> lk(mu_);
  int fd = createSocket(ip, port);
  if (fd < 0) {
    std::cerr << "[CtrlSocketManager] Failed to connect to " << ip << ":" << port << std::endl;
    return -1;
  }
  ip_to_fd_[ip] = fd;
  return fd;
}

int CtrlSocketManager::createSocket(const std::string& ip, uint16_t port) {
  int sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    ::close(sockfd);
    return -1;
  }

  if (::connect(sockfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(sockfd);
    return -1;
  }
  return sockfd;
}

// ================= Message I/O =================

bool CtrlSocketManager::sendCtrlMsg(const std::string& ip, CtrlMsgType type, const void* payload, size_t len, uint16_t flags) {
  int fd = getCtrlSockFd(ip, default_port_);
  if (fd < 0) return false;

  CtrlMsgHeader hdr{static_cast<uint16_t>(type), flags, static_cast<uint32_t>(len)};
  if (!sendAll(fd, &hdr, sizeof(hdr))) return false;
  if (payload && len > 0 && !sendAll(fd, payload, len)) return false;
  return true;
}

bool CtrlSocketManager::recvCtrlMsg(const std::string& ip, CtrlMsgHeader& hdr, std::vector<uint8_t>& payload) {
  int fd = getCtrlSockFd(ip, default_port_);
  if (fd < 0) return false;

  if (!recvAll(fd, &hdr, sizeof(hdr))) return false;
  payload.resize(hdr.length);
  if (hdr.length > 0 && !recvAll(fd, payload.data(), hdr.length)) return false;
  return true;
}

bool CtrlSocketManager::sendCtrlInt(const std::string& ip, int value) {
  return sendCtrlMsg(ip, CTRL_INT, &value, sizeof(value));
}

bool CtrlSocketManager::recvCtrlInt(const std::string& ip, int& value) {
  CtrlMsgHeader hdr;
  std::vector<uint8_t> payload;
  if (!recvCtrlMsg(ip, hdr, payload)) return false;
  if (hdr.type != CTRL_INT || payload.size() != sizeof(int)) return false;
  std::memcpy(&value, payload.data(), sizeof(int));
  return true;
}

// ================= Cleanup =================

void CtrlSocketManager::closeConnection(const std::string& ip) {
  std::unique_lock<std::mutex> lock(mu_);
  auto it = ip_to_fd_.find(ip);
  if (it != ip_to_fd_.end()) {
    int fd = it->second;
    if (fd >= 0) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
    }
    ip_to_fd_.erase(it);
    std::cout << "[CtrlSocketManager] Closed control connection for " << ip << std::endl;
  } else {
    std::cerr << "[CtrlSocketManager] closeConnection: no entry for " << ip << std::endl;
  }
}

void CtrlSocketManager::closeAll() {
  std::unique_lock<std::mutex> lock(mu_);
  for (std::unordered_map<std::string, int>::iterator it = ip_to_fd_.begin();
       it != ip_to_fd_.end(); ++it) {
    int fd = it->second;
    ::close(fd);
  }
  ip_to_fd_.clear();
}

// ================= Utility =================

bool CtrlSocketManager::sendAll(int fd, const void* buf, size_t len) {
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buf);
  while (len > 0) {
    ssize_t sent = ::send(fd, ptr, len, 0);
    if (sent <= 0) return false;
    ptr += sent;
    len -= sent;
  }
  return true;
}

bool CtrlSocketManager::recvAll(int fd, void* buf, size_t len) {
  uint8_t* ptr = reinterpret_cast<uint8_t*>(buf);
  while (len > 0) {
    ssize_t recvd = ::recv(fd, ptr, len, 0);
    if (recvd <= 0) return false;
    ptr += recvd;
    len -= recvd;
  }
  return true;
}

} // namespace hmc
