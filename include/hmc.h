/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_H
#define HMC_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
// #include <cuda.h>
#endif
#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif
#ifdef ENABLE_MUSA
#include <musa.h>
#include <musa_runtime.h>
#endif
#ifdef ENABLE_NEUWARE
#include "cn_api.h" // CNresult
#include "cnrt.h"
#include "mlu_op.h"
#endif

#include "mem.h"
#include "status.h"

#include <atomic>
#include <chrono>
#include <csignal> // For signal
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility> // For std::pair
#include <mutex>

namespace hmc {

/* communicator */
class ConnManager;
class Endpoint;

class ConnBuffer {
public:
  void *ptr = nullptr;
  size_t buffer_size;
  Memory *mem_ops = nullptr;

  ConnBuffer(int device_id, size_t buffer_size,
             MemoryType mem_type = MemoryType::DEFAULT);
  // buffer必须在首次分配，并且不允许重新分配，否则指针发生改变，会导致通信的缓冲区失效

  status_t writeFromCpu(void *src, size_t size, size_t bias = 0);
  status_t readToCpu(void *dest, size_t size, size_t bias = 0);
  status_t writeFromGpu(void *src, size_t size, size_t bias = 0);
  status_t readToGpu(void *src, size_t size, size_t bias = 0);

  ~ConnBuffer();
};

enum class ConnType { RDMA, UCX };

/* Socket Control Manager */
enum CtrlMsgType : uint16_t {
  CTRL_INT = 0x01,
  CTRL_STRUCT = 0x02,
};

struct CtrlMsgHeader {
  uint16_t type;
  uint16_t flags;
  uint32_t length; // payload lens
};
static_assert(sizeof(CtrlMsgHeader) == 8);

class CtrlSocketManager {
public:
  static CtrlSocketManager& instance();

  CtrlSocketManager(const CtrlSocketManager&) = delete;
  CtrlSocketManager& operator=(const CtrlSocketManager&) = delete;

  bool is_server_{false};
  bool isServer() const { return is_server_; }

  // --- Server side ---
  bool startServer(const std::string& bindIp, uint16_t port);
  void stopServer();

  // --- Client side ---
  int getCtrlSockFd(const std::string& ip, uint16_t port);

  // --- Message APIs ---
  bool sendCtrlMsg(const std::string& ip, CtrlMsgType type, const void* payload, size_t len, uint16_t flags = 0);
  bool recvCtrlMsg(const std::string& ip, CtrlMsgHeader& hdr, std::vector<uint8_t>& payload);

  bool sendCtrlInt(const std::string& ip, int value);
  bool recvCtrlInt(const std::string& ip, int& value);

  template <typename T>
  bool sendCtrlStruct(const std::string& ip, const T& obj);

  template <typename T>
  bool recvCtrlStruct(const std::string& ip, T& obj);

  void closeConnection(const std::string& ip);
  void closeAll();

  ~CtrlSocketManager();

private:
  CtrlSocketManager();

  void acceptLoop();

  int createSocket(const std::string& ip, uint16_t port);
  static bool sendAll(int fd, const void* buf, size_t len);
  static bool recvAll(int fd, void* buf, size_t len);

private:
  std::unordered_map<std::string, int> ip_to_fd_;
  std::mutex mu_;

  int listen_fd_{-1};
  std::thread listener_thread_;
  std::atomic<bool> running_{false};
  uint16_t default_port_ = 5555;
};

template <typename T>
bool CtrlSocketManager::sendCtrlStruct(const std::string& ip, const T& obj) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be POD");
  return sendCtrlMsg(ip, CTRL_STRUCT, &obj, sizeof(T));
}

template <typename T>
bool CtrlSocketManager::recvCtrlStruct(const std::string& ip, T& obj) {
  CtrlMsgHeader hdr;
  std::vector<uint8_t> payload;
  if (!recvCtrlMsg(ip, hdr, payload)) return false;
  if (hdr.type != CTRL_STRUCT || payload.size() != sizeof(T)) return false;
  std::memcpy(&obj, payload.data(), sizeof(T));
  return true;
}


/* Communicator */
class Communicator {
private:
  std::shared_ptr<ConnBuffer> buffer;
  std::shared_ptr<ConnManager>
      conn_manager; // must be shared, enable shared obj

public:
  Communicator(std::shared_ptr<ConnBuffer> buffer);

  status_t writeTo(std::string ip, size_t ptr_bias, size_t size,
                   ConnType connType = ConnType::RDMA);
  status_t readFrom(std::string ip, size_t ptr_bias, size_t size,
                    ConnType connType = ConnType::RDMA);

  status_t send(std::string ip, size_t ptr_bias, size_t size,
                   ConnType connType = ConnType::RDMA);
  status_t recv(std::string ip, size_t ptr_bias, size_t size,
                    ConnType connType = ConnType::RDMA); // blocked p2p rdma recv 

  status_t sendDataTo(std::string ip, void *send_buf, size_t buf_size, MemoryType buf_type, ConnType connType = ConnType::RDMA); // uhm interface, for big data
  status_t recvDataFrom(std::string ip, void *recv_buf, size_t buf_size, MemoryType buf_type, size_t *flag, ConnType connType = ConnType::RDMA); // flag, recv size

  status_t initServer(std::string ip, uint16_t port, ConnType serverType);
  status_t closeServer();
  status_t connectTo(std::string ip, uint16_t port, ConnType connType);
  status_t disConnect(std::string ip, ConnType connType);
  status_t checkConn(std::string ip, ConnType connType);

  ~Communicator();
};

} // namespace hmc

#endif