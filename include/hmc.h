/**
 * @file hmc.h
 * @brief Main header file for the HMC (Heterogeneous Memories Communication)
 * framework.
 *
 * This file provides the core APIs for memory management, RDMA/UCX-based
 * communication, and TCP control messaging between heterogeneous devices (CPU,
 * GPU, MLU, NPU, etc.).
 *
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */

#ifndef HMC_H
#define HMC_H

// ===== Platform-specific includes =====
#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
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

// ===== Core includes =====
#include "mem.h"
#include "status.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace hmc {

// Forward declarations
class ConnManager;
class Endpoint;

/**
 * @class ConnBuffer
 * @brief Represents a registered communication buffer used in RDMA/UCX
 * transfers.
 *
 * This class manages a memory region that can be shared across processes or
 * devices. Once registered, the buffer pointer must remain stable to ensure
 * RDMA validity.
 */
class ConnBuffer {
public:
  void *ptr = nullptr;       ///< Pointer to allocated memory
  size_t buffer_size;        ///< Buffer size in bytes
  Memory *mem_ops = nullptr; ///< Associated memory operator

  ConnBuffer(int device_id, size_t buffer_size,
             MemoryType mem_type = MemoryType::DEFAULT);

  status_t writeFromCpu(void *src, size_t size, size_t bias = 0);
  status_t readToCpu(void *dest, size_t size, size_t bias = 0);
  status_t writeFromGpu(void *src, size_t size, size_t bias = 0);
  status_t readToGpu(void *src, size_t size, size_t bias = 0);
  status_t copyWithin(size_t dst_bias, size_t src_bias, size_t size);

  ~ConnBuffer();
};

/**
 * @enum ConnType
 * @brief Communication backend type.
 */
enum class ConnType {
  RDMA, ///< Remote Direct Memory Access
  UCX   ///< Unified Communication X framework
};

/**
 * @enum CtrlMsgType
 * @brief Control message type for TCP synchronization.
 */
enum CtrlMsgType : uint16_t {
  CTRL_INT = 0x01,    ///< Integer message
  CTRL_STRUCT = 0x02, ///< Struct message
};

/**
 * @struct CtrlMsgHeader
 * @brief Header of a control message packet.
 */
struct CtrlMsgHeader {
  uint16_t type;   ///< Message type (see CtrlMsgType)
  uint16_t flags;  ///< Reserved or user-defined flags
  uint32_t length; ///< Payload length in bytes
};
static_assert(sizeof(CtrlMsgHeader) == 8, "Invalid CtrlMsgHeader size");

/**
 * @class CtrlSocketManager
 * @brief TCP-based control message manager for synchronization and signaling.
 *
 * This singleton manages control connections between hosts. It supports sending
 * and receiving small messages such as integers or user-defined structs.
 */
class CtrlSocketManager {
public:
  static CtrlSocketManager &instance();

  static uint16_t port() { return instance().default_port_; };

  CtrlSocketManager(const CtrlSocketManager &) = delete;
  CtrlSocketManager &operator=(const CtrlSocketManager &) = delete;

  bool is_server_{false};
  bool isServer() const { return is_server_; }

  // ----- Server -----
  bool startServer(const std::string &bindIp);
  void stopServer();

  // ----- Client -----
  int getCtrlSockFd(const std::string &ip);

  // ----- Message APIs -----
  bool sendCtrlMsg(const std::string &ip, CtrlMsgType type, const void *payload,
                   size_t len, uint16_t flags = 0);
  bool recvCtrlMsg(const std::string &ip, CtrlMsgHeader &hdr,
                   std::vector<uint8_t> &payload);

  bool sendCtrlInt(const std::string &ip, int value);
  bool recvCtrlInt(const std::string &ip, int &value);
  bool sendCtrlU64(const std::string &ip, uint64_t v);
  bool recvCtrlU64(const std::string &ip, uint64_t &v);

  template <typename T>
  bool sendCtrlStruct(const std::string &ip, const T &obj);

  template <typename T> bool recvCtrlStruct(const std::string &ip, T &obj);

  void closeConnection(const std::string &ip);
  void closeAll();

  ~CtrlSocketManager();

private:
  CtrlSocketManager();
  void acceptLoop();

  int createSocket(const std::string &ip, uint16_t port);
  static bool sendAll(int fd, const void *buf, size_t len);
  static bool recvAll(int fd, void *buf, size_t len);

  std::unordered_map<std::string, int> ip_to_fd_;
  std::mutex mu_;

  int listen_fd_{-1};
  std::thread listener_thread_;
  std::atomic<bool> running_{false};
  uint16_t default_port_ = 5555;
};

// ===== Template Implementations =====

/**
 * @brief Send a trivially copyable struct as a control message.
 */
template <typename T>
bool CtrlSocketManager::sendCtrlStruct(const std::string &ip, const T &obj) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be POD");
  return sendCtrlMsg(ip, CTRL_STRUCT, &obj, sizeof(T));
}

/**
 * @brief Receive a struct message from the control socket.
 */
template <typename T>
bool CtrlSocketManager::recvCtrlStruct(const std::string &ip, T &obj) {
  CtrlMsgHeader hdr;
  std::vector<uint8_t> payload;
  if (!recvCtrlMsg(ip, hdr, payload))
    return false;
  if (hdr.type != CTRL_STRUCT || payload.size() != sizeof(T))
    return false;
  std::memcpy(&obj, payload.data(), sizeof(T));
  return true;
}

/**
 * @class Communicator
 * @brief Unified RDMA/UCX communication interface.
 *
 * Provides high-level APIs for data transmission and synchronization between
 * devices using RDMA or UCX backends.
 */
class Communicator {
private:
  std::shared_ptr<ConnBuffer> buffer;        ///< Registered memory buffer
  std::shared_ptr<ConnManager> conn_manager; ///< Underlying connection manager

  std::mutex inflight_mu_;
  std::unordered_map<uint64_t, Endpoint*> inflight_ep_;

public:
  explicit Communicator(std::shared_ptr<ConnBuffer> buffer, size_t num_chs = 1);

  // --- Core Operations ---
  // single side write/read
  status_t put(std::string ip,
             size_t local_off,
             size_t remote_off,
             size_t size,
             ConnType connType = ConnType::RDMA);

  status_t get(std::string ip,
             size_t local_off,
             size_t remote_off,
             size_t size,
             ConnType connType = ConnType::RDMA);

  status_t putNB(std::string ip, size_t local_off, size_t remote_off, size_t size,
               uint64_t *wr_id, ConnType connType = ConnType::RDMA);
  status_t getNB(std::string ip, size_t local_off, size_t remote_off, size_t size,
                uint64_t *wr_id, ConnType connType = ConnType::RDMA);
  status_t wait(uint64_t wr_id);
  status_t wait(const std::vector<uint64_t>& wr_ids);

  status_t ctrlSend(std::string ip, uint64_t tag);
  status_t ctrlRecv(std::string ip, uint64_t *tag);

  // --- High-level Data APIs (UHM interface, only RDMA) ---
  status_t sendDataTo(std::string ip, void *send_buf, size_t buf_size,
                      MemoryType buf_type, ConnType connType = ConnType::RDMA);

  status_t recvDataFrom(std::string ip, void *recv_buf, size_t buf_size,
                        MemoryType buf_type, size_t *flag,
                        ConnType connType = ConnType::RDMA);

  // --- Connection Management ---
  status_t initServer(std::string ip, uint16_t port, ConnType serverType);
  status_t closeServer();
  status_t connectTo(std::string ip, uint16_t port, ConnType connType);
  status_t disConnect(std::string ip, ConnType connType);
  status_t checkConn(std::string ip, ConnType connType);

  ~Communicator();
};

} // namespace hmc

#endif // HMC_H
