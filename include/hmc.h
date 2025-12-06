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
 * @brief Control message manager (TCP + UDS). Peer is identified by CtrlId (e.g. rank).
 *
 * - Same-host: use UDS.
 * - Cross-host: use TCP.
 * - Server learns peer CtrlId via a HELLO message on each new connection.
 *
 * Thread-safety:
 *   - connect/send/recv/close are thread-safe.
 */
class CtrlSocketManager {
public:
  using CtrlId = uint32_t; // rank

  static CtrlSocketManager& instance();

  CtrlSocketManager(const CtrlSocketManager&) = delete;
  CtrlSocketManager& operator=(const CtrlSocketManager&) = delete;

  // -------- Server --------
  // Start listeners. Pass empty uds_path to disable UDS.
  bool start(const std::string& bind_ip, uint16_t tcp_port,
             const std::string& uds_path = "");
  void stop();

  bool isServer() const { return is_server_; }

  // -------- Client/Dial --------
  // Connect to peer via TCP or UDS, and register self_id to peer (HELLO).
  bool connectTcp(CtrlId peer_id, const std::string& ip, uint16_t port, CtrlId self_id);
  bool connectUds(CtrlId peer_id, const std::string& uds_path, CtrlId self_id);

  // Helper: generate per-rank uds path (recommended for multi-proc same-host).
  static std::string udsPathFor(const std::string& dir, CtrlId peer_id);

  // -------- Message I/O (by CtrlId) --------
  bool send(CtrlId peer_id, CtrlMsgType type, const void* payload, size_t len, uint16_t flags = 0);
  bool recv(CtrlId peer_id, CtrlMsgHeader& hdr, std::vector<uint8_t>& payload);

  bool sendInt(CtrlId peer_id, int v);
  bool recvInt(CtrlId peer_id, int& v);

  bool sendU64(CtrlId peer_id, uint64_t v);
  bool recvU64(CtrlId peer_id, uint64_t& v);

  template <typename T>
  bool sendStruct(CtrlId peer_id, const T& obj);

  template <typename T>
  bool recvStruct(CtrlId peer_id, T& obj);

  // -------- Cleanup --------
  void close(CtrlId peer_id);
  void closeAll();

  ~CtrlSocketManager();

private:
  CtrlSocketManager();

  // accept threads
  void tcpAcceptLoop_();
  void udsAcceptLoop_();

  // connection bootstrap: fd must first exchange HELLO(self_id) and register peer_id
  bool sendHello_(int fd, CtrlId self_id);
  bool recvHelloAndBind_(int fd, const std::string& from_hint);

  // dialing
  int dialTcp_(const std::string& ip, uint16_t port);
  int dialUds_(const std::string& uds_path);

  // raw io
  static bool sendAll_(int fd, const void* buf, size_t len);
  static bool recvAll_(int fd, void* buf, size_t len);

  int fdOf_(CtrlId peer_id) const;

private:
  struct Conn {
    int fd{-1};
  };

  bool is_server_{false};

  mutable std::mutex mu_;
  std::unordered_map<CtrlId, Conn> id_to_conn_;

  // server state
  std::atomic<bool> running_{false};

  // TCP listener
  int tcp_listen_fd_{-1};
  std::thread tcp_thread_;
  std::string bind_ip_;
  uint16_t tcp_port_{0};

  // UDS listener
  int uds_listen_fd_{-1};
  std::thread uds_thread_;
  std::string uds_path_;

  // for tmp socket fd
  std::unordered_map<std::string, std::deque<int>> pending_by_ip_;

  // mutable std::condition_variable cv_;
};

// Templates
template <typename T>
bool CtrlSocketManager::sendStruct(CtrlId peer_id, const T& obj) {
  static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
  // Replace CTRL_STRUCT with your real enum value.
  return send(peer_id, static_cast<CtrlMsgType>(CtrlMsgType::CTRL_STRUCT), &obj, sizeof(T));
}

template <typename T>
bool CtrlSocketManager::recvStruct(CtrlId peer_id, T& obj) {
  CtrlMsgHeader hdr;
  std::vector<uint8_t> payload;
  if (!recv(peer_id, hdr, payload)) return false;

  if (hdr.type != static_cast<uint16_t>(CtrlMsgType::CTRL_STRUCT) || payload.size() != sizeof(T))
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
  using CtrlId = hmc::CtrlSocketManager::CtrlId; // rank
  enum class CtrlTransport : uint8_t { TCP = 0, UDS = 1 };
  struct CtrlLink {
    CtrlTransport transport{CtrlTransport::TCP};
    std::string ip;
    uint16_t port{0};
    std::string uds_path;
  };

  explicit Communicator(std::shared_ptr<ConnBuffer> buffer, size_t num_chs = 1);

  // --- Core Operations ---
  // single side write/read
  status_t put(std::string ip,
             uint16_t port,
             size_t local_off,
             size_t remote_off,
             size_t size,
             ConnType connType = ConnType::RDMA);

  status_t get(std::string ip,
             uint16_t port,
             size_t local_off,
             size_t remote_off,
             size_t size,
             ConnType connType = ConnType::RDMA);

  status_t putNB(std::string ip,
                 uint16_t port,
                 size_t local_off,
                 size_t remote_off,
                 size_t size,
                 uint64_t* wr_id,
                 ConnType connType = ConnType::RDMA);

  status_t getNB(std::string ip,
                 uint16_t port,
                 size_t local_off,
                 size_t remote_off,
                 size_t size,
                 uint64_t* wr_id,
                 ConnType connType = ConnType::RDMA);

  status_t wait(uint64_t wr_id);
  status_t wait(const std::vector<uint64_t>& wr_ids);

  // --- High-level Data APIs (UHM interface, only RDMA, only p2p with different IP) ---
  status_t sendDataTo(std::string ip, uint16_t port, void *send_buf, size_t buf_size,
                      MemoryType buf_type, ConnType connType = ConnType::RDMA);

  status_t recvDataFrom(std::string ip, void *recv_buf, size_t buf_size,
                        MemoryType buf_type, size_t *flag,
                        ConnType connType = ConnType::RDMA);

  status_t ctrlSend(CtrlId peer, uint64_t tag);
  status_t ctrlRecv(CtrlId peer, uint64_t* tag);

  // --- Connection Management ---
  status_t initCtrlServer(const std::string& bind_ip, uint16_t tcp_port,
                          const std::string& uds_path = "");
  status_t closeCtrl();
  status_t connectCtrl(CtrlId peer_id, CtrlId self_id, const CtrlLink& link);
  status_t closeCtrlPeer(CtrlId peer_id);
  static std::string udsPathFor(const std::string& dir, CtrlId peer_id);

  status_t initServer(const std::string& bind_ip,
                      uint16_t data_port,
                      uint16_t ctrl_tcp_port,
                      const std::string& ctrl_uds_path,
                      ConnType serverType);
  status_t closeServer();
  status_t connectTo(CtrlId peer_id,
                     CtrlId self_id,
                     const std::string& peer_ip,
                     uint16_t data_port,
                     const CtrlLink& ctrl_link,
                     ConnType connType);
  status_t disConnect(const std::string& ip, uint16_t port, ConnType connType);
  status_t checkConn(const std::string& ip, uint16_t port, ConnType connType);

  ~Communicator();
};

} // namespace hmc

#endif // HMC_H
