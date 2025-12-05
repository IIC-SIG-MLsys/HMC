/**
 * @file net_rdma.h
 * @brief RDMA-based transport implementation for HMC (Heterogeneous Memory Communication)
 *
 * This module provides RDMA connection management, QP setup, and data transfer
 * APIs for high-performance communication between heterogeneous memory devices.
 * 
 * The multi-QP extension allows each endpoint to create multiple queue pairs
 * (QP) per connection to enable parallel data transmission, improving
 * throughput and scalability on multi-threaded systems.
 *
 * @note Each RDMAEndpoint manages its own cm_id, PD, CQ, and QP group.
 *       The default number of QPs is 1, configurable up to MAX_QPS.
 *
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */

#ifndef HMC_NET_RDMA_H
#define HMC_NET_RDMA_H

#include "net.h"

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <vector>
#include <mutex>

#define MAX_QPS 16  ///< Maximum supported number of queue pairs per endpoint

namespace hmc {

/* -------------------------------------------------------------------------- */
/*                            UHM Buffer Structures                           */
/* -------------------------------------------------------------------------- */

#define UHM_STATE_TYPE uint32_t

/**
 * @enum UHMBufferStateType
 * @brief Flags for two-buffer RDMA streaming state synchronization.
 */
typedef enum {
  UHM_BUFFER_CAN_WRITE = 0, ///< Indicates the sender can write to the buffer
  UHM_BUFFER_CAN_READ = 1,  ///< Indicates the receiver can read from the buffer
  UHM_BUFFER_FINISHED = 2   ///< Indicates the data transfer is complete
} UHMBufferStateType;

/**
 * @struct UHMBufferState
 * @brief Shared buffer state structure for user-hosted memory (UHM) protocol.
 */
struct __attribute__((packed)) UHMBufferState {
  volatile UHM_STATE_TYPE state[2]; ///< State for two half-buffers (ping-pong)
  volatile UHM_STATE_TYPE length;   ///< Data length field for current transfer
};

/**
 * @struct rdma_buffer_attr
 * @brief Metadata for RDMA buffer registration and multi-QP exchange.
 */
struct __attribute__((packed)) rdma_buffer_attr {
  uint64_t address;                         ///< Base address of data buffer
  uint32_t length;                          ///< Buffer length in bytes
  uint32_t key;                             ///< Local memory region (MR) key
  uint64_t uhm_buffer_state_address;        ///< Address of buffer state struct
  uint32_t uhm_buffer_state_key;            ///< MR key for state struct
  uint32_t qp_nums;                         ///< Number of queue pairs in use
  uint32_t qp_num_list[MAX_QPS];            ///< List of QP numbers exchanged
  uint8_t  gid[16];       // 对端/本端 GID（RoCEv2）
  uint8_t  sgid_index;    // 本端使用的 sgid index（一般 0 或 1）
};

/* -------------------------------------------------------------------------- */
/*                               RDMA Endpoint                                */
/* -------------------------------------------------------------------------- */

/**
 * @class RDMAEndpoint
 * @brief Represents one RDMA connection endpoint with support for multiple QPs.
 *
 * Each endpoint maintains its own protection domain, completion queue,
 * memory regions, and queue pairs. Multiple QPs allow concurrent operations.
 */
class RDMAEndpoint : public Endpoint {
public:
  RDMAEndpoint(std::shared_ptr<ConnBuffer> buffer, size_t num_qps = 1);
  ~RDMAEndpoint();

  status_t closeEndpoint() override;

  /* QP Management */
  status_t setupQPs();                   ///< Create and initialize multiple QPs
  ibv_qp* getQP(size_t idx);             ///< Select a QP by index (round-robin)
  status_t transitionExtraQPsToRTS();

  /* Data Transfer */
  status_t writeData(size_t local_off, size_t remote_off, size_t size) override;
  status_t readData(size_t local_off, size_t remote_off, size_t size) override;

  status_t writeDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wr_id) override;
  status_t readDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wr_id) override;

  status_t waitWrId(uint64_t wr_id) override;
  status_t waitWrIdMulti(const std::vector<uint64_t>& target_wr_ids,
                                     std::chrono::milliseconds timeout = std::chrono::seconds(5));

  /* User-hosted memory (UHM) send/receive */
  status_t uhm_send(void *input_buffer, const size_t send_flags,
                    MemoryType mem_type) override;
  status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                    size_t *recv_flags, MemoryType mem_type) override;

  /* Memory Management */
  status_t registerMemory(void *addr, size_t length, struct ibv_mr **mr);
  status_t deRegisterMemory(struct ibv_mr *mr);
  status_t setupBuffers();
  void showRdmaBufferAttr(const struct rdma_buffer_attr *attr);
  void cleanRdmaResources();

  /* Low-level RDMA operations */
  status_t postSend(void *addr, size_t length, struct ibv_mr *mr,
                    uint64_t wr_id, bool signaled = true, size_t qp_idx = 0);
  status_t postRecv(void *addr, size_t length, struct ibv_mr *mr,
                    uint64_t wr_id, size_t qp_idx = 0);
  status_t postWrite(void *local_addr, void *remote_addr, size_t length,
                     struct ibv_mr *local_mr, uint32_t remote_key,
                     uint64_t wr_id, bool signaled, size_t qp_idx = 0);
  status_t postRead(void *local_addr, void *remote_addr, size_t length,
                    struct ibv_mr *local_mr, uint32_t remote_key,
                    uint64_t wr_id, bool signaled, size_t qp_idx = 0);

public:
  size_t num_qps_ = 1;                          ///< Number of queue pairs
  std::shared_ptr<ConnBuffer> buffer;           ///< Shared communication buffer
  bool is_buffer_ok = false;
  status_t connStatus = status_t::ERROR;        ///< Connection state

  struct rdma_cm_id *cm_id = NULL;              ///< RDMA connection identifier
  struct ibv_qp_init_attr qp_init_attr;         ///< QP initialization attributes
  std::vector<ibv_qp*> qps_;                    ///< Multiple QPs per endpoint
  std::mutex qp_select_mu_;                     ///< Protect round-robin QP access

  struct ibv_pd *pd = NULL;                     ///< Protection domain
  struct ibv_cq *cq = NULL;                     ///< Completion queue
  struct ibv_mr *remote_metadata_mr = NULL;     ///< Remote metadata MR
  struct ibv_mr *local_metadata_mr = NULL;      ///< Local metadata MR
  struct ibv_mr *buffer_mr = NULL;              ///< Main data buffer MR
  struct ibv_mr *uhm_buffer_state_mr = NULL;    ///< Buffer state MR

  struct rdma_buffer_attr remote_metadata_attr; ///< Remote buffer attributes
  struct rdma_buffer_attr local_metadata_attr;  ///< Local buffer attributes
  struct UHMBufferState uhm_buffer_state;       ///< Shared UHM state object

  struct rdma_event_channel *cm_event_channel = NULL; ///< RDMA event channel
  struct ibv_comp_channel *completion_channel = NULL; ///< Completion channel

  uint8_t initiator_depth = 8;
  uint8_t responder_resources = 8;
  uint8_t retry_count = 3;

  uint16_t cq_capacity = 16;
  uint16_t max_sge = 2;
  uint16_t max_wr = 8;
  // ConnectX-5
  // max_qp_wr  = 32768
  // max_cqe    = 4194303

  uint8_t port_num_      = 1;  // HCA 端口
  uint8_t sgid_index_    = 1;
  uint8_t grh_hop_limit_ = 1;

private:
  std::atomic<uint64_t> next_wr_id_{1};
};

/* -------------------------------------------------------------------------- */
/*                                RDMA Server                                 */
/* -------------------------------------------------------------------------- */

class RDMAServer : public Server {
public:
  RDMAServer(std::shared_ptr<ConnBuffer> buffer,
             std::shared_ptr<ConnManager> conn_manager,
             size_t num_qps = 1);

  ~RDMAServer();

  status_t listen(std::string ip, uint16_t port) override;
  std::unique_ptr<RDMAEndpoint> handleConnection(rdma_cm_id *id);

  status_t stopListen() override;

private:
  std::shared_ptr<ConnBuffer> buffer;
  struct rdma_cm_id *server_cm_id = NULL;
  struct rdma_event_channel *cm_event_channel = NULL;
  size_t num_qps_ = 1;

  int meta_listen_fd_ = -1;
  uint16_t meta_port_ = 0;

  status_t exchangeMetaData(std::string ip, uint16_t port,
                            std::unique_ptr<RDMAEndpoint> &endpoint);
};


/* -------------------------------------------------------------------------- */
/*                                RDMA Client                                 */
/* -------------------------------------------------------------------------- */

class RDMAClient : public Client {
public:
  RDMAClient(std::shared_ptr<ConnBuffer> buffer,
             size_t num_qps = 1,
             int max_retry_times = 10,
             int retry_delay_ms = 1000);
  ~RDMAClient();

  std::unique_ptr<Endpoint> connect(std::string ip, uint16_t port);

private:
  int max_retry_times;
  int retry_delay_ms;
  int retry_count = 0;
  int resolve_addr_timeout_ms = 2000;
  struct sockaddr_in sockaddr;
  std::shared_ptr<ConnBuffer> buffer;
  size_t num_qps_ = 1;

  status_t exchangeMetaData(std::string ip, uint16_t port,
                            std::unique_ptr<RDMAEndpoint> &endpoint);
};

} // namespace hmc

#endif // HMC_NET_RDMA_H
