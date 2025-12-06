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
#include <atomic>
#include <cstdint>

#define MAX_QPS 16

namespace hmc {

/**
 * @brief meta 通道用的握手包：用 conn_id 做并发连接匹配。
 * 所有字段按网络字节序传输（big endian）。
 */
struct __attribute__((packed)) MetaHello {
  uint64_t conn_id_be;        // htobe64(conn_id)
  uint16_t client_port_be;    // htons(client_rdma_port)
};

struct __attribute__((packed)) MetaReply {
  uint64_t conn_id_be;        // htobe64(conn_id) for validation
  uint16_t server_port_be;    // htons(server_rdma_port)
};

#define UHM_STATE_TYPE uint32_t
typedef enum {
  UHM_BUFFER_CAN_WRITE = 0,
  UHM_BUFFER_CAN_READ = 1,
  UHM_BUFFER_FINISHED = 2
} UHMBufferStateType;

struct __attribute__((packed)) UHMBufferState {
  volatile UHM_STATE_TYPE state[2];
  volatile UHM_STATE_TYPE length;
};

struct __attribute__((packed)) rdma_buffer_attr {
  uint64_t address;
  uint32_t length;
  uint32_t key;
  uint64_t uhm_buffer_state_address;
  uint32_t uhm_buffer_state_key;
  uint32_t qp_nums;
  uint32_t qp_num_list[MAX_QPS];
  uint8_t  gid[16];
  uint8_t  sgid_index;
};

class RDMAEndpoint : public Endpoint {
public:
  RDMAEndpoint(std::shared_ptr<ConnBuffer> buffer, size_t num_qps = 1);
  ~RDMAEndpoint();

  status_t closeEndpoint() override;

  status_t setupQPs();
  ibv_qp* getQP(size_t idx);
  status_t transitionExtraQPsToRTS();

  status_t writeData(size_t local_off, size_t remote_off, size_t size) override;
  status_t readData(size_t local_off, size_t remote_off, size_t size) override;

  status_t writeDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wr_id) override;
  status_t readDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wr_id) override;

  status_t waitWrId(uint64_t wr_id) override;
  status_t waitWrIdMulti(const std::vector<uint64_t>& target_wr_ids,
                         std::chrono::milliseconds timeout = std::chrono::seconds(5));

  status_t uhm_send(void *input_buffer, const size_t send_flags,
                    MemoryType mem_type) override;
  status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                    size_t *recv_flags, MemoryType mem_type) override;

  status_t registerMemory(void *addr, size_t length, struct ibv_mr **mr);
  status_t deRegisterMemory(struct ibv_mr *mr);
  status_t setupBuffers();
  void showRdmaBufferAttr(const struct rdma_buffer_attr *attr);
  void cleanRdmaResources();

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
  size_t num_qps_ = 1;
  std::shared_ptr<ConnBuffer> buffer;
  bool is_buffer_ok = false;
  status_t connStatus = status_t::ERROR;

  struct rdma_cm_id *cm_id = NULL;
  struct ibv_qp_init_attr qp_init_attr;
  std::vector<ibv_qp*> qps_;
  std::mutex qp_select_mu_;

  struct ibv_pd *pd = NULL;
  struct ibv_cq *cq = NULL;
  struct ibv_mr *remote_metadata_mr = NULL;
  struct ibv_mr *local_metadata_mr = NULL;
  struct ibv_mr *buffer_mr = NULL;
  struct ibv_mr *uhm_buffer_state_mr = NULL;

  struct rdma_buffer_attr remote_metadata_attr;
  struct rdma_buffer_attr local_metadata_attr;
  struct UHMBufferState uhm_buffer_state;

  struct rdma_event_channel *cm_event_channel = NULL;
  struct ibv_comp_channel *completion_channel = NULL;

  uint8_t initiator_depth = 8;
  uint8_t responder_resources = 8;
  uint8_t retry_count = 3;

  uint16_t cq_capacity = 16;
  uint16_t max_sge = 2;
  uint16_t max_wr = 8;

  uint8_t port_num_      = 1;
  uint8_t sgid_index_    = 1;
  uint8_t grh_hop_limit_ = 1;

  uint64_t conn_id_ = 0;

private:
  std::atomic<uint64_t> next_wr_id_{1};
};

/* -------------------------------- Server -------------------------------- */
extern uint16_t g_rdma_listen_port;

class RDMAServer : public Server {
public:
  RDMAServer(std::shared_ptr<ConnBuffer> buffer,
             std::shared_ptr<ConnManager> conn_manager,
             size_t num_qps = 1);
  ~RDMAServer();

  status_t listen(std::string ip, uint16_t port) override;
  std::unique_ptr<RDMAEndpoint> handleConnection(rdma_cm_id *id,
                                                 uint64_t conn_id);
  status_t stopListen() override;

private:
  std::shared_ptr<ConnBuffer> buffer;
  struct rdma_cm_id *server_cm_id = NULL;
  struct rdma_event_channel *cm_event_channel = NULL;
  size_t num_qps_ = 1;

  int meta_listen_fd_ = -1;
  uint16_t meta_port_ = 0;

  status_t exchangeMetaData(uint64_t expected_conn_id,
                            std::unique_ptr<RDMAEndpoint> &endpoint,
                            std::string *out_peer_ip,
                            uint16_t *out_client_rdma_port,
                            uint16_t *out_server_rdma_port);
};

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

  status_t exchangeMetaData(std::string ip, uint16_t server_rddp_port,
                            std::unique_ptr<RDMAEndpoint> &endpoint);
};

} // namespace hmc

#endif