/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_NET_RDMA_H
#define HDDT_NET_RDMA_H

#include "net.h"

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

struct __attribute((packed)) rdma_buffer_attr {
  uint64_t address;
  uint32_t length;
  uint32_t key;
};

namespace hddt {

class RDMAEndpoint: public Endpoint {
/* cm_id, qp, cq, mr 是RDMA通信需要持有的四个关键元素，每个Endpoint应持有隔离的四元组 */
public:
    RDMAEndpoint(void *buffer, size_t buffer_size);
    status_t closeEndpoint() override;

    status_t writeData(size_t data_bias, size_t size) override;
    status_t readData(size_t data_bias, size_t size) override;
    
    status_t writeDataNB(size_t data_bias, size_t size);
    status_t readDataNB(size_t data_bias, size_t size);

    status_t pollCompletion(int num_completions_to_process);

    status_t registerMemory(void *addr, size_t length, struct ibv_mr **mr);
    status_t deRegisterMemory(struct ibv_mr *mr);
    status_t setupBuffers();
    void showRdmaBufferAttr(const struct rdma_buffer_attr *attr);
    void cleanRdmaResources();

    status_t postSend(void *addr, size_t length,
                        struct ibv_mr *mr, enum ibv_wr_opcode opcode,
                        bool signaled = true);
    status_t postRecv(void *addr, size_t length,
                        struct ibv_mr *mr);
    status_t postWrite(void *local_addr, void *remote_addr,
                        size_t length, struct ibv_mr *local_mr,
                        uint32_t remote_key, bool signaled);
    status_t postRead(void *local_addr, void *remote_addr,
                        size_t length, struct ibv_mr *local_mr,
                        uint32_t remote_key, bool signaled);

    ~RDMAEndpoint();

public:
    void *buffer = NULL; // register
    size_t buffer_size = 0;
    bool is_buffer_ok = false;

    // the RDMA connection identifier : cm(connection management)
    struct rdma_cm_id *cm_id = NULL; // server创建的时候传入的是remote_cm_id
    // qp(queue pair)
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_qp* qp; // Queue Pair
    // Protect Domain
    struct ibv_pd *pd = NULL;
    // completion queue
    struct ibv_cq *cq = NULL;
    // Memory Region
    struct ibv_mr *remote_metadata_mr = NULL;
    struct ibv_mr *local_metadata_mr = NULL;
    struct ibv_mr *buffer_mr = NULL;
    // RDMA buffer attributes
    struct rdma_buffer_attr remote_metadata_attr;
    struct rdma_buffer_attr local_metadata_attr;
    // Event Channel : report asynchronous communication event
    struct rdma_event_channel *cm_event_channel = NULL;
    // Completion Channel
    struct ibv_comp_channel *completion_channel = NULL;

    // suggest 2-8
    uint8_t initiator_depth = 8;
    uint8_t responder_resources = 8;
    uint8_t retry_count = 3;

    uint16_t cq_capacity = 16;
    uint16_t max_sge = 2;
    uint16_t max_wr = 8;
};


class RDMAServer : public Server {
public:
    RDMAServer(std::shared_ptr<ConnBuffer> buffer, std::shared_ptr<ConnManager> conn_manager);
    ~RDMAServer();

    status_t listen(std::string ip, uint16_t port) override;

    std::unique_ptr<Endpoint> handleConnection(rdma_cm_id *id);
    status_t exchangeMetadata(std::unique_ptr<RDMAEndpoint>& endpoint);
private:
    std::shared_ptr<ConnBuffer> buffer;
    struct rdma_cm_id *server_cm_id = NULL; // server用来监听的cm
    struct rdma_event_channel *cm_event_channel = NULL;
};


class RDMAClient : public Client {
public:
    RDMAClient(std::shared_ptr<ConnBuffer> buffer, int max_retry_times = 10, int retry_delay_ms = 1000);
    ~RDMAClient();

    std::unique_ptr<Endpoint> connect(std::string ip, uint16_t port);

    status_t exchangeMetadata(std::unique_ptr<RDMAEndpoint>& ep);
private:
    int max_retry_times;
    int retry_delay_ms;
    int retry_count = 0;
    int resolve_addr_timeout_ms = 2000;
    struct sockaddr_in sockaddr;
    std::shared_ptr<ConnBuffer> buffer;
};

}
#endif