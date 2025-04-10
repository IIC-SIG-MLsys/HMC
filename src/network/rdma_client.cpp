/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_rdma.h"
#include <arpa/inet.h> // inet_ntoa

namespace hmc {

RDMAClient::RDMAClient(std::shared_ptr<ConnBuffer> buffer, int max_retry_times,
                       int retry_delay_ms)
    : buffer(buffer), max_retry_times(max_retry_times),
      retry_delay_ms(retry_delay_ms) {}
RDMAClient::~RDMAClient(){};

std::unique_ptr<Endpoint> RDMAClient::connect(std::string ip, uint16_t port) {
  setup_signal_handler(); // 退出信号监控

  int ret = -1;
  struct rdma_cm_event *cm_event = nullptr;
  struct rdma_conn_param conn_param;

  // sockaddr
  std::memset(&sockaddr, 0, sizeof(sockaddr));
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &sockaddr.sin_addr) <= 0) {
    logError("Client::Start: Failed to convert IP address to binary form");
  }

  auto endpoint =
      std::make_unique<RDMAEndpoint>(buffer);
  endpoint->role = EndpointType::Client;

  while (this->retry_count < this->max_retry_times) {
    /** SETUP **/
    endpoint->cm_event_channel = rdma_create_event_channel(); // 创建事件通道
    if (!endpoint->cm_event_channel) {
      logError("Client::setup_client: Failed to create event channel");
      goto failed;
    }
    ret = rdma_create_id(endpoint->cm_event_channel, &endpoint->cm_id, nullptr,
                         RDMA_PS_TCP); // 创建RDMA标识id
    if (ret) {
      logError("Client::setup_client: Failed to create cm id");
      goto failed;
    }
    ret = rdma_resolve_addr(endpoint->cm_id, nullptr,
                            (struct sockaddr *)&this->sockaddr,
                            resolve_addr_timeout_ms); // 解析地址
    if (ret) {
      logError("Client::setup_client: Failed to resolve addr");
      goto retry;
    }
    ret = rdma_get_cm_event(endpoint->cm_event_channel,
                            &cm_event); // 等地址解析完成
    if (ret) {
      logError("Client::setup_client: Failed to get cm event");
      goto retry;
    }
    if (cm_event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
      logError("Client::setup_client: Unexpected cm event %s",
               rdma_event_str(cm_event->event));
      rdma_ack_cm_event(cm_event);
      goto retry;
    }
    rdma_ack_cm_event(cm_event);
    ret = rdma_resolve_route(endpoint->cm_id,
                             resolve_addr_timeout_ms); // 解析路由
    if (ret) {
      logError("Client::setup_client: Failed to resolve route, ret = %d", ret);
      goto retry;
    }
    logDebug("Client::setup_client: route resolved");
    ret = rdma_get_cm_event(endpoint->cm_event_channel,
                            &cm_event); // 等待路由解析完成
    if (ret) {
      logError("Client::setup_client: Failed to get cm event");
      goto retry;
    }
    if (cm_event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
      logError("Client::setup_client: Unexpected cm event %s",
               rdma_event_str(cm_event->event));
      rdma_ack_cm_event(cm_event);
      goto retry;
    }
    rdma_ack_cm_event(cm_event);

    endpoint->completion_channel =
        ibv_create_comp_channel(endpoint->cm_id->verbs); // 创建完成通道
    if (!endpoint->completion_channel) {
      logError("Failed to create completion channel");
      goto failed;
    }
    endpoint->cq =
        ibv_create_cq(endpoint->cm_id->verbs, endpoint->cq_capacity, nullptr,
                      endpoint->completion_channel, 0); // 创建完成队列
    if (!endpoint->cq) {
      logError("Failed to create completion queue");
      goto failed;
    }
    endpoint->pd = ibv_alloc_pd(endpoint->cm_id->verbs); // 创建PD
    if (!endpoint->pd) {
      logError("Failed to allocate protection domain");
      goto failed;
    }
    memset(&endpoint->qp_init_attr, 0,
           sizeof(endpoint->qp_init_attr)); // 初始化QP属性
    endpoint->qp_init_attr.qp_type = IBV_QPT_RC;
    endpoint->qp_init_attr.send_cq = endpoint->cq;
    endpoint->qp_init_attr.recv_cq = endpoint->cq;
    endpoint->qp_init_attr.cap.max_send_wr = endpoint->max_wr;
    endpoint->qp_init_attr.cap.max_recv_wr = endpoint->max_wr;
    endpoint->qp_init_attr.cap.max_send_sge = endpoint->max_sge;
    endpoint->qp_init_attr.cap.max_recv_sge = endpoint->max_sge;
    ret = rdma_create_qp(endpoint->cm_id, endpoint->pd,
                         &endpoint->qp_init_attr); // 创建QP
    if (ret) {
      logError("Client::setup_client: Failed to create QP");
      goto failed;
    }
    endpoint->qp = endpoint->cm_id->qp;
    if (endpoint->setupBuffers() !=
        status_t::SUCCESS) { // 将数据buffer以及其attr
      logError("Client::setup_client: Failed to setup buffers");
      goto failed;
    }
    // prePostExchangeMetadata(endpoint);

    /** START **/
    memset(&conn_param, 0, sizeof(conn_param));
    conn_param.responder_resources = endpoint->responder_resources;
    conn_param.initiator_depth = endpoint->initiator_depth;
    conn_param.retry_count = 3;
    if (rdma_connect(endpoint->cm_id, &conn_param)) { // 发起连接
      logError("Client::start_client: Failed to connect to server");
      goto retry;
    }
    if (rdma_get_cm_event(endpoint->cm_event_channel, &cm_event)) {
      logError("Client::start_client: Failed to get RDMA_CM_EVENT_ESTABLISHED "
               "event");
      goto retry;
    }
    if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {
      logError("Client::start_client: Unexpected event %s",
               rdma_event_str(cm_event->event));
      rdma_ack_cm_event(cm_event);
      goto retry;
    }
    rdma_ack_cm_event(cm_event);
    logInfo("Client:: Connection to %s:%d established", ip.c_str(), port);

    // 交换元数据：buffer的元信息交换
    if (exchangeMetadata(endpoint) !=
        status_t::SUCCESS) { // metadata for both side.
      logError("Client::start_client: Failed to exchange metadata");
      goto failed;
    }
    logDebug("Client started successfully");
    endpoint->connStatus = status_t::SUCCESS;
    return endpoint;

  retry:
    this->retry_count++;
    logError("Client::Start: Retry to connect server (%d/%d)",
             this->retry_count, this->max_retry_times);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(this->retry_delay_ms));
    endpoint->cleanRdmaResources();
    if (should_exit())
      break;
  }

failed:
  endpoint.reset();
  // 返回 unique_ptr 给调用者
  return nullptr;
}

status_t
RDMAClient::prePostExchangeMetadata(std::unique_ptr<RDMAEndpoint> &endpoint) {
  // 接收服务器的元数据：发送前先准备一个接收，因为对面是阻塞先收后发。
  struct ibv_recv_wr recv_wr, *bad_recv_wr = nullptr;
  struct ibv_sge recv_sge;

  memset(&recv_wr, 0, sizeof(recv_wr));
  memset(&recv_sge, 0, sizeof(recv_sge));

  recv_sge.addr = (uint64_t)&endpoint->remote_metadata_attr;
  recv_sge.length = sizeof(endpoint->remote_metadata_attr);
  recv_sge.lkey = endpoint->remote_metadata_mr->lkey;

  recv_wr.wr_id = 0;
  recv_wr.sg_list = &recv_sge;
  recv_wr.num_sge = 1;

  // 发布接收请求
  if (ibv_post_recv(endpoint->qp, &recv_wr, &bad_recv_wr)) {
    logError("Client::exchange_metadata: Failed to post recv");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAClient::exchangeMetadata(std::unique_ptr<RDMAEndpoint> &endpoint) {
  /** buffer 的元信息交换 **/

  // 准备发送本地元数据
  struct ibv_send_wr send_wr, *bad_send_wr = nullptr;
  struct ibv_sge send_sge;

  memset(&send_wr, 0, sizeof(send_wr));
  memset(&send_sge, 0, sizeof(send_sge));

  send_sge.addr = (uint64_t)&endpoint->local_metadata_attr;
  send_sge.length = sizeof(endpoint->local_metadata_attr);
  send_sge.lkey = endpoint->local_metadata_mr->lkey;

  send_wr.wr_id = 0;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND;
  send_wr.send_flags = IBV_SEND_SIGNALED;

  // 发送本地元数据
  if (ibv_post_send(endpoint->qp, &send_wr, &bad_send_wr)) {
    logError("Client::exchange_metadata: Failed to post send");
    return status_t::ERROR;
  }

  if (prePostExchangeMetadata(endpoint) !=
      status_t::SUCCESS) { // metadata for both side.
    return status_t::ERROR;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(3));

  // 等待接收和发送完成
  if (endpoint->pollCompletion(2) != status_t::SUCCESS) {
    logError("Client::exchange_metadata: Failed to complete metadata "
             "exchange");
    return status_t::ERROR;
  }

  // 打印调试信息
  logDebug("Client::exchange_metadata: Local metadata:");
  endpoint->showRdmaBufferAttr(&endpoint->local_metadata_attr);
  logDebug("Client::exchange_metadata: Remote metadata:");
  endpoint->showRdmaBufferAttr(&endpoint->remote_metadata_attr);

  if (endpoint->remote_metadata_attr.address == 0) {
    logError("Client::exchange_metadata: Failed to get remote metadata");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

} // namespace hmc