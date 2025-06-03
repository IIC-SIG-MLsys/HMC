/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_rdma.h"

#include <arpa/inet.h> // inet_ntoa

namespace hmc {

RDMAServer::RDMAServer(std::shared_ptr<ConnBuffer> buffer,
                       std::shared_ptr<ConnManager> conn_manager)
    : buffer(buffer), Server(conn_manager) {}

RDMAServer::~RDMAServer() {}

status_t RDMAServer::listen(std::string ip, uint16_t port) {
  setup_signal_handler(); // 退出信号监控

  int ret = -1;
  struct sockaddr_in sockaddr;
  struct rdma_cm_event *cm_event = nullptr;
  struct sockaddr_in *client_addr;
  char recv_ip[INET_ADDRSTRLEN];
  uint16_t recv_port = 0;

  cm_event_channel = rdma_create_event_channel(); // 创建事件通道
  if (!cm_event_channel) {
    logError("Server::Start: Failed to create event channel");
  }
  ret = rdma_create_id(this->cm_event_channel, &this->server_cm_id, nullptr,
                       RDMA_PS_TCP); // 2. 创建RDMA标识
  if (ret) {
    logError("Server::Start: Failed to create cm id");
    goto cleanup;
  }
  memset(&sockaddr, 0, sizeof(sockaddr)); // 绑定地址
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_port = htons(port);
  inet_pton(AF_INET, ip.c_str(), &sockaddr.sin_addr);
  ret = rdma_bind_addr(this->server_cm_id, (struct sockaddr *)&sockaddr);
  if (ret) {
    logError("Server::Start: Failed to bind address");
    goto cleanup;
  }
  ret = rdma_listen(this->server_cm_id, 1); // 开始监听
  if (ret) {
    logError("Server::Start: Failed to listen");
    goto cleanup;
  }
  logInfo("Server::Start: Server is listening at: %s:%d", ip.c_str(), port);

  while (!should_exit()) {
    // 处理请求事件
    rdma_get_cm_event(cm_event_channel, &cm_event);
    struct rdma_cm_event event_copy;
    memcpy(&event_copy, cm_event, sizeof(*cm_event));
    rdma_ack_cm_event(cm_event); // 马上做ACK处理，ACK之后cm_event即失效
    /* 注意：ack_cm_event会消耗掉这个事件，使得内部的部分内容失效，比如上下文verbs,所以一定要处理完之后再ack
     */

    // 解析ip, port
    // Ensure the address is of type sockaddr_in for IPv4.
    if (event_copy.id->route.addr.src_addr.sa_family == AF_INET) {
      client_addr = (struct sockaddr_in *)&event_copy.id->route.addr.dst_addr;
      // Get port number
      port = ntohs(client_addr->sin_port);
      // Convert IP address to a string
      if (!(inet_ntop(AF_INET, &(client_addr->sin_addr), recv_ip,
                      INET_ADDRSTRLEN))) {
        logInfo("Failed to Parse ip port from event, skip this event");
        continue;
      }
    } else {
      logInfo("Recv Disconnect non-Ipv4 msg, we only support Ipv4");
      continue;
    }

    /* 这里的事件处理会消耗时间，存在影响处理并发链接的性能问题，可以队列处理，但是队列会导致事件不严格有序，导致链接建立存在潜在的失败风险
     */
    if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST) {
      // 连接请求
      logInfo("Recv Connect msg from %s:%d", recv_ip, recv_port);
      if (ret) {
        logError("Server::Start: Failed to acknowledge cm event");
        continue;
      }
      logDebug("endpoint cm_id %p", event_copy.id);
      std::unique_ptr<hmc::Endpoint> ep = handleConnection(event_copy.id);
      conn_manager->_addEndpoint(ip, std::move(ep));
      // conn_manager->_printEndpointMap();
      // handleConnection的时候，内部会处理一个链接建立完成事件
    } else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED) {
      // 断开请求：根据对端的ip信息，清理本地对应的ep
      logInfo("Recv Disconnect msg from %s:%d", recv_ip, recv_port);
      if (ret) {
        logError("Server::Start: Failed to acknowledge cm event");
        continue;
      }
      rdma_disconnect(event_copy.id); // 双向断连机制,需要服务端也回复断开
      conn_manager->_removeEndpoint(recv_ip);
      // conn_manager->_printEndpointMap();
      logInfo("Disconnect success");
    }
  }

cleanup:
  if (cm_event_channel) {
    rdma_destroy_event_channel(cm_event_channel);
    cm_event_channel = nullptr;
  }
  if (server_cm_id) {
    rdma_destroy_id(server_cm_id);
    server_cm_id = nullptr;
  }
  return status_t::SUCCESS;
}

std::unique_ptr<Endpoint> RDMAServer::handleConnection(rdma_cm_id *id) {
  int ret = -1;
  struct rdma_cm_event *cm_event = nullptr;

  logDebug("buffer: buffer->ptr %p, buffer->size %zu", buffer->ptr,
           buffer->buffer_size);
  auto endpoint =
      std::make_unique<RDMAEndpoint>(buffer);
  endpoint->role = EndpointType::Server;

  endpoint->cm_id = id;

  /* SETUP */
  endpoint->completion_channel =
      ibv_create_comp_channel(endpoint->cm_id->verbs); // 创建完成通道
  if (!endpoint->completion_channel) {
    logError("Server::setup: Failed to create completion channel");
    goto failed;
  }
  endpoint->cq =
      ibv_create_cq(endpoint->cm_id->verbs, endpoint->cq_capacity, nullptr,
                    endpoint->completion_channel, 0); // 创建完成队列
  if (!endpoint->cq) {
    logError("Server::setup: Failed to create completion queue");
    goto failed;
  }
  endpoint->pd = ibv_alloc_pd(endpoint->cm_id->verbs); // 创建PD
  if (!endpoint->pd) {
    logError("Server::setup: Failed to allocate protection domain");
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
    logError("Server::setup: Failed to create QP");
    goto failed;
  }
  endpoint->qp = endpoint->cm_id->qp;
  if (endpoint->setupBuffers() != status_t::SUCCESS) { // 将数据buffer以及其attr
    logError("Server::setup: Failed to setup buffers");
    goto failed;
  }

  // 交换元数据：buffer的元信息交换预接受准备
  if (prePostExchangeMetadata(endpoint) !=
      status_t::SUCCESS) { // metadata for both side.
    goto failed;
  }

  /* ACCEPT New Connection, get MetaData */
  struct rdma_conn_param conn_param;
  memset(&conn_param, 0, sizeof(conn_param));
  conn_param.responder_resources = endpoint->responder_resources;
  conn_param.initiator_depth = endpoint->initiator_depth;

  if (rdma_accept(endpoint->cm_id, &conn_param)) {
    logError(
        "Server::server_accept_newconnection: Failed to accept connection");
    goto failed;
  }
  if (rdma_get_cm_event(this->cm_event_channel, &cm_event)) {
    logError("Server::server_accept_newconnection: Failed to get "
             "RDMA_CM_EVENT_ESTABLISHED event");
    goto failed;
  }
  if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {
    logError("Server::server_accept_newconnection: Unexpected event %s",
             rdma_event_str(cm_event->event));
    rdma_ack_cm_event(cm_event);
    goto failed;
  }

  // 交换元数据：buffer的元信息交换
  if (exchangeMetadata(endpoint) !=
      status_t::SUCCESS) { // metadata for both side.
    logError("Client::start_client: Failed to exchange metadata");
    goto failed;
  }
  logDebug("Server new connection started successfully");

  // 成功创建
  endpoint->connStatus = status_t::SUCCESS;
  return endpoint;

failed:
  endpoint.reset();
  return nullptr;
}

status_t
RDMAServer::prePostExchangeMetadata(std::unique_ptr<RDMAEndpoint> &endpoint) {
  // 接收对端的元数据：发送前先准备一个接收。
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
    logError("Server::exchange_metadata: Failed to post recv");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAServer::exchangeMetadata(std::unique_ptr<RDMAEndpoint> &endpoint) {
  struct ibv_send_wr send_wr, *bad_send_wr = nullptr;
  struct ibv_sge send_sge;

  // 发送localmeta到远程
  memset(&send_wr, 0, sizeof(send_wr));
  memset(&send_sge, 0, sizeof(send_sge));

  send_sge.addr = (uint64_t)&endpoint->local_metadata_attr;
  send_sge.length = sizeof(endpoint->local_metadata_attr);
  send_sge.lkey = endpoint->local_metadata_mr->lkey;

  send_wr.wr_id = 0;
  send_wr.next = nullptr;
  send_wr.sg_list = &send_sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND;
  send_wr.send_flags = IBV_SEND_SIGNALED;

  // 发布发送请求
  if (ibv_post_send(endpoint->qp, &send_wr, &bad_send_wr)) {
    logError("Server::exchange_metadata: Failed to post send");
    return status_t::ERROR;
  }

  // 等待发送完成
  if (endpoint->pollCompletion(2) != status_t::SUCCESS) {
    logError("Client::exchange_metadata: Failed to complete metadata "
             "exchange");
    return status_t::ERROR;
  }

  // 打印调试信息
  logDebug("Server::exchange_metadata: Local metadata:");
  endpoint->showRdmaBufferAttr(&endpoint->local_metadata_attr);
  logDebug("Server::exchange_metadata: Remote metadata:");
  endpoint->showRdmaBufferAttr(&endpoint->remote_metadata_attr);

  if (endpoint->remote_metadata_attr.address == 0) {
    logError("Server::exchange_metadata: Failed to get remote metadata");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAServer::stopListen() {
  std::raise(SIGINT);
  // 唤醒 server 监听线程（阻塞在event_channel上）
  // 创建 dummy cm_id 打断 rdma_cm_get_event
  rdma_cm_id *dummy_id = nullptr;
  if (rdma_create_id(cm_event_channel, &dummy_id, nullptr, RDMA_PS_TCP) == 0) {
    rdma_destroy_id(dummy_id); // destroy 之后会触发一个 event
  }
  return status_t::SUCCESS;
}

} // namespace hmc