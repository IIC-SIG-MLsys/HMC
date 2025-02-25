/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_rdma.h"

#include <arpa/inet.h> // inet_ntoa

namespace hddt {

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
    // 解析ip, port
    // Ensure the address is of type sockaddr_in for IPv4.
    if (cm_event->id->route.addr.src_addr.sa_family == AF_INET) {
      client_addr = (struct sockaddr_in *)&cm_event->id->route.addr.dst_addr;
      // Get port number
      port = ntohs(client_addr->sin_port);
      // Convert IP address to a string
      if (!(inet_ntop(AF_INET, &(client_addr->sin_addr), recv_ip,
                      INET_ADDRSTRLEN))) {
        rdma_ack_cm_event(cm_event);
        logInfo("Failed to Parse ip port from event, skip this event");
        continue;
      }
    } else {
      rdma_ack_cm_event(cm_event);
      logInfo("Recv Disconnect non-Ipv4 msg, we only support Ipv4");
      continue;
    }

    /* 这里的事件处理会消耗时间，存在影响处理并发链接的性能问题，可以队列处理，但是队列会导致事件不严格有序，导致链接建立存在潜在的失败风险
     */
    if (cm_event->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
      // 连接请求
      logInfo("Recv Connect msg from %s:%d", recv_ip, recv_port);
      if (ret) {
        logError("Server::Start: Failed to acknowledge cm event");
        continue;
      }
      logDebug("endpoint cm_id %p", cm_event->id);
      std::unique_ptr<hddt::Endpoint> ep = handleConnection(cm_event->id);
      conn_manager->_addEndpoint(ip, std::move(ep));
      // conn_manager->_printEndpointMap();
      rdma_ack_cm_event(
          cm_event); /* 注意：ack_cm_event会消耗掉这个事件，使得内部的部分内容失效，比如上下文verbs,所以一定要处理完之后再ack
                      */
      // handleConnection的时候，内部会处理一个链接建立完成事件
    } else if (cm_event->event == RDMA_CM_EVENT_DISCONNECTED) {
      // 断开请求：根据对端的ip信息，清理本地对应的ep
      logInfo("Recv Disconnect msg from %s:%d", recv_ip, recv_port);
      rdma_ack_cm_event(cm_event);
      if (ret) {
        logError("Server::Start: Failed to acknowledge cm event");
        continue;
      }
      conn_manager->_removeEndpoint(recv_ip);
      // conn_manager->_printEndpointMap();
      logInfo("Disconnect success");
    } else {
      rdma_ack_cm_event(cm_event); /* ack anyway */
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
      std::make_unique<RDMAEndpoint>(buffer->ptr, buffer->buffer_size);
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
  return endpoint;

failed:
  // endpoint自动销毁
  return nullptr;
}

status_t RDMAServer::exchangeMetadata(std::unique_ptr<RDMAEndpoint> &endpoint) {
  struct ibv_send_wr send_wr, *bad_send_wr = nullptr;
  struct ibv_recv_wr recv_wr, *bad_recv_wr = nullptr;
  struct ibv_sge send_sge, recv_sge;

  // 接收远程的元数据
  memset(&recv_wr, 0, sizeof(recv_wr));
  memset(&recv_sge, 0, sizeof(recv_sge));

  recv_sge.addr = (uint64_t)&endpoint->remote_metadata_attr;
  recv_sge.length = sizeof(endpoint->remote_metadata_attr);
  recv_sge.lkey = endpoint->remote_metadata_mr->lkey;

  memset(&recv_wr, 0, sizeof(recv_wr));
  recv_wr.wr_id = 0;
  recv_wr.sg_list = &recv_sge;
  recv_wr.num_sge = 1;

  /* 由于需要先准备好recv才能接受对端发送的内容，所以需要有一方等待对方准备好接收消息
   */
  // 发布接收请求
  if (ibv_post_recv(endpoint->qp, &recv_wr, &bad_recv_wr)) {
    logError(
        "Server::server_send_metadata_to_newconnection: Failed to post recv");
    return status_t::ERROR;
  }

  // 等待接收完成
  if (endpoint->pollCompletion(1) != status_t::SUCCESS) {
    logError("Client::client_exchange_metadata: Failed to complete metadata "
             "exchange");
    return status_t::ERROR;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(
      1000)); // 发送之前应该等待一下，等待对方已经准备好接收事件。
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
  if (endpoint->pollCompletion(1) != status_t::SUCCESS) {
    logError("Client::client_exchange_metadata: Failed to complete metadata "
             "exchange");
    return status_t::ERROR;
  }

  // 打印调试信息
  logDebug("Server::server_send_metadata_to_newconnection: Local metadata:");
  endpoint->showRdmaBufferAttr(&endpoint->local_metadata_attr);
  logDebug("Server::server_send_metadata_to_newconnection: Remote metadata:");
  endpoint->showRdmaBufferAttr(&endpoint->remote_metadata_attr);

  /** TODO：在endpoint->setupBuffers()增加，支持更多的数据buffer;
   * 在这里完成buffer的数据信息交换 **/

  return status_t::SUCCESS;
}

} // namespace hddt