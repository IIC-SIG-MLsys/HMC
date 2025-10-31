/**
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */
#include "net_rdma.h"
#include <arpa/inet.h> // inet_ntoa

namespace hmc {

/* -------------------------------------------------------------------------- */
/*                               Constructor                                  */
/* -------------------------------------------------------------------------- */

RDMAServer::RDMAServer(std::shared_ptr<ConnBuffer> buffer,
                       std::shared_ptr<ConnManager> conn_manager,
                       size_t num_qps /* default = 1 */)
    : buffer(buffer), Server(conn_manager), num_qps_(num_qps) {}

RDMAServer::~RDMAServer() {}

/* -------------------------------------------------------------------------- */
/*                              Listen Loop                                   */
/* -------------------------------------------------------------------------- */

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
    return status_t::ERROR;
  }

  ret = rdma_create_id(this->cm_event_channel, &this->server_cm_id, nullptr,
                       RDMA_PS_TCP);
  if (ret) {
    logError("Server::Start: Failed to create cm id");
    goto cleanup;
  }

  memset(&sockaddr, 0, sizeof(sockaddr));
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_port = htons(port);
  inet_pton(AF_INET, ip.c_str(), &sockaddr.sin_addr);
  ret = rdma_bind_addr(this->server_cm_id, (struct sockaddr *)&sockaddr);
  if (ret) {
    logError("Server::Start: Failed to bind address");
    goto cleanup;
  }

  ret = rdma_listen(this->server_cm_id, 16);
  if (ret) {
    logError("Server::Start: Failed to listen");
    goto cleanup;
  }
  logInfo("Server::Start: Listening on %s:%d (max QPs per conn: %zu)", ip.c_str(), port, num_qps_);

  while (!should_exit()) {
    if (rdma_get_cm_event(cm_event_channel, &cm_event)) continue;

    struct rdma_cm_event event_copy;
    memcpy(&event_copy, cm_event, sizeof(*cm_event));
    rdma_ack_cm_event(cm_event);

    if (event_copy.id->route.addr.src_addr.sa_family == AF_INET) {
      client_addr = (struct sockaddr_in *)&event_copy.id->route.addr.dst_addr;
      recv_port = ntohs(client_addr->sin_port);
      if (!(inet_ntop(AF_INET, &(client_addr->sin_addr), recv_ip,
                      INET_ADDRSTRLEN))) {
        logError("Failed to parse client address");
        continue;
      }
    } else {
      logError("Unsupported address family (non-IPv4)");
      continue;
    }

    if (event_copy.event == RDMA_CM_EVENT_CONNECT_REQUEST) {
      logInfo("Recv Connect msg from %s:%d", recv_ip, recv_port);
      std::unique_ptr<hmc::RDMAEndpoint> ep = handleConnection(event_copy.id);
      if (!ep) {
        logError("Server::handleConnection failed for %s", recv_ip);
        continue;
      }

      if (exchangeMetaData(recv_ip, ep) != status_t::SUCCESS) {
        logError("Server::exchangeMetaData failed for %s", recv_ip);
        continue;
      }

      ep->transitionExtraQPsToRTS(); // make all qp RTS

      conn_manager->_addEndpoint(recv_ip, std::move(ep));
      logInfo("Server:: connection established with %s:%d", recv_ip, recv_port);

    } else if (event_copy.event == RDMA_CM_EVENT_DISCONNECTED) {
      logInfo("Recv Disconnect msg from %s:%d", recv_ip, recv_port);
      rdma_disconnect(event_copy.id);
      conn_manager->_removeEndpoint(recv_ip);
      logInfo("Disconnect success for %s", recv_ip);
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

/* -------------------------------------------------------------------------- */
/*                         Connection Setup                                   */
/* -------------------------------------------------------------------------- */

std::unique_ptr<RDMAEndpoint> RDMAServer::handleConnection(rdma_cm_id *id) {
  int ret = -1;
  struct rdma_cm_event *cm_event = nullptr;

  logDebug("buffer: buffer->ptr %p, buffer->size %zu", buffer->ptr,
           buffer->buffer_size);

  // ⚙️ 这里使用参数控制 QP 数量，默认 = 1
  auto endpoint = std::make_unique<RDMAEndpoint>(buffer, num_qps_);
  endpoint->role = EndpointType::Server;
  endpoint->cm_id = id;

  endpoint->completion_channel =
      ibv_create_comp_channel(endpoint->cm_id->verbs);
  if (!endpoint->completion_channel) {
    logError("Server::setup: Failed to create completion channel");
    goto failed;
  }

  endpoint->cq =
      ibv_create_cq(endpoint->cm_id->verbs, endpoint->cq_capacity, nullptr,
                    endpoint->completion_channel, 0);
  if (!endpoint->cq) {
    logError("Server::setup: Failed to create completion queue");
    goto failed;
  }

  endpoint->pd = ibv_alloc_pd(endpoint->cm_id->verbs);
  if (!endpoint->pd) {
    logError("Server::setup: Failed to allocate protection domain");
    goto failed;
  }

  // 多 QP 初始化（兼容单 QP 模式）
  if (endpoint->setupQPs() != status_t::SUCCESS) {
    logError("Server::setup: Failed to create QPs");
    goto failed;
  }

  if (endpoint->setupBuffers() != status_t::SUCCESS) {
    logError("Server::setup: Failed to setup buffers");
    goto failed;
  }

  /* Accept New Connection */
  struct rdma_conn_param conn_param;
  memset(&conn_param, 0, sizeof(conn_param));
  conn_param.responder_resources = endpoint->responder_resources;
  conn_param.initiator_depth = endpoint->initiator_depth;

  if (rdma_accept(endpoint->cm_id, &conn_param)) {
    logError("Server::accept: Failed to accept connection");
    goto failed;
  }

  if (rdma_get_cm_event(this->cm_event_channel, &cm_event)) {
    logError("Server::accept: Failed to get RDMA_CM_EVENT_ESTABLISHED");
    goto failed;
  }

  if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {
    logError("Server::accept: Unexpected event %s",
             rdma_event_str(cm_event->event));
    rdma_ack_cm_event(cm_event);
    goto failed;
  }

  rdma_ack_cm_event(cm_event);
  endpoint->connStatus = status_t::SUCCESS;
  logInfo("Server:: connection established (QPs: %zu)", num_qps_);
  return endpoint;

failed:
  endpoint.reset();
  return nullptr;
}

/* -------------------------------------------------------------------------- */
/*                             Metadata Exchange                              */
/* -------------------------------------------------------------------------- */

status_t RDMAServer::exchangeMetaData(std::string ip,
                                      std::unique_ptr<RDMAEndpoint> &endpoint) {
  auto &ctrl = hmc::CtrlSocketManager::instance();

  // 接收 client metadata
  decltype(endpoint->remote_metadata_attr) client_meta{};
  if (!ctrl.recvCtrlStruct(ip, client_meta)) {
    logError("[RDMAServer] exchangeMetaData: recv client metadata failed");
    return status_t::ERROR;
  }
  endpoint->remote_metadata_attr = client_meta;

  // 发送 server metadata（多 QP 信息）
  if (!ctrl.sendCtrlStruct(ip, endpoint->local_metadata_attr)) {
    logError("[RDMAServer] exchangeMetaData: send local metadata failed");
    return status_t::ERROR;
  }

  // 校验
  if (endpoint->remote_metadata_attr.address == 0 ||
      endpoint->remote_metadata_attr.length == 0 ||
      endpoint->remote_metadata_attr.key == 0) {
    logError("[RDMAServer] exchangeMetaData: invalid remote metadata");
    return status_t::ERROR;
  }

  // 输出信息（多 QP）
  logInfo("[RDMAServer] exchangeMetaData: local_qps=%u, remote_qps=%u",
          endpoint->local_metadata_attr.qp_nums,
          endpoint->remote_metadata_attr.qp_nums);

  for (uint32_t i = 0; i < endpoint->local_metadata_attr.qp_nums; ++i) {
    logDebug("  local QP[%u] num = %u", i,
             endpoint->local_metadata_attr.qp_num_list[i]);
  }
  for (uint32_t i = 0; i < endpoint->remote_metadata_attr.qp_nums; ++i) {
    logDebug("  remote QP[%u] num = %u", i,
             endpoint->remote_metadata_attr.qp_num_list[i]);
  }

  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                            Stop Listening                                  */
/* -------------------------------------------------------------------------- */

status_t RDMAServer::stopListen() {
  std::raise(SIGINT);
  rdma_cm_id *dummy_id = nullptr;
  if (rdma_create_id(cm_event_channel, &dummy_id, nullptr, RDMA_PS_TCP) == 0) {
    rdma_destroy_id(dummy_id);
  }
  return status_t::SUCCESS;
}

} // namespace hmc
