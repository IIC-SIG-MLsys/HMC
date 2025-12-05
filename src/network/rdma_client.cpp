/**
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */
#include "net_rdma.h"
#include <hmc.h>

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

bool sendAll(int fd, const void* buf, size_t len) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
  while (len > 0) {
    ssize_t n = ::send(fd, p, len, 0);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool recvAll(int fd, void* buf, size_t len) {
  uint8_t* p = reinterpret_cast<uint8_t*>(buf);
  while (len > 0) {
    ssize_t n = ::recv(fd, p, len, 0);
    if (n <= 0) return false;
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool sendU16(int fd, uint16_t v) { return sendAll(fd, &v, sizeof(v)); }

bool sendMsg(int fd, hmc::CtrlMsgType type, const void* payload, size_t len,
             uint16_t flags = 0) {
  hmc::CtrlMsgHeader hdr{static_cast<uint16_t>(type), flags,
                         static_cast<uint32_t>(len)};
  if (!sendAll(fd, &hdr, sizeof(hdr))) return false;
  if (payload && len > 0 && !sendAll(fd, payload, len)) return false;
  return true;
}

bool recvMsg(int fd, hmc::CtrlMsgHeader& hdr, std::vector<uint8_t>& payload) {
  if (!recvAll(fd, &hdr, sizeof(hdr))) return false;
  payload.resize(hdr.length);
  if (hdr.length > 0 && !recvAll(fd, payload.data(), hdr.length)) return false;
  return true;
}

int dialOnce(const std::string& ip, uint16_t port) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) {
    ::close(fd);
    return -1;
  }
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

} // namespace

namespace hmc {

RDMAClient::RDMAClient(std::shared_ptr<ConnBuffer> buffer,
                       size_t num_qps,
                       int max_retry_times,
                       int retry_delay_ms)
    : buffer(buffer),
      max_retry_times(max_retry_times),
      retry_delay_ms(retry_delay_ms),
      num_qps_(num_qps) {}

RDMAClient::~RDMAClient() {}

std::unique_ptr<Endpoint> RDMAClient::connect(std::string ip, uint16_t port) {
  setup_signal_handler();

  int ret = -1;
  struct rdma_cm_event* cm_event = nullptr;
  struct rdma_conn_param conn_param;

  std::memset(&sockaddr, 0, sizeof(sockaddr));
  sockaddr.sin_family = AF_INET;
  sockaddr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &sockaddr.sin_addr) <= 0) {
    logError("Client::Start: Failed to convert IP address");
  }

  auto endpoint = std::make_unique<RDMAEndpoint>(buffer, num_qps_);
  endpoint->role = EndpointType::Client;

  this->retry_count = 0;

  while (this->retry_count < this->max_retry_times) {
    endpoint->cm_event_channel = rdma_create_event_channel();
    if (!endpoint->cm_event_channel) {
      logError("Client::setup_client: Failed to create event channel");
      goto failed;
    }

    ret = rdma_create_id(endpoint->cm_event_channel, &endpoint->cm_id, nullptr,
                         RDMA_PS_TCP);
    if (ret) {
      logError("Client::setup_client: Failed to create cm id");
      goto failed;
    }

    ret = rdma_resolve_addr(endpoint->cm_id, nullptr,
                            (struct sockaddr*)&this->sockaddr,
                            resolve_addr_timeout_ms);
    if (ret) {
      logError("Client::setup_client: Failed to resolve addr");
      goto retry;
    }

    if (rdma_get_cm_event(endpoint->cm_event_channel, &cm_event)) {
      logError("Client::setup_client: Failed to get cm event (ADDR_RESOLVED)");
      goto retry;
    }
    if (cm_event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
      logError("Unexpected event %s", rdma_event_str(cm_event->event));
      rdma_ack_cm_event(cm_event);
      goto retry;
    }
    rdma_ack_cm_event(cm_event);

    ret = rdma_resolve_route(endpoint->cm_id, resolve_addr_timeout_ms);
    if (ret) {
      logError("Client::setup_client: Failed to resolve route");
      goto retry;
    }
    if (rdma_get_cm_event(endpoint->cm_event_channel, &cm_event)) {
      logError("Client::setup_client: Failed to get cm event (ROUTE_RESOLVED)");
      goto retry;
    }
    if (cm_event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
      logError("Unexpected event %s", rdma_event_str(cm_event->event));
      rdma_ack_cm_event(cm_event);
      goto retry;
    }
    rdma_ack_cm_event(cm_event);

    endpoint->completion_channel =
        ibv_create_comp_channel(endpoint->cm_id->verbs);
    if (!endpoint->completion_channel) {
      logError("Client::setup_client: Failed to create completion channel");
      goto failed;
    }

    endpoint->cq =
        ibv_create_cq(endpoint->cm_id->verbs, endpoint->cq_capacity, nullptr,
                      endpoint->completion_channel, 0);
    if (!endpoint->cq) {
      logError("Client::setup_client: Failed to create CQ");
      goto failed;
    }

    endpoint->pd = ibv_alloc_pd(endpoint->cm_id->verbs);
    if (!endpoint->pd) {
      logError("Client::setup_client: Failed to allocate PD");
      goto failed;
    }

    if (endpoint->setupQPs() != status_t::SUCCESS) {
      logError("Client::setup_client: Failed to create %zu QPs", num_qps_);
      goto failed;
    }

    if (endpoint->setupBuffers() != status_t::SUCCESS) {
      logError("Client::setup_client: Failed to setup buffers");
      goto failed;
    }

    std::memset(&conn_param, 0, sizeof(conn_param));
    conn_param.responder_resources = endpoint->responder_resources;
    conn_param.initiator_depth = endpoint->initiator_depth;
    conn_param.retry_count = endpoint->retry_count;

    if (rdma_connect(endpoint->cm_id, &conn_param)) {
      logError("Client::connect: Failed to connect to server");
      goto retry;
    }

    if (rdma_get_cm_event(endpoint->cm_event_channel, &cm_event)) {
      logError("Client::connect: Failed to get RDMA_CM_EVENT_ESTABLISHED");
      goto retry;
    }
    if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {
      logError("Client::connect: Unexpected event %s",
               rdma_event_str(cm_event->event));
      rdma_ack_cm_event(cm_event);
      goto retry;
    }
    rdma_ack_cm_event(cm_event);

    logInfo("Client:: Connected to %s:%d (QPs: %zu)", ip.c_str(), port, num_qps_);

    if (exchangeMetaData(ip, port, endpoint) != status_t::SUCCESS) goto failed;

    endpoint->transitionExtraQPsToRTS();
    endpoint->connStatus = status_t::SUCCESS;
    logInfo("Client:: Connection established successfully");
    return endpoint;

  retry:
    this->retry_count++;
    logError("Client:: Retry to connect server (%d/%d)", this->retry_count,
             this->max_retry_times);
    std::this_thread::sleep_for(std::chrono::milliseconds(this->retry_delay_ms));
    endpoint->cleanRdmaResources();
    if (should_exit()) break;
  }

failed:
  endpoint.reset();
  return nullptr;
}

status_t RDMAClient::exchangeMetaData(std::string ip, uint16_t port,
                                      std::unique_ptr<RDMAEndpoint>& endpoint) {
  uint16_t meta_port = static_cast<uint16_t>(port + 10000);

  int fd = dialOnce(ip, meta_port);
  if (fd < 0) {
    logError("[RDMAClient] exchangeMetaData: dialOnce %s:%u failed: %s",
             ip.c_str(), meta_port, strerror(errno));
    return status_t::ERROR;
  }

  const uint16_t local_data_port = port;
  if (!sendU16(fd, local_data_port)) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    logError("[RDMAClient] exchangeMetaData: send local_data_port failed");
    return status_t::ERROR;
  }

  if (!sendMsg(fd, hmc::CtrlMsgType::CTRL_STRUCT, &endpoint->local_metadata_attr,
               sizeof(endpoint->local_metadata_attr))) {
    ::shutdown(fd, SHUT_RDWR);
    ::close(fd);
    logError("[RDMAClient] exchangeMetaData: send local metadata failed");
    return status_t::ERROR;
  }

  decltype(endpoint->remote_metadata_attr) server_meta{};
  {
    hmc::CtrlMsgHeader hdr{};
    std::vector<uint8_t> payload;
    if (!recvMsg(fd, hdr, payload) ||
        hdr.type != static_cast<uint16_t>(hmc::CtrlMsgType::CTRL_STRUCT) ||
        payload.size() != sizeof(server_meta)) {
      ::shutdown(fd, SHUT_RDWR);
      ::close(fd);
      logError("[RDMAClient] exchangeMetaData: recv server metadata failed");
      return status_t::ERROR;
    }
    std::memcpy(&server_meta, payload.data(), sizeof(server_meta));
  }
  endpoint->remote_metadata_attr = server_meta;

  ::shutdown(fd, SHUT_RDWR);
  ::close(fd);

  if (endpoint->remote_metadata_attr.address == 0 ||
      endpoint->remote_metadata_attr.length == 0 ||
      endpoint->remote_metadata_attr.key == 0) {
    logError("[RDMAClient] exchangeMetaData: invalid remote metadata");
    return status_t::ERROR;
  }

  logInfo("[RDMAClient] exchangeMetaData: local_qps=%u, remote_qps=%u",
          endpoint->local_metadata_attr.qp_nums,
          endpoint->remote_metadata_attr.qp_nums);

  for (uint32_t i = 0; i < endpoint->local_metadata_attr.qp_nums; ++i)
    logDebug("  local QP[%u] num = %u", i,
             endpoint->local_metadata_attr.qp_num_list[i]);
  for (uint32_t i = 0; i < endpoint->remote_metadata_attr.qp_nums; ++i)
    logDebug("  remote QP[%u] num = %u", i,
             endpoint->remote_metadata_attr.qp_num_list[i]);

  return status_t::SUCCESS;
}

} // namespace hmc
