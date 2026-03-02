/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "./net.h"
#include "./net_rdma.h"
#include <hmc.h>

namespace hmc {

Communicator::Communicator(std::shared_ptr<ConnBuffer> buffer, size_t num_chs)
    : buffer(buffer) {
  conn_manager = std::make_shared<ConnManager>(buffer, num_chs);
}

Communicator::~Communicator() {
  logDebug("close communicator");
  conn_manager.reset();
  logDebug("finished");
}

status_t Communicator::put(std::string ip,
                           uint16_t port,
                           size_t local_off,
                           size_t remote_off,
                           size_t size,
                           ConnType connType) {
  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [local_off, remote_off, size](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        return ep->writeData(local_off, remote_off, size);
      });
}

status_t Communicator::get(std::string ip,
                           uint16_t port,
                           size_t local_off,
                           size_t remote_off,
                           size_t size,
                           ConnType connType) {
  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [local_off, remote_off, size](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        return ep->readData(local_off, remote_off, size);
      });
}

status_t Communicator::putNB(std::string ip,
                             uint16_t port,
                             size_t local_off,
                             size_t remote_off,
                             size_t size,
                             uint64_t *wr_id,
                             ConnType connType) {
  if (wr_id) *wr_id = 0;

  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [this, local_off, remote_off, size, wr_id](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;

        uint64_t id = 0;
        status_t r = ep->writeDataNB(local_off, remote_off, size, &id);
        if (r != status_t::SUCCESS) return r;

        if (wr_id) *wr_id = id;

        if (id != 0) {
          std::lock_guard<std::mutex> lk(inflight_mu_);
          inflight_ep_[id] = ep;
        }
        return status_t::SUCCESS;
      });
}

status_t Communicator::getNB(std::string ip,
                             uint16_t port,
                             size_t local_off,
                             size_t remote_off,
                             size_t size,
                             uint64_t *wr_id,
                             ConnType connType) {
  if (wr_id) *wr_id = 0;

  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [this, local_off, remote_off, size, wr_id](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;

        uint64_t id = 0;
        status_t r = ep->readDataNB(local_off, remote_off, size, &id);
        if (r != status_t::SUCCESS) return r;

        if (wr_id) *wr_id = id;

        if (id != 0) {
          std::lock_guard<std::mutex> lk(inflight_mu_);
          inflight_ep_[id] = ep;
        }
        return status_t::SUCCESS;
      });
}

status_t Communicator::wait(uint64_t wr_id) {
  if (wr_id == 0) return status_t::SUCCESS;

  Endpoint *ep = nullptr;
  {
    std::lock_guard<std::mutex> lk(inflight_mu_);
    auto it = inflight_ep_.find(wr_id);
    if (it == inflight_ep_.end()) return status_t::NOT_FOUND;
    ep = it->second;
    inflight_ep_.erase(it);
  }
  if (!ep) return status_t::ERROR;

  return ep->waitWrId(wr_id);
}

status_t Communicator::wait(const std::vector<uint64_t> &wr_ids) {
  std::unordered_map<Endpoint *, std::vector<uint64_t>> by_ep;
  by_ep.reserve(8);

  {
    std::lock_guard<std::mutex> lk(inflight_mu_);
    for (uint64_t id : wr_ids) {
      if (id == 0) continue;

      auto it = inflight_ep_.find(id);
      if (it == inflight_ep_.end()) return status_t::NOT_FOUND;

      by_ep[it->second].push_back(id);
      inflight_ep_.erase(it);
    }
  }

  for (auto &kv : by_ep) {
    Endpoint *ep = kv.first;
    auto &ids = kv.second;
    if (!ep) return status_t::ERROR;
    if (ids.empty()) continue;

    status_t r = ep->waitWrIdMulti(ids, std::chrono::milliseconds(1000));
    if (r != status_t::SUCCESS) return r;
  }

  return status_t::SUCCESS;
}

status_t Communicator::putPipeline(std::string ip,
                                   uint16_t port,
                                   size_t local_off,
                                   size_t remote_off,
                                   size_t size,
                                   size_t chunk_size,
                                   size_t max_inflight,
                                   ConnType connType) {
  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [local_off, remote_off, size, chunk_size, max_inflight](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        auto *rdma_ep = dynamic_cast<RDMAEndpoint *>(ep);
        if (!rdma_ep) return status_t::UNSUPPORT;
        return rdma_ep->writeDataPipeline(local_off, remote_off, size, chunk_size, max_inflight);
      });
}

status_t Communicator::getPipeline(std::string ip,
                                   uint16_t port,
                                   size_t local_off,
                                   size_t remote_off,
                                   size_t size,
                                   size_t chunk_size,
                                   size_t max_inflight,
                                   ConnType connType) {
  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [local_off, remote_off, size, chunk_size, max_inflight](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        auto *rdma_ep = dynamic_cast<RDMAEndpoint *>(ep);
        if (!rdma_ep) return status_t::UNSUPPORT;
        return rdma_ep->readDataPipeline(local_off, remote_off, size, chunk_size, max_inflight);
      });
}

status_t Communicator::ctrlSend(CtrlId peer, uint64_t tag) {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  return ctrl.sendU64(peer, tag) ? status_t::SUCCESS : status_t::ERROR;
}

status_t Communicator::ctrlRecv(CtrlId peer, uint64_t *tag) {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  uint64_t t = 0;
  if (!ctrl.recvU64(peer, t)) return status_t::ERROR;
  if (tag) *tag = t;
  return status_t::SUCCESS;
}

status_t Communicator::sendDataTo(std::string ip, uint16_t port, void *send_buf, size_t buf_size,
                                  MemoryType buf_type, ConnType connType) {
  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [send_buf, buf_size, buf_type](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        return ep->uhm_send(send_buf, buf_size, buf_type);
      });
}

status_t Communicator::recvDataFrom(std::string ip, uint16_t port, void *recv_buf, size_t buf_size,
                                    MemoryType buf_type, size_t *flag,
                                    ConnType connType) {
  status_t sret = checkConn(ip, port, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, port, connType, [recv_buf, buf_size, flag, buf_type](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        return ep->uhm_recv(recv_buf, buf_size, flag, buf_type);
      });
}

status_t Communicator::initCtrlServer(const std::string &bind_ip, uint16_t tcp_port,
                                      const std::string &uds_path) {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  return ctrl.start(bind_ip, tcp_port, uds_path) ? status_t::SUCCESS : status_t::ERROR;
}

status_t Communicator::closeCtrl() {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  ctrl.stop();
  ctrl.closeAll();
  return status_t::SUCCESS;
}

status_t Communicator::connectCtrl(CtrlId peer_id, CtrlId self_id, const CtrlLink &link) {
  auto &ctrl = hmc::CtrlSocketManager::instance();

  if (link.transport == CtrlTransport::UDS) {
    if (link.uds_path.empty()) return status_t::ERROR;
    return ctrl.connectUds(peer_id, link.uds_path, self_id) ? status_t::SUCCESS
                                                           : status_t::ERROR;
  }

  if (link.ip.empty() || link.port == 0) return status_t::ERROR;
  return ctrl.connectTcp(peer_id, link.ip, link.port, self_id) ? status_t::SUCCESS
                                                               : status_t::ERROR;
}

status_t Communicator::closeCtrlPeer(CtrlId peer_id) {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  ctrl.close(peer_id);
  return status_t::SUCCESS;
}

std::string Communicator::udsPathFor(const std::string &dir, CtrlId peer_id) {
  return hmc::CtrlSocketManager::udsPathFor(dir, peer_id);
}

status_t Communicator::initServer(const std::string &bind_ip,
                                  uint16_t data_port,
                                  uint16_t ctrl_tcp_port,
                                  const std::string &ctrl_uds_path,
                                  ConnType serverType) {
  status_t s = initCtrlServer(bind_ip, ctrl_tcp_port, ctrl_uds_path);
  if (s != status_t::SUCCESS) return s;
  return conn_manager->initiateServer(bind_ip, data_port, serverType);
}

status_t Communicator::closeServer() {
  closeCtrl();
  return conn_manager->stopServer();
}

status_t Communicator::connectTo(CtrlId peer_id,
                                 CtrlId self_id,
                                 const std::string &peer_ip,
                                 uint16_t data_port,
                                 const CtrlLink &ctrl_link,
                                 ConnType connType) {
  if (checkConn(peer_ip, data_port, connType) == status_t::SUCCESS) return status_t::SUCCESS;
                                
  // auto &ctrl = CtrlSocketManager::instance();
  // if (self_id < peer_id) {
    status_t cs = connectCtrl(peer_id, self_id, ctrl_link);
    if (cs != status_t::SUCCESS) return cs;
  // } else {
  //   if (!ctrl.waitPeer(peer_id, 30000)) {
  //     return status_t::TIMEOUT;
  //   }
  // }

  status_t ret = conn_manager->initiateConnectionAsClient(peer_ip, data_port, connType);
  if (ret != status_t::SUCCESS) {
    closeCtrlPeer(peer_id);
    return ret;
  }
  return status_t::SUCCESS;
}

status_t Communicator::disConnect(const std::string &ip, uint16_t port, ConnType connType) {
  if (checkConn(ip, port, connType) == status_t::SUCCESS) {
    conn_manager->_removeEndpoint(ip, port, connType);
  }
  return status_t::SUCCESS;
}

status_t Communicator::checkConn(const std::string &ip, uint16_t port, ConnType connType) {
  return conn_manager->withEndpoint(
      ip, port, connType,
      [&ip, port](Endpoint *ep) -> status_t {
        if (ep) return status_t::SUCCESS;
        logDebug("[Communicator] did not have connection to %s:%u", ip.c_str(), (unsigned)port);
        return status_t::ERROR;
      });
}

} // namespace hmc
