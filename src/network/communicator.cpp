/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "./net.h"
#include <hmc.h>

namespace hmc {

Communicator::Communicator(std::shared_ptr<ConnBuffer> buffer, size_t num_chs)
    : buffer(buffer) {
  conn_manager = std::make_shared<ConnManager>(buffer, num_chs);
};

Communicator::~Communicator() {
  logDebug("close communicator");
  conn_manager.reset();
  logDebug("finished");
}

status_t Communicator::put(std::string ip,
                           size_t local_off,
                           size_t remote_off,
                           size_t size,
                           ConnType connType) {
  status_t sret = checkConn(ip, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, connType, [local_off, remote_off, size](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        return ep->writeData(local_off, remote_off, size);
      });
}

status_t Communicator::get(std::string ip,
                           size_t local_off,
                           size_t remote_off,
                           size_t size,
                           ConnType connType) {
  status_t sret = checkConn(ip, connType);
  if (sret != status_t::SUCCESS) return sret;

  return conn_manager->withEndpoint(
      ip, connType, [local_off, remote_off, size](Endpoint *ep) -> status_t {
        if (!ep) return status_t::ERROR;
        return ep->readData(local_off, remote_off, size);
      });
}

status_t Communicator::putNB(std::string ip,
                             size_t local_off,
                             size_t remote_off,
                             size_t size,
                             uint64_t *wr_id,
                             ConnType connType) {
  if (wr_id) *wr_id = 0;

  status_t sret = checkConn(ip, connType);
  if (sret != status_t::SUCCESS) return sret;

  status_t ret = conn_manager->withEndpoint(
      ip, connType, [this, local_off, remote_off, size, wr_id](Endpoint *ep) -> status_t {
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

  return ret;
}

status_t Communicator::getNB(std::string ip,
                             size_t local_off,
                             size_t remote_off,
                             size_t size,
                             uint64_t *wr_id,
                             ConnType connType) {
  if (wr_id) *wr_id = 0;

  status_t sret = checkConn(ip, connType);
  if (sret != status_t::SUCCESS) return sret;

  status_t ret = conn_manager->withEndpoint(
      ip, connType, [this, local_off, remote_off, size, wr_id](Endpoint *ep) -> status_t {
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

  return ret;
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

status_t Communicator::wait(const std::vector<uint64_t>& wr_ids) {
  std::unordered_map<Endpoint*, std::vector<uint64_t>> by_ep;
  by_ep.reserve(8);

  {
    std::lock_guard<std::mutex> lk(inflight_mu_);
    for (uint64_t id : wr_ids) {
      if (id == 0) continue;

      auto it = inflight_ep_.find(id);
      if (it == inflight_ep_.end()) {
        return status_t::NOT_FOUND;
      }
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

status_t Communicator::ctrlSend(std::string ip, uint64_t tag) {
  // use rdma send/recv  or  ucx tagMsg later
  auto &ctrl = hmc::CtrlSocketManager::instance();
  if (ctrl.sendCtrlU64(ip, tag)) return status_t::SUCCESS;
  else return status_t::ERROR;
}

status_t Communicator::ctrlRecv(std::string ip, uint64_t *tag) {
  // use rdma send/recv  or  ucx tagMsg later
  auto &ctrl = hmc::CtrlSocketManager::instance();
  uint64_t t = 0;
  if (!ctrl.recvCtrlU64(ip, t)) return status_t::ERROR;
  if (tag) *tag = t;
  return status_t::SUCCESS;
}

status_t Communicator::sendDataTo(std::string ip, void *send_buf,
                                  size_t buf_size, MemoryType buf_type,
                                  ConnType connType) {
  status_t sret = checkConn(ip, connType);
  if (sret != status_t::SUCCESS) {
    return sret;
  }

  return conn_manager->withEndpoint(
      ip, connType, [send_buf, buf_size, buf_type](Endpoint *ep) -> status_t {
        if (!ep)
          return status_t::ERROR;
        return ep->uhm_send(send_buf, buf_size, buf_type);
      });
};

status_t Communicator::recvDataFrom(std::string ip, void *recv_buf,
                                    size_t buf_size, MemoryType buf_type,
                                    size_t *flag, ConnType connType) {
  status_t sret = checkConn(ip, connType);
  if (sret != status_t::SUCCESS) {
    return sret;
  }

  return conn_manager->withEndpoint(
      ip, connType, [recv_buf, buf_size, flag, buf_type](Endpoint *ep) -> status_t {
        if (!ep)
          return status_t::ERROR;
        return ep->uhm_recv(recv_buf, buf_size, flag, buf_type);
      });
};

status_t Communicator::connectTo(std::string ip, uint16_t port,
                                 ConnType connType) {
  if (checkConn(ip, connType) == status_t::SUCCESS) {
    return status_t::SUCCESS;
  }

  auto &ctrl = hmc::CtrlSocketManager::instance();
  int ctrl_fd = ctrl.getCtrlSockFd(ip); // 客户端主动连接
  if (ctrl_fd < 0) {
    std::cerr << "[Communicator] Failed to connect control channel to " << ip
              << ":" << ctrl.port() << "\n";
    return status_t::ERROR;
  }

  // connect to node, create a new Endpoint
  auto ret = conn_manager->initiateConnectionAsClient(ip, port, connType);
  if (ret != status_t::SUCCESS)
    ctrl.closeConnection(ip);
  return ret;
};

status_t Communicator::initServer(std::string ip, uint16_t port,
                                  ConnType serverType) {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  ctrl.startServer(ip); // only first time
  return conn_manager->initiateServer(ip, port, serverType);
};

status_t Communicator::closeServer() {
  auto &ctrl = hmc::CtrlSocketManager::instance();
  ctrl.stopServer();
  return conn_manager->stopServer();
};

status_t Communicator::disConnect(std::string ip, ConnType connType) {
  if (checkConn(ip, connType) == status_t::SUCCESS) {
    conn_manager->_removeEndpoint(ip, connType);
  }
  return status_t::SUCCESS;
};

status_t Communicator::checkConn(std::string ip, ConnType connType) {
  return conn_manager->withEndpoint(
      ip, connType,
      [](Endpoint *ep) -> status_t {
        return ep ? status_t::SUCCESS : status_t::ERROR;
      });
};

// [discard] 现在采用conn_manager的withEndpoint管理ep生命周期
// Endpoint *Communicator::_getEndpointByRank(uint32_t ip) {
//   auto addr = _getPortByIp(ip);
//   if (addr == nullptr) {
//     logError("Communicator::connect: can't get addr by ip %s", ip);
//     return nullptr;
//   }

//   auto ip = addr->first;

//   // check if already have endpoint
//   Endpoint *ep = conn_manager->getEndpoint(ip);
//   if (ep != nullptr) {
//     logDebug("Communicator::connect: get endpoint by ip %s success",
//              ip);
//     return ep;
//   }

//   logDebug("Communicator::connect: get endpoint by ip %s failed",
//   ip); return nullptr;
// };

/* ConnBuffer */
ConnBuffer::ConnBuffer(int device_id, size_t buffer_size, MemoryType mem_type)
    : buffer_size(buffer_size) {
  mem_ops = new Memory(device_id, mem_type);
  mem_ops->allocatePeerableBuffer(&ptr, buffer_size);
};

ConnBuffer::~ConnBuffer() {
  mem_ops->freeBuffer(ptr);
  mem_ops->free();
}

// 从CPU向ConnBuffer写入数据
status_t ConnBuffer::writeFromCpu(void *src, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("writeFromCpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  return mem_ops->copyHostToDevice(static_cast<char *>(ptr) + bias, src, size);
}

// 从ConnBuffer读取数据到CPU
status_t ConnBuffer::readToCpu(void *dest, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("readToCpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    memcpy(dest, static_cast<char *>(ptr) + bias, size);
    return status_t::SUCCESS;
  }
  return mem_ops->copyDeviceToHost(dest, static_cast<char *>(ptr) + bias, size);
}

// 从GPU向ConnBuffer写入数据
status_t ConnBuffer::writeFromGpu(void *src, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("writeFromGpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    logError("Error write data from GPU to CPU using CPU ConnBuffer, Please "
             "use gpu mem_ops.");
    return status_t::ERROR;
  }
  return mem_ops->copyDeviceToDevice(static_cast<char *>(ptr) + bias, src,
                                     size);
}

// 从ConnBuffer读取数据到GPU
status_t ConnBuffer::readToGpu(void *dest, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("readToGpu: Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    logError("Error read data from CPU to GPU using CPU ConnBuffer, Please use "
             "gpu mem_ops.");
    return status_t::ERROR;
  }
  return mem_ops->copyDeviceToDevice(dest, static_cast<char *>(ptr) + bias,
                                     size);
}

status_t ConnBuffer::copyWithin(size_t dst_bias, size_t src_bias, size_t size) {
  if (dst_bias + size > buffer_size || src_bias + size > buffer_size) {
    logError("copyWithin: Invalid bias/size");
    return status_t::ERROR;
  }
  if (size == 0 || dst_bias == src_bias) {
    return status_t::SUCCESS;
  }

  void* dst = static_cast<char*>(ptr) + dst_bias;
  void* src = static_cast<char*>(ptr) + src_bias;

  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    // overlap-safe
    memmove(dst, src, size);
    return status_t::SUCCESS;
  }

  // device-to-device copy on the same device (CUDA/ROCm/MLU/Moore)
  return mem_ops->copyDeviceToDevice(dst, src, size);
}

} // namespace hmc
