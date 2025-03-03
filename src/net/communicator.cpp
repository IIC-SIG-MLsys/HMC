/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "./net.h"
#include <hddt.h>

namespace hddt {

Communicator::Communicator(std::shared_ptr<ConnBuffer> buffer)
    : buffer(buffer) {
  conn_manager = std::make_shared<ConnManager>(buffer);
};

Communicator::~Communicator() {
  logDebug("close communicator");
  conn_manager.reset();
  logDebug("finished");
}

status_t Communicator::writeTo(uint32_t node_rank, size_t ptr_bias, size_t size,
                               ConnType connType) {

  if (checkConn(node_rank, connType) != status_t::SUCCESS) {
    logError("Communicator::connect: endpoint by rank %d does not exist, try "
             "to connect",
             node_rank);

    const std::pair<std::string, uint16_t> *addr_info =
        _getAddrByRank(node_rank);
    auto ip = addr_info->first;
    auto port = addr_info->second;

    if (conn_manager->initiateConnectionAsClient(ip, port, connType) !=
        status_t::SUCCESS) {
      logError("Communicator::connect: connect to %s:%d failed", ip.c_str(),
               port);
      return status_t::ERROR;
    }
  }

  auto addr = _getAddrByRank(node_rank);
  if (addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return status_t::ERROR;
  }
  auto ip = addr->first;

  return conn_manager->withEndpoint(
      ip, [ptr_bias, size](Endpoint *ep) -> status_t {
        if (!ep)
          return status_t::ERROR;
        return ep->writeData(ptr_bias, size); // 传递返回状态
      });
};

status_t Communicator::readFrom(uint32_t node_rank, size_t ptr_bias,
                                size_t size, ConnType connType) {
  if (checkConn(node_rank, connType) != status_t::SUCCESS) {
    logError("Communicator::connect: endpoint by rank %d does not exist, try "
             "to connect",
             node_rank);

    const std::pair<std::string, uint16_t> *addr_info =
        _getAddrByRank(node_rank);
    auto ip = addr_info->first;
    auto port = addr_info->second;

    if (conn_manager->initiateConnectionAsClient(ip, port, connType) !=
        status_t::SUCCESS) {
      logError("Communicator::connect: connect to %s:%d failed", ip.c_str(),
               port);
      return status_t::ERROR;
    }
  }

  auto addr = _getAddrByRank(node_rank);
  if (addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return status_t::ERROR;
  }
  auto ip = addr->first;

  return conn_manager->withEndpoint(
      ip, [ptr_bias, size](Endpoint *ep) -> status_t {
        if (!ep)
          return status_t::ERROR;
        return ep->readData(ptr_bias, size); // 传递返回状态
      });
};

status_t Communicator::connectTo(uint32_t node_rank, ConnType connType) {
  if (checkConn(node_rank, connType) == status_t::SUCCESS) {
    return status_t::SUCCESS;
  }

  auto addr = _getAddrByRank(node_rank);
  if (addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return status_t::ERROR;
  }

  auto ip = addr->first;
  auto port = addr->second;

  // connect to node, create a new Endpoint
  return conn_manager->initiateConnectionAsClient(ip, port, connType);
};

status_t Communicator::initServer(std::string ip, uint16_t port,
                                  ConnType serverType) {
  return conn_manager->initiateServer(ip, port, serverType);
};

status_t Communicator::disConnect(uint32_t node_rank, ConnType connType) {
  auto addr = _getAddrByRank(node_rank);
  if (addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return status_t::ERROR;
  }
  auto ip = addr->first;

  conn_manager->_removeEndpoint(ip);
  return status_t::SUCCESS;
};

status_t Communicator::checkConn(uint32_t node_rank, ConnType connType) {
  auto addr = _getAddrByRank(node_rank);
  if (addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return status_t::ERROR;
  }

  auto ip = addr->first;

  // // check if already have endpoint
  // Endpoint *ep = conn_manager->getEndpoint(ip);
  // if (ep != nullptr) {
  //   logDebug("Communicator::connect: there already have a endpoint by rank
  //   %d",
  //            node_rank);
  //   return status_t::SUCCESS;
  // }

  return conn_manager->withEndpoint(
      addr->first,
      [](Endpoint *ep) -> status_t { // 显式指定返回类型
        return ep ? status_t::SUCCESS : status_t::ERROR;
      });
};

status_t Communicator::addNewRankAddr(uint32_t rank, std::string ip,
                                      uint16_t port) {
  // 使用 operator[] 直接访问或创建键对应的值，并更新它
  rank_addr_map[rank] = std::make_pair(ip, port);
  return status_t::SUCCESS;
}

status_t Communicator::delRankAddr(uint32_t rank) {
  auto it = rank_addr_map.find(rank);
  if (it == rank_addr_map.end()) { // 如果找不到该 rank
    return status_t::NOT_FOUND;
  }
  rank_addr_map.erase(it); // 移除该 rank 对应的条目
  return status_t::SUCCESS;
}

const std::pair<std::string, uint16_t> *
Communicator::_getAddrByRank(uint32_t node_rank) {
  auto it = rank_addr_map.find(node_rank);
  if (it == rank_addr_map.end()) {
    // 如果没有找到对应的 rank，返回 nullptr 表示未找到
    return nullptr;
  }
  // 找到了对应的 rank，返回其地址信息的地址
  return &it->second;
};

// [discard] 采用conn_manager的withEndpoint管理ep生命周期
// Endpoint *Communicator::_getEndpointByRank(uint32_t node_rank) {
//   auto addr = _getAddrByRank(node_rank);
//   if (addr == nullptr) {
//     logError("Communicator::connect: can't get addr by rank %d", node_rank);
//     return nullptr;
//   }

//   auto ip = addr->first;

//   // check if already have endpoint
//   Endpoint *ep = conn_manager->getEndpoint(ip);
//   if (ep != nullptr) {
//     logDebug("Communicator::connect: get endpoint by rank %d success",
//              node_rank);
//     return ep;
//   }

//   logDebug("Communicator::connect: get endpoint by rank %d failed",
//   node_rank); return nullptr;
// };

/* ConnBuffer */

ConnBuffer::ConnBuffer(int device_id, size_t buffer_size, MemoryType mem_type)
    : buffer_size(buffer_size) {
  mem_ops = new Memory(1, mem_type);
  mem_ops->allocatePeerableBuffer(&ptr, buffer_size);
};

ConnBuffer::~ConnBuffer() {
  mem_ops->freeBuffer(ptr);
  mem_ops->free();
}

// 从CPU向ConnBuffer写入数据
status_t ConnBuffer::writeFromCpu(void *src, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("Invalid data bias and size");
    return status_t::ERROR;
  }
  return mem_ops->copyHostToDevice(static_cast<char *>(ptr) + bias, src, size);
}

// 从ConnBuffer读取数据到CPU
status_t ConnBuffer::readToCpu(void *dest, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    memcpy(dest, static_cast<char *>(ptr), size);
    return status_t::SUCCESS;
  }
  return mem_ops->copyDeviceToHost(dest, static_cast<char *>(ptr) + bias, size);
}

// 从GPU向ConnBuffer写入数据
status_t ConnBuffer::writeFromGpu(void *src, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    mem_ops->copyDeviceToHost(static_cast<char *>(ptr) + bias, src, size);
    return status_t::SUCCESS;
  }
  return mem_ops->copyDeviceToDevice(static_cast<char *>(ptr) + bias, src,
                                     size);
}

// 从ConnBuffer读取数据到GPU
status_t ConnBuffer::readToGpu(void *dest, size_t size, size_t bias) {
  if (bias + size > buffer_size) {
    logError("Invalid data bias and size");
    return status_t::ERROR;
  }
  if (mem_ops->getMemoryType() == MemoryType::CPU) {
    mem_ops->copyHostToDevice(dest, static_cast<char *>(ptr) + bias, size);
    return status_t::SUCCESS;
  }
  return mem_ops->copyDeviceToDevice(dest, static_cast<char *>(ptr) + bias,
                                     size);
}

} // namespace hddt
