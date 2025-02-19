/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include <hddt.h>
#include "./net/net.h"

namespace hddt{

/*
  Factory Function for create a Communicator
*/
// [[nodiscard]] std::unique_ptr<Communicator>
// CreateCommunicator(Memory *mem_op, CommunicatorType comm_type, bool is_server,
//                    bool is_client, std::string client_ip, uint16_t client_port,
//                    std::string server_ip, uint16_t server_port, int retry_times,
//                    int retry_delay_time) {
//   if (comm_type == CommunicatorType::DEFAULT) {
//     if (support_rdma()) {
//       comm_type = CommunicatorType::RDMA;
//     } else {
//       comm_type = CommunicatorType::TCP;
//     }
//   }

//   switch (comm_type) {
//   case CommunicatorType::RDMA:
//     return std::make_unique<RDMACommunicator>(
//         mem_op, is_server, is_client, client_ip, client_port, server_ip,
//         server_port, retry_times, retry_delay_time);
//   case CommunicatorType::TCP:
//     return std::make_unique<TCPCommunicator>(
//         mem_op, is_server, is_client, client_ip, client_port, server_ip,
//         server_port, retry_times, retry_delay_time);
//   default:
//     return nullptr;
//   }
// }

Communicator::Communicator(std::shared_ptr<ConnBuffer> buffer) : buffer(buffer) {
  conn_manager = std::make_shared<ConnManager>(buffer);
};

Communicator::~Communicator() {
  logDebug("close communicator");
  conn_manager.reset();
  logDebug("finished");
}

status_t Communicator::sendData(uint32_t node_rank, size_t ptr_bias, size_t size) {
  // TODO:
  auto ep = _getEndpointByRank(node_rank);
  if(ep == nullptr) {
    logError("Communicator::connect: endpoint by rank %d does not exist", node_rank);
    return status_t::ERROR;
  }
  char* ptr = static_cast<char*>(buffer->ptr) + (ptr_bias/sizeof(char));
  return ep->sendData(ptr, size);
};

status_t Communicator::recvData(uint32_t node_rank, size_t* recv_flag) {
  // TDDO:
  auto ep = _getEndpointByRank(node_rank);
  if(ep == nullptr) {
    logError("Communicator::connect: endpoint by rank %d does not exist", node_rank);
    return status_t::ERROR;
  }

  return ep->recvData(recv_flag);
};

status_t Communicator::connectTo(uint32_t node_rank, ConnType connType){
  auto addr = _getAddrByRank(node_rank);
  if(addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return status_t::ERROR;
  }

  auto ip = addr->first;
  auto port = addr->second;

  // check if already have endpoint
  Endpoint* ep = conn_manager->getEndpoint(ip, port);
  if(ep != nullptr) {
    logDebug("Communicator::connect: there already have a endpoint by rank %d", node_rank);
    return status_t::SUCCESS;
  }

  // connect to node, create a new Endpoint
  return conn_manager->initiateConnectionAsClient(ip, port, connType);
};

status_t Communicator::initServer(std::string ip, uint16_t port, ConnType serverType){
  return conn_manager->initiateServer(ip, port, serverType);
};

status_t Communicator::addNewRankAddr(uint32_t rank, std::string ip, uint16_t port) {
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

const std::pair<std::string, uint16_t>* Communicator::_getAddrByRank(uint32_t node_rank) {
    auto it = rank_addr_map.find(node_rank);
    if (it == rank_addr_map.end()) {
        // 如果没有找到对应的 rank，返回 nullptr 表示未找到
        return nullptr;
    }
    // 找到了对应的 rank，返回其地址信息的地址
    return &it->second;
};

Endpoint* Communicator::_getEndpointByRank(uint32_t node_rank) {
  auto addr = _getAddrByRank(node_rank);
  if(addr == nullptr) {
    logError("Communicator::connect: can't get addr by rank %d", node_rank);
    return nullptr;
  }

  auto ip = addr->first;
  auto port = addr->second;

  // check if already have endpoint
  Endpoint* ep = conn_manager->getEndpoint(ip, port);
  if(ep != nullptr) {
    logDebug("Communicator::connect: get endpoint by rank %d success", node_rank);
    return ep;
  }

  logDebug("Communicator::connect: get endpoint by rank %d failed", node_rank);
  return nullptr;
};

}


