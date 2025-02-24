/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net.h"
#include "net_rdma.h"
#include "net_ucx.h"

#include <functional>

namespace hddt {

// Factory function to create Server based on configuration
std::unique_ptr<Server>
createServer(ConnType serverType, std::shared_ptr<ConnBuffer> buffer,
             std::shared_ptr<ConnManager> conn_manager) {
  switch (serverType) {
  case ConnType::RDMA:
    return std::make_unique<RDMAServer>(buffer, conn_manager);
  case ConnType::UCX:
    return std::make_unique<UCXServer>(conn_manager); // TODO:
  default:
    throw std::invalid_argument("Unknown server type");
  }
}

ConnManager::ConnManager(std::shared_ptr<ConnBuffer> buffer) : buffer(buffer) {}

status_t ConnManager::initiateServer(std::string ip, uint16_t port,
                                     ConnType serverType) {
  auto conn_manager_shared = shared_from_this();

  // 使用工厂方法创建 Server 实例
  server = createServer(serverType, buffer, conn_manager_shared);

  // 启动一个新的线程运行服务器的监听循环
  server_thread = std::thread([this, ip = std::move(ip), port]() mutable {
    try {
      this->server->listen(ip, port);
    } catch (const std::exception &e) {
      // 处理异常，例如记录错误日志等
      logError("Server listen error: %s", e.what());
    }
  });

  std::this_thread::sleep_for(
      std::chrono::milliseconds(1000)); // we should wait for server start

  return status_t::SUCCESS;
};

status_t ConnManager::initiateConnectionAsClient(std::string targetIp,
                                                 uint16_t targetPort,
                                                 ConnType clientType) {
  std::unique_ptr<Endpoint> endpoint;

  switch (clientType) {
  case ConnType::RDMA: {
    auto client = new RDMAClient(buffer);
    endpoint = client->connect(targetIp, targetPort);
    break;
  }
  case ConnType::UCX: {
    auto client = new UCXClient(buffer);
    endpoint = client->connect(targetIp, targetPort);
    break;
  }
  default:
    return status_t::INVALID_CONFIG;
  }

  if (!endpoint) {
    return status_t::ERROR;
  }

  std::lock_guard<std::mutex> lock(endpoint_map_mutex); // 确保线程安全
  endpoint_map[targetIp] = std::move(endpoint);

  return status_t::SUCCESS;
}

void ConnManager::_addEndpoint(std::string ip,
                               std::unique_ptr<Endpoint> endpoint) {
  if (endpoint) {
    std::lock_guard<std::mutex> lock(endpoint_map_mutex); // 确保线程安全
    endpoint_map[ip] = std::move(endpoint);
  } else {
    logDebug("Get a invalid Endpoint, can not add it to the endpoint_map");
  }
}

void ConnManager::_removeEndpoint(std::string ip) {
  std::lock_guard<std::mutex> lock(endpoint_map_mutex); // 确保线程安全
  std::unique_ptr<Endpoint> ep =
      std::move(endpoint_map[ip]); // 删除键值的时候，必须先移交所有权，才能删除
  this->endpoint_map.erase(ip);
  ep.reset();
  // _printEndpointMap();
}

ConnManager::~ConnManager() {
  if (server_thread.joinable()) {
    server_thread.join(); // 确保线程正确结束
  }
}

} // namespace hddt