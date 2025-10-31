/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net.h"
#include "net_rdma.h"
#include "net_ucx.h"

#include <functional>

namespace hmc {

// Factory function to create Server based on configuration
std::unique_ptr<Server>
createServer(ConnType serverType, std::shared_ptr<ConnBuffer> buffer,
             std::shared_ptr<ConnManager> conn_manager, size_t num_chs) {
  switch (serverType) {
  case ConnType::RDMA:
    return std::make_unique<RDMAServer>(buffer, conn_manager, num_chs);
  case ConnType::UCX:
    return std::make_unique<UCXServer>(conn_manager); // TODO:
  default:
    throw std::invalid_argument("Unknown server type");
  }
}

ConnManager::ConnManager(std::shared_ptr<ConnBuffer> buffer, size_t num_chs) : buffer(buffer), num_chs_(num_chs) {}

status_t ConnManager::initiateServer(std::string ip, uint16_t port,
                                     ConnType serverType) {
  auto conn_manager_shared = shared_from_this();

  // 使用工厂方法创建 Server 实例
  server = createServer(serverType, buffer, conn_manager_shared, num_chs_);

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

status_t ConnManager::stopServer() {
  if (server) {
    server->stopListen();
    // 等待 server thread 退出
    if (server_thread.joinable()) {
      server_thread.join();
    }
  }

  return status_t::SUCCESS;
};

status_t ConnManager::initiateConnectionAsClient(std::string targetIp,
                                                 uint16_t targetPort,
                                                 ConnType clientType) {
  std::unique_ptr<Endpoint> endpoint;

  switch (clientType) {
  case ConnType::RDMA: {
    auto client = new RDMAClient(buffer, num_chs_);
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
  auto &entry = endpoint_map[targetIp];
  // std::lock_guard<std::mutex> entry_lock(entry.mutex); //
  // 必是单独访问,不需要锁
  entry.endpoint = std::move(endpoint);

  return status_t::SUCCESS;
}

void ConnManager::_addEndpoint(std::string ip,
                               std::unique_ptr<Endpoint> endpoint) {
  if (endpoint) {
    std::lock_guard<std::mutex> lock(endpoint_map_mutex); // 确保线程安全
    auto &entry = endpoint_map[ip];
    // std::lock_guard<std::mutex> entry_lock(entry.mutex);  //
    // 必是单独访问,不需要锁
    entry.endpoint = std::move(endpoint);
  } else {
    logDebug("Get a invalid Endpoint, can not add it to the endpoint_map");
  }
}

void ConnManager::_removeEndpoint(std::string ip) {
  // std::lock_guard<std::mutex> lock(endpoint_map_mutex); // 确保线程安全
  // std::unique_ptr<Endpoint> ep = std::move(endpoint_map[ip]); //
  // 删除键值的时候，必须先移交所有权，才能删除 this->endpoint_map.erase(ip);
  // ep.reset();
  // // _printEndpointMap();

  std::unique_ptr<Endpoint> to_delete;
  // 两阶段删除保证
  {
    std::lock_guard<std::mutex> lock(endpoint_map_mutex);
    auto it = endpoint_map.find(ip);
    if (it == endpoint_map.end())
      return;
    // 锁定条目后转移所有权
    // 强制删除ep时,无需锁,因为一方已经决定要断开了
    to_delete = std::move(it->second.endpoint);
    endpoint_map.erase(it);
  }
  // 在此处同步等待资源释放
  to_delete.reset(); // 显式的关闭ep
  return;
}

ConnManager::~ConnManager() {
  if (server_thread.joinable()) {
    server_thread.join(); // 确保线程正确结束
  }
}

} // namespace hmc