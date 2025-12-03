#include "net_ucx.h"

#include "../utils/log.h"

namespace hmc {

UCXServer::UCXServer(std::shared_ptr<ConnManager> conn_manager)
    : Server(std::move(conn_manager)) {}

UCXServer::~UCXServer() { stopListen(); }

status_t UCXServer::listen(std::string ip, uint16_t port) {
  ip_ = std::move(ip);
  port_ = port;

  if (running_) {
    return status_t::SUCCESS;
  }

  // 启动 TCP 控制面 server，供 CtrlSocketManager 使用
  auto &ctrl = CtrlSocketManager::instance();
  if (!ctrl.startServer(ip_, port_ + 1)) {
    logError("UCXServer::listen: startServer failed on %s:%u",
             ip_.c_str(), port_ + 1);
    return status_t::ERROR;
  }

  running_ = true;
  logInfo("UCXServer listening (CtrlSocket) on %s:%u",
          ip_.c_str(), port_);
  return status_t::SUCCESS;
}

status_t UCXServer::stopListen() {
  if (!running_) {
    return status_t::SUCCESS;
  }
  running_ = false;

  auto &ctrl = CtrlSocketManager::instance();
  ctrl.stopServer();

  logInfo("UCXServer::stopListen: stopped");
  return status_t::SUCCESS;
}

// 目前 UCX 的连接完全由 UCXClient 在各端主动建立，Server 不主动创建 Endpoint。
// 这里保留接口，返回 nullptr。
std::unique_ptr<Endpoint>
UCXServer::handleConnection(std::string /*ip*/, uint16_t /*port*/) {
  return nullptr;
}

} // namespace hmc
