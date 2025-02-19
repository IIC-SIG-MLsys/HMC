/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_ucx.h"

namespace hddt {

UCXServer::UCXServer(std::shared_ptr<ConnManager> conn_manager) : Server(conn_manager) {}
UCXServer::~UCXServer(){}

status_t UCXServer::listen(const std::string& ip, uint16_t port) {
    // 实现监听逻辑
    logInfo("UCX Server is listening on %s:%d", ip.c_str(), port);
    // 这里应该有循环等待新的连接请求，并调用 handleNewConnection
    return status_t::SUCCESS;
}

std::unique_ptr<Endpoint> UCXServer::handleConnection(const std::string& ip, uint16_t port) {
    // 模拟获取新的 ucx_context 和 ucx_connection
    ucp_conn_h new_ucx_context = 0; // 示例值
    ucp_ep_h new_ucx_connection = 0; // 示例值

    return std::make_unique<UCXEndpoint>(new_ucx_context, new_ucx_connection);
}

}