/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_ucx.h"

namespace hddt {

UCXClient::UCXClient(std::shared_ptr<ConnBuffer> buffer) : buffer(buffer) {}
UCXClient::~UCXClient(){}

std::unique_ptr<Endpoint> UCXClient::connect(const std::string& ip, uint16_t port) {
    // 模拟连接过程
    std::cout << "Connecting to " << ip << ":" << port << " using UCX..." << std::endl;

    // 假设连接成功，创建一个新的 RdmaEndpoint 实例
    ucp_conn_h ucx_context = 0;
    ucp_ep_h ucx_connection = 0; // fake value
    auto endpoint = std::make_unique<UCXEndpoint>(ucx_context, ucx_connection);

    // 返回 unique_ptr 给调用者
    return endpoint;
}

}