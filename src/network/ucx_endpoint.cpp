/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_ucx.h"

namespace hmc {

UCXEndpoint::UCXEndpoint(ucp_conn_h ucx_context, ucp_ep_h ucx_connection)
    : ucx_context(ucx_context), ucx_connection(ucx_connection) {}

status_t UCXEndpoint::writeData(size_t data_bias, size_t size) {
  // 实现发送数据逻辑
  return status_t::SUCCESS;
}

status_t UCXEndpoint::readData(size_t data_bias, size_t size) {
  // 实现接收数据逻辑
  return status_t::SUCCESS;
}

status_t UCXEndpoint::closeEndpoint() { return status_t::SUCCESS; };

UCXEndpoint::~UCXEndpoint() { closeEndpoint(); }

} // namespace hmc