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
  return status_t::SUCCESS;
}

status_t UCXEndpoint::recvData(size_t data_bias, size_t size) {
  return status_t::SUCCESS;
}

status_t UCXEndpoint::writeDataNB(size_t data_bias, size_t size){
  return status_t::SUCCESS;
}

status_t UCXEndpoint::readDataNB(size_t data_bias, size_t size){
  return status_t::SUCCESS;
}

status_t UCXEndpoint::recvDataNB(size_t data_bias, size_t size){
  return status_t::SUCCESS;
}

status_t UCXEndpoint::pollCompletion(int num_completions_to_process){
  return status_t::SUCCESS;
}

status_t UCXEndpoint::uhm_send(void *input_buffer, const size_t send_flags, MemoryType mem_type) {
  return status_t::UNSUPPORT;
}

status_t UCXEndpoint::uhm_recv(void *output_buffer, const size_t buffer_size,
                      size_t *recv_flags, MemoryType mem_type) {
  return status_t::UNSUPPORT;
}

status_t UCXEndpoint::closeEndpoint() { return status_t::SUCCESS; };

UCXEndpoint::~UCXEndpoint() { closeEndpoint(); }

} // namespace hmc