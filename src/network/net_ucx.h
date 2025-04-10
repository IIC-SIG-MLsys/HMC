/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_NET_UCX_H
#define HMC_NET_UCX_H

#include "net.h"

#define ucp_conn_h int // TODO: support UCX
#define ucp_ep_h int

namespace hmc {

class UCXEndpoint : public Endpoint {
public:
  UCXEndpoint(ucp_conn_h ucx_context, ucp_ep_h ucx_connection);

  status_t writeData(size_t data_bias, size_t size) override;
  status_t readData(size_t data_bias, size_t size) override;

  status_t writeDataNB(size_t data_bias, size_t size) override;
  status_t readDataNB(size_t data_bias, size_t size) override;
  status_t pollCompletion(int num_completions_to_process) override;

  status_t uhm_send(void *input_buffer, const size_t send_flags, MemoryType mem_type) override;
  status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                      size_t *recv_flags, MemoryType mem_type) override;

  status_t closeEndpoint() override;

  ~UCXEndpoint();

private:
  ucp_conn_h ucx_context;
  ucp_ep_h ucx_connection;
};

class UCXServer : public Server {
public:
  UCXServer(std::shared_ptr<ConnManager> conn_manager);

  status_t listen(std::string ip, uint16_t port) override;
  status_t stopListen() override;
  std::unique_ptr<Endpoint> handleConnection(std::string ip, uint16_t port);

  ~UCXServer();

private:
  std::string ip = "0.0.0.0";
  uint16_t port = 1234;
};

class UCXClient {
public:
  UCXClient(std::shared_ptr<ConnBuffer> buffer);
  std::unique_ptr<Endpoint> connect(std::string ip, uint16_t port);

  ~UCXClient();

private:
  std::shared_ptr<ConnBuffer> buffer;
};

} // namespace hmc
#endif