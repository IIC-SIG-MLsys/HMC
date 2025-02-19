/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_NET_UCX_H
#define HDDT_NET_UCX_H

#include "net.h"

#define ucp_conn_h int // TODO: support UCX
#define ucp_ep_h int 

namespace hddt {

class UCXEndpoint : public Endpoint {
public:
  UCXEndpoint(ucp_conn_h ucx_context, ucp_ep_h ucx_connection);

  status_t sendData(const void* data, size_t size) override;
  status_t recvData(size_t* flag) override;
  status_t closeEndpoint() override;

  ~UCXEndpoint();
private:
  ucp_conn_h ucx_context;
  ucp_ep_h ucx_connection;
};

class UCXServer : public Server {
public:
  UCXServer(std::shared_ptr<ConnManager> conn_manager);

  status_t listen(const std::string& ip, uint16_t port) override;
  std::unique_ptr<Endpoint> handleConnection(const std::string& ip, uint16_t port);

  ~UCXServer();
private:
  std::string ip = "0.0.0.0";
  uint16_t port = 1234;
};


class UCXClient {
public:
    UCXClient(std::shared_ptr<ConnBuffer> buffer);
    std::unique_ptr<Endpoint> connect(const std::string& ip, uint16_t port);

    ~UCXClient();
private:
    std::shared_ptr<ConnBuffer> buffer;
};

}
#endif