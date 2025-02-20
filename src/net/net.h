/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_NET_H
#define HDDT_NET_H

#include <hddt.h>
#include "../utils/log.h"
#include "../utils/signal_handle.h"

#include <string>
#include <utility> // For std::pair
#include <mutex>

namespace hddt {

enum class EndpointType { Client, Server };

/*
Endpoint is high level abstract of comm point.
*/
class Endpoint {
public:
    Endpoint() = default;
    virtual ~Endpoint() = default;

    virtual status_t writeData(size_t data_bias, size_t size) = 0;
    virtual status_t readData(size_t data_bias, size_t size) = 0;
    virtual status_t closeEndpoint() = 0;

    EndpointType role;
};


/*
Client/Server.
*/
class Client {
public:
  virtual std::unique_ptr<Endpoint> connect(const std::string& ip, uint16_t port) = 0;
  virtual ~Client() = default;
};

class Server {
public:
    Server(std::shared_ptr<ConnManager> conn_manager) : conn_manager(conn_manager) {}
    virtual ~Server() = default;

    // 监听连接请求
    virtual status_t listen(const std::string& ip, uint16_t port) = 0;
protected:
    std::shared_ptr<ConnManager> conn_manager;
};

/*
ConnManager runs a server for recv new Conn and create new QP into a new Endpoint.
It also run active Connect as a Client.
*/
struct PairHash {
    std::size_t operator()(const std::pair<std::string, uint16_t>& p) const {
        return std::hash<std::string>()(p.first) ^ (std::hash<uint16_t>()(p.second) << 1);
    }
};

class ConnManager : public std::enable_shared_from_this<ConnManager> {
public:
  ConnManager(std::shared_ptr<ConnBuffer> buffer);
  // shared ptr 不能在构造函数里面shared from this,需要构造完成后，单独初始化
  status_t initiateServer(std::string ip, uint16_t port, ConnType serverType);

  // 客户端发起的连接操作
  status_t initiateConnectionAsClient(const std::string& targetIp, uint16_t targetPort, ConnType clientType);

  // 返回 Endpoint 指针，对象所有权依然在endpoint_map
  Endpoint* getEndpoint(const std::string& ip, uint16_t port) {
      auto key = std::make_pair(ip, port);
      auto it = endpoint_map.find(key);
      if (it == endpoint_map.end()) {
          return nullptr;
      }
      return it->second.get(); // 返回引用
  }

  void _addEndpoint(const std::string& ip, uint16_t port, std::unique_ptr<Endpoint> endpoint);
  void _removeEndpoint(const std::string& ip, uint16_t port);

  ~ConnManager();

private:
  std::unordered_map<std::pair<std::string, uint16_t>, std::unique_ptr<Endpoint>, PairHash> endpoint_map;
  std::shared_ptr<ConnBuffer> buffer;
  std::mutex endpoint_map_mutex; // 用于保护对 endpoint_map 的访问

  std::unique_ptr<Server> server;
  std::thread server_thread; // 用于运行服务器监听循环的线程
};

/* RDMA endpoint的主动断开逻辑 */
/* 由于每个端上均有一个server,故可以通过从本地任意一个client端向对方发送断开消息(携带ip端口号)来完成对任意连接的断开
    如果某端没有满足条件的上述EP，则新建一个EP,通过这个EP向对端发送消息，之后再主动销毁本EP。
    即：server端被动建立的EP,如果想从server端主动关闭，则从server端的一个client属性的EP向对端的server发送断开消息
    断开总是由client属性的EP发起的。
 */

}

#endif