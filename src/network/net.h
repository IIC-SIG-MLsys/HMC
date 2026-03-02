/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_NET_H
#define HMC_NET_H

#include "../utils/log.h"
#include "../utils/signal_handle.h"
#include <hmc.h>

#include <memory>
#include <mutex>
#include <string>
#include <utility> // For std::pair

namespace hmc {

enum class EndpointType { Client, Server };

/*
Endpoint is high level abstract of comm point.
*/
class Endpoint {
public:
  Endpoint() = default;
  virtual ~Endpoint() = default;

  virtual status_t writeData(size_t local_off, size_t remote_off, size_t size) = 0;
  virtual status_t readData(size_t local_off, size_t remote_off, size_t size) = 0;

  // no block interface
  virtual status_t writeDataNB(size_t local_off, size_t remote_off, size_t size,
                               uint64_t *wr_id) = 0;
  virtual status_t readDataNB(size_t local_off, size_t remote_off, size_t size,
                              uint64_t *wr_id) = 0;
  virtual status_t waitWrId(uint64_t wr_id) = 0;
  virtual status_t waitWrIdMulti(const std::vector<uint64_t>& target_wr_ids,
                                     std::chrono::milliseconds timeout) = 0;

  // uhm interface, only for RDMAEndpoint
  virtual status_t uhm_send(void *input_buffer, const size_t send_flags,
                            MemoryType mem_type) = 0;
  virtual status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                            size_t *recv_flags, MemoryType mem_type) = 0;

  virtual status_t closeEndpoint() = 0;

  EndpointType role;
};

/*
Client/Server.
*/
class Client {
public:
  virtual std::unique_ptr<Endpoint> connect(std::string ip, uint16_t port) = 0;
  virtual ~Client() = default;
};

class Server {
public:
  Server(std::shared_ptr<ConnManager> conn_manager)
      : conn_manager(conn_manager) {}
  virtual ~Server() = default;

  // 监听连接请求
  virtual status_t listen(std::string ip, uint16_t port) = 0;
  virtual status_t stopListen() = 0;

protected:
  std::shared_ptr<ConnManager> conn_manager;
};

/*
ConnManager runs a server for recv new Conn and create new QP into a new
Endpoint. It also run active Connect as a Client.
*/
struct ConnTypeHash {
  size_t operator()(const ConnType& t) const noexcept {
    return static_cast<size_t>(t);
  }
};

struct PeerKey {
  std::string ip;
  uint16_t port;

  bool operator==(const PeerKey& o) const noexcept {
    return port == o.port && ip == o.ip;
  }
};

struct PeerKeyHash {
  size_t operator()(const PeerKey& k) const noexcept {
    size_t h1 = std::hash<std::string>{}(k.ip);
    size_t h2 = std::hash<uint16_t>{}(k.port);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
  }
};

class ConnManager : public std::enable_shared_from_this<ConnManager> {
public:
  struct EndpointEntry {
    std::unique_ptr<Endpoint> endpoint;
    std::mutex mutex; // 排他性使用, ep一次只可以被一个调用使用
  };
  using EndpointMap = std::unordered_map<PeerKey, std::shared_ptr<EndpointEntry>, PeerKeyHash>;

  ConnManager(std::shared_ptr<ConnBuffer> buffer, size_t num_chs);
  // shared ptr 不能在构造函数里面shared from this,需要构造完成后，单独初始化
  status_t initiateServer(std::string ip, uint16_t port, ConnType serverType);
  status_t stopServer(); // stop all
  status_t stopServer(ConnType t);

  // 客户端发起的连接操作
  status_t initiateConnectionAsClient(std::string targetIp, uint16_t targetPort,
                                      ConnType clientType);

  // [discard] 返回指针的方式，无法保证对象的并发安全
  // 返回 Endpoint 指针，对象所有权依然在endpoint_map
  // Endpoint *getEndpoint(std::string ip) {
  //   auto it = endpoint_map.find(ip);
  //   if (it == endpoint_map.end()) {
  //     return nullptr;
  //   }
  //   return it->second.get(); // 返回引用
  // }

  // 安全访问接口
  template <typename F> status_t withEndpoint(const std::string &ip, uint16_t port, ConnType type, F &&func) {
    std::shared_ptr<EndpointEntry> entry;
    PeerKey key{ip, port};
    EndpointMap *map = endpointMapFor(type);
    std::mutex *map_mu = endpointMapMutexFor(type);
    if (!map || !map_mu) {
      return status_t::INVALID_CONFIG;
    }

    {
      std::lock_guard<std::mutex> lock(*map_mu);
      auto it = map->find(key);
      if (it == map->end()) {
        return status_t::NOT_FOUND;
      }
      entry = it->second;
    }

    if (!entry) {
      return status_t::ERROR;
    }

    // 第二阶段：锁定并执行操作
    std::lock_guard<std::mutex> entry_lock(entry->mutex);
    if (!entry->endpoint) {
      return status_t::ERROR;
    }

    return func(entry->endpoint.get());
  }

  void _addEndpoint(std::string ip, uint16_t port, std::unique_ptr<Endpoint> endpoint, ConnType type);
  void _removeEndpoint(std::string ip, uint16_t port, ConnType type);
  status_t _removeEndpointIfMatch(std::string ip, uint16_t port,
                                             ConnType connType,
                                             uint64_t expected_conn_id);

  void _printEndpointMap() {
    logInfo("Number of key-value pairs: %lu", rdma_endpoint_map.size()+ucx_endpoint_map.size());
    std::cout << "Keys in the rdma_endpoint_map:" << std::endl;
    for (const auto &pair : rdma_endpoint_map) {
      std::cout << pair.first.ip << " " << pair.first.port << std::endl;
    }
    std::cout << "Keys in the ucx_endpoint_map:" << std::endl;
    for (const auto &pair : ucx_endpoint_map) {
      std::cout << pair.first.ip << " " << pair.first.port << std::endl;
    }
  }

  ~ConnManager();

private:
  EndpointMap *endpointMapFor(ConnType type) {
    switch (type) {
    case ConnType::RDMA:
      return &rdma_endpoint_map;
    case ConnType::UCX:
      return &ucx_endpoint_map;
    default:
      return nullptr;
    }
  }

  std::mutex *endpointMapMutexFor(ConnType type) {
    switch (type) {
    case ConnType::RDMA:
      return &rdma_endpoint_map_mutex;
    case ConnType::UCX:
      return &ucx_endpoint_map_mutex;
    default:
      return nullptr;
    }
  }

  size_t num_chs_ = 1;
  EndpointMap rdma_endpoint_map;
  EndpointMap ucx_endpoint_map;
  std::shared_ptr<ConnBuffer> buffer;
  std::mutex rdma_endpoint_map_mutex;
  std::mutex ucx_endpoint_map_mutex;

  std::mutex server_mu_;
  std::unordered_map<ConnType, std::unique_ptr<Server>, ConnTypeHash> servers_;
  std::unordered_map<ConnType, std::thread, ConnTypeHash> server_threads_;
};

/* RDMA endpoint的主动断开逻辑 */
/* 由于每个端上均有一个server,故可以通过从本地任意一个client端向对方发送断开消息(携带ip端口号)来完成对任意连接的断开
    如果某端没有满足条件的上述EP，则新建一个EP,通过这个EP向对端发送消息，之后再主动销毁本EP。
    即：server端被动建立的EP,如果想从server端主动关闭，则从server端的一个client属性的EP向对端的server发送断开消息
    断开总是由client属性的EP发起的。
 */

} // namespace hmc

#endif
