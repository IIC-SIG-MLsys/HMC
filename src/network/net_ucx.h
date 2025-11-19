/**
 * @file net_ucx.h
 * @brief UCX-based network backend for HMC (TAG-based UHM implementation).
 *
 * 只实现 UHM 接口（uhm_send/uhm_recv），基于 UCX TAG API：
 *   - ucp_tag_send_nbx
 *   - ucp_tag_recv_nbx
 *
 * 不再做 RMA / ucp_mem_map / rkey。
 */

#ifndef HMC_NET_UCX_H
#define HMC_NET_UCX_H

#include "net.h"

#include <ucp/api/ucp.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace hmc {

/**
 * @class UCXContext
 * @brief 封装 UCX context / worker 资源。
 *
 * 这里只开启 TAG 特性，用于实现 uhm_send/uhm_recv。
 */
class UCXContext {
public:
  UCXContext();
  ~UCXContext();

  UCXContext(const UCXContext &) = delete;
  UCXContext &operator=(const UCXContext &) = delete;

  // 初始化 UCX：ucp_init + ucp_worker_create
  status_t init();

  // 显式释放（析构时也会调用）
  void finalize();

  ucp_context_h context() const { return context_; }
  ucp_worker_h worker() const { return worker_; }

  // 打包本端 worker 地址，用于通过 CtrlSocketManager 发送给对端
  status_t packWorkerAddress(std::vector<std::uint8_t> &out);

  // 推进 worker 进度
  void progress();

private:
  ucp_context_h context_{nullptr};
  ucp_worker_h worker_{nullptr};

  std::mutex worker_mutex_;
  bool initialized_{false};
};

/**
 * @class UCXEndpoint
 * @brief UCX 实现的 Endpoint，使用 TAG API 实现 UHM。
 *
 * - writeData / readData / *NB 接口暂不支持，直接返回 UNSUPPORT。
 * - uhm_send / uhm_recv 使用 ucp_tag_send_nbx / ucp_tag_recv_nbx。
 */
class UCXEndpoint : public Endpoint {
public:
  UCXEndpoint(std::shared_ptr<UCXContext> ctx, ucp_ep_h ep);
  ~UCXEndpoint() override;

  // RDMA-style 接口暂不支持
  status_t writeData(size_t, size_t) override { return status_t::UNSUPPORT; }
  status_t readData(size_t, size_t) override { return status_t::UNSUPPORT; }

  status_t writeDataNB(size_t, size_t, uint64_t *) override {
    return status_t::UNSUPPORT;
  }
  status_t readDataNB(size_t, size_t, uint64_t *) override {
    return status_t::UNSUPPORT;
  }
  status_t waitWrId(uint64_t) override { return status_t::UNSUPPORT; }

  // UHM：TAG 发送/接收
  status_t uhm_send(void *input_buffer, const size_t send_size,
                    MemoryType mem_type) override;
  status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                    size_t *recv_flags, MemoryType mem_type) override;

  status_t closeEndpoint() override;

private:
  status_t waitRequest(void *request);

  std::shared_ptr<UCXContext> ctx_;
  ucp_ep_h ep_{nullptr};

  static constexpr ucp_tag_t kTag = 0xCAFEu;
};

/**
 * @class UCXServer
 * @brief UCX 后端的 Server，对应 createServer 中的 UCX 分支。
 *
 * 对 UCX 来说，这里主要负责：
 *  - 启动 CtrlSocketManager 的 TCP server（给 UCXClient 用来做 worker 地址握手）
 * 实际的 UCX endpoint 创建由 UCXClient 完成。
 */
class UCXServer : public Server {
public:
  explicit UCXServer(std::shared_ptr<ConnManager> conn_manager);

  status_t listen(std::string ip, uint16_t port) override;
  status_t stopListen() override;

  // 保留接口，目前不使用
  std::unique_ptr<Endpoint> handleConnection(std::string ip, uint16_t port);

  ~UCXServer() override;

private:
  std::string ip_{"0.0.0.0"};
  uint16_t port_{0};
  bool running_{false};
};

/**
 * @class UCXClient
 * @brief UCX 后端的 Client，实现 ConnManager::initiateConnectionAsClient 中的 UCX 分支。
 *
 * 注意：
 *  - 构造函数签名必须保持 UCXClient(std::shared_ptr<ConnBuffer> buffer)
 *  - 实际上这里不依赖 buffer，只是为了和现有接口兼容
 *  - 进程内共享一个 UCXContext（global_ctx_）
 */
class UCXClient : public Client {
public:
  explicit UCXClient(std::shared_ptr<ConnBuffer> buffer);
  ~UCXClient() override;

  std::unique_ptr<Endpoint> connect(std::string ip,
                                    uint16_t port) override;

private:
  std::shared_ptr<ConnBuffer> buffer_;
  std::shared_ptr<UCXContext> ctx_;

  // 进程内共享 UCXContext
  static std::weak_ptr<UCXContext> global_ctx_;
};

} // namespace hmc

#endif // HMC_NET_UCX_H
