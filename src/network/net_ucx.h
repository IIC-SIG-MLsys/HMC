#ifndef HMC_NET_UCX_H
#define HMC_NET_UCX_H

#include "net.h"

#include <ucp/api/ucp.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace hmc {

struct UcxRemoteMemInfo {
  std::uint64_t base_addr{0}; 
  std::uint64_t size{0}; 
  std::uint32_t rkey_len{0};
};

struct UcxRequest {
  std::atomic<bool> completed{false};
  std::atomic<ucs_status_t> status{UCS_INPROGRESS};

  std::atomic<std::size_t> recv_len{0};

  std::uint64_t wr_id{0};
};

enum class UcxMsgType : std::uint8_t {
  kData = 1,
  kCtrl = 2,
};

struct UcxTagCodec {
  static constexpr ucp_tag_t kMaskAll = static_cast<ucp_tag_t>(-1);

  // [63:56]=type, [55:40]=channel, [39:8]=seq, [7:0]=reserved
  static constexpr ucp_tag_t make(UcxMsgType t, std::uint16_t ch, std::uint32_t seq) {
    return (static_cast<ucp_tag_t>(static_cast<std::uint8_t>(t)) << 56) |
           (static_cast<ucp_tag_t>(ch) << 40) |
           (static_cast<ucp_tag_t>(seq) << 8);
  }
};


class UCXContext {
public:
  UCXContext();
  ~UCXContext();

  UCXContext(const UCXContext &) = delete;
  UCXContext &operator=(const UCXContext &) = delete;

  status_t init();
  void finalize();

  ucp_context_h context() const { return context_; }
  ucp_worker_h  worker()  const { return worker_; }

  void progress();

  status_t packWorkerAddress(std::vector<std::uint8_t> &out);

  status_t startListener(const std::string &ip, std::uint16_t port);
  void stopListener();

  ucp_conn_request_h popConnRequest();

private:
  static void onConnectRequest(ucp_conn_request_h conn_request, void *arg);

private:
  ucp_context_h context_{nullptr};
  ucp_worker_h  worker_{nullptr};

  ucp_listener_h listener_{nullptr};

  std::mutex cr_mutex_;
  std::vector<ucp_conn_request_h> pending_conn_reqs_;

  std::mutex worker_mutex_;
  bool initialized_{false};
};


class UCXEndpoint : public Endpoint {
public:
  UCXEndpoint(std::shared_ptr<UCXContext> ctx,
              ucp_ep_h ep,
              std::shared_ptr<ConnBuffer> buffer);
  ~UCXEndpoint() override;

  status_t writeData(size_t local_off, size_t remote_off, size_t size) override;  // put
  status_t readData(size_t local_off, size_t remote_off, size_t size) override;   // get

  status_t writeDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wrid) override;
  status_t readDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wrid) override;
  status_t waitWrId(uint64_t wrid) override;
  status_t waitWrIdMulti(const std::vector<uint64_t>& target_wr_ids,
                                     std::chrono::milliseconds timeout = std::chrono::seconds(5)) override;

  status_t uhm_send(void *, const size_t, MemoryType) override {
    return status_t::UNSUPPORT;
  }
  status_t uhm_recv(void *, const size_t, size_t *, MemoryType) override {
    return status_t::UNSUPPORT;
  }

  // UCX TAG message
  status_t tagSend(void *buf, size_t len,
                   uint16_t channel = 0,
                   uint32_t seq = 0 /* 0=auto */);

  status_t tagRecv(void *buf, size_t capacity,
                   size_t *recv_len,
                   uint16_t channel = 0,
                   uint32_t seq = 0 /* 0=wildcard, 用 mask 控制 */);

  status_t tagSendAsync(void *buf, size_t len, void **out_req,
                        uint16_t channel = 0, uint32_t seq = 0);
  status_t tagRecvAsync(void *buf, size_t capacity, void **out_req,
                        size_t *recv_len,
                        uint16_t channel = 0, uint32_t seq = 0);

  status_t closeEndpoint() override;

  status_t setRemoteMemInfo(const UcxRemoteMemInfo &hdr,
                            const std::vector<std::uint8_t> &rkey_bytes);
  status_t exportLocalMemInfo(UcxRemoteMemInfo &hdr,
                              std::vector<std::uint8_t> &rkey_bytes);

private:
  status_t ensureLocalMemRegistered();
  status_t waitRequest(void *request);

  ucp_tag_t nextDefaultTag();

  static void onSendComplete(void *request, ucs_status_t status, void *user_data);
  static void onRecvComplete(void *request, ucs_status_t status,
                             const ucp_tag_recv_info_t *info, void *user_data);
  static void onRmaComplete(void *request, ucs_status_t status, void *user_data);

private:
  std::shared_ptr<UCXContext> ctx_;
  ucp_ep_h ep_{nullptr};
  std::shared_ptr<ConnBuffer> buffer_;

  std::atomic<std::uint32_t> tag_seq_{1};
  std::uint16_t default_channel_{0};
  UcxMsgType default_type_{UcxMsgType::kData};

  bool local_mem_ready_{false};
  ucp_mem_h local_mem_{nullptr};

  std::uint64_t local_base_{0};
  std::uint64_t local_size_{0};

  std::uint64_t remote_base_{0};
  std::uint64_t remote_size_{0};
  ucp_rkey_h remote_rkey_{nullptr};

  std::mutex req_mutex_;
  std::unordered_map<std::uint64_t, void *> inflight_; // wrid -> ucx request ptr
  std::atomic<std::uint64_t> next_wrid_{1};
};


class UCXServer : public Server {
public:
  explicit UCXServer(std::shared_ptr<ConnManager> conn_manager,
                     std::shared_ptr<ConnBuffer> buffer);

  status_t listen(std::string ip, uint16_t port) override;
  status_t stopListen() override;

  ~UCXServer() override;

private:
  std::string ip_{"0.0.0.0"};
  uint16_t port_{0};
  bool running_{false};

  std::shared_ptr<ConnBuffer> buffer_;
  std::shared_ptr<UCXContext> ctx_;
  std::atomic<bool> accept_running_{false};
  std::thread accept_th_;
};


class UCXClient : public Client {
public:
  explicit UCXClient(std::shared_ptr<ConnBuffer> buffer);
  ~UCXClient() override;

  std::unique_ptr<Endpoint> connect(std::string ip, uint16_t port) override;

private:
  std::shared_ptr<ConnBuffer> buffer_;
  std::shared_ptr<UCXContext> ctx_;

  static std::weak_ptr<UCXContext> global_ctx_;
};

} // namespace hmc

#endif
