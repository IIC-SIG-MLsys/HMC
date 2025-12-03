#include "net_ucx.h"

#include "../utils/log.h"
#include <hmc.h>

#include <cstring>

namespace hmc {

// ======== UCX 状态到 status_t 的简单转换 ========

static status_t ucs_to_status(ucs_status_t st) {
  switch (st) {
  case UCS_OK:
    return status_t::SUCCESS;
  case UCS_ERR_UNSUPPORTED:
    return status_t::UNSUPPORT;
  case UCS_ERR_TIMED_OUT:
    return status_t::TIMEOUT;
  default:
    return status_t::ERROR;
  }
}

// ======== UCXContext 实现 ========

UCXContext::UCXContext() = default;

UCXContext::~UCXContext() { finalize(); }

status_t UCXContext::init() {
  if (initialized_) {
    return status_t::SUCCESS;
  }

  ucp_config_t *config = nullptr;
  ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
  if (st != UCS_OK) {
    logError("ucp_config_read failed: %d", st);
    return ucs_to_status(st);
  }

  // === 关键：强制使用 IB RC，而不是 TCP ===
  // 等价于环境变量：UCX_TLS=rc,self,sm
  ucs_status_t st2 = ucp_config_modify(config, "TLS", "rc,self,sm");
  if (st2 != UCS_OK) {
    logError("ucp_config_modify(TLS=rc,self,sm) failed: %d", st2);
    ucp_config_release(config);
    return ucs_to_status(st2);
  }
  // 如果你以后想通过环境变量控制，就把上面这段注释掉也行

  ucp_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features   = UCP_FEATURE_TAG;  // 只用 TAG API

  ucp_context_h ctx = nullptr;
  st = ucp_init(&params, config, &ctx);
  ucp_config_release(config);
  if (st != UCS_OK) {
    logError("ucp_init failed: %d", st);
    return ucs_to_status(st);
  }

  ucp_worker_params_t wparams;
  std::memset(&wparams, 0, sizeof(wparams));
  wparams.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_SINGLE; // 避免 multi-thread warning

  ucp_worker_h worker = nullptr;
  st = ucp_worker_create(ctx, &wparams, &worker);
  if (st != UCS_OK) {
    logError("ucp_worker_create failed: %d", st);
    ucp_cleanup(ctx);
    return ucs_to_status(st);
  }

  context_    = ctx;
  worker_     = worker;
  initialized_ = true;

  logInfo("UCXContext initialized (TAG mode, TLS=rc,self,sm)");
  return status_t::SUCCESS;
}


void UCXContext::finalize() {
  if (!initialized_)
    return;

  if (worker_) {
    ucp_worker_destroy(worker_);
    worker_ = nullptr;
  }

  if (context_) {
    ucp_cleanup(context_);
    context_ = nullptr;
  }

  initialized_ = false;
}

status_t UCXContext::packWorkerAddress(std::vector<std::uint8_t> &out) {
  if (!initialized_) {
    auto s = init();
    if (s != status_t::SUCCESS)
      return s;
  }

  ucp_address_t *addr = nullptr;
  size_t addr_len = 0;
  ucs_status_t st = ucp_worker_get_address(worker_, &addr, &addr_len);
  if (st != UCS_OK) {
    logError("ucp_worker_get_address failed: %d", st);
    return ucs_to_status(st);
  }

  out.assign(reinterpret_cast<std::uint8_t *>(addr),
             reinterpret_cast<std::uint8_t *>(addr) + addr_len);

  ucp_worker_release_address(worker_, addr);
  return status_t::SUCCESS;
}

void UCXContext::progress() {
  std::lock_guard<std::mutex> lock(worker_mutex_);
  if (worker_) {
    ucp_worker_progress(worker_);
  }
}

// ======== UCXClient 实现 ========

// 控制通道上的握手结构（必须是 POD）
struct UCXHandshake {
  uint32_t worker_addr_len;
};

// 静态成员定义
std::weak_ptr<UCXContext> UCXClient::global_ctx_;

UCXClient::UCXClient(std::shared_ptr<ConnBuffer> buffer)
    : buffer_(std::move(buffer)) {
  auto shared = global_ctx_.lock();
  if (!shared) {
    shared = std::make_shared<UCXContext>();
    global_ctx_ = shared;
  }
  ctx_ = shared;
}

UCXClient::~UCXClient() = default;

std::unique_ptr<Endpoint> UCXClient::connect(std::string ip, uint16_t port) {
  (void)port; // 控制连接已经在 Communicator::connectTo 中建立，这里不再使用

  if (!ctx_) {
    logError("UCXClient::connect: null UCXContext");
    return nullptr;
  }

  auto st = ctx_->init();
  if (st != status_t::SUCCESS) {
    logError("UCXClient::connect: UCXContext init failed");
    return nullptr;
  }

  CtrlSocketManager &ctrl = CtrlSocketManager::instance();

  // 打包本端 worker 地址
  std::vector<std::uint8_t> local_worker_addr;
  if (ctx_->packWorkerAddress(local_worker_addr) != status_t::SUCCESS) {
    logError("UCXClient::connect: packWorkerAddress failed");
    return nullptr;
  }

  UCXHandshake local{};
  local.worker_addr_len =
      static_cast<uint32_t>(local_worker_addr.size());

  // 对称握手：双方都先 send 再 recv，不会死锁
  if (!ctrl.sendCtrlStruct(ip, local)) {
    logError("UCXClient::connect: send local handshake failed");
    return nullptr;
  }

  UCXHandshake remote{};
  if (!ctrl.recvCtrlStruct(ip, remote)) {
    logError("UCXClient::connect: recv remote handshake failed");
    return nullptr;
  }

  // 发送本端 worker 地址
  if (!ctrl.sendCtrlMsg(ip, CTRL_STRUCT, local_worker_addr.data(),
                        local_worker_addr.size())) {
    logError("UCXClient::connect: send local worker addr failed");
    return nullptr;
  }

  // 接收对端 worker 地址
  CtrlMsgHeader hdr{};
  std::vector<std::uint8_t> remote_worker_addr;
  if (!ctrl.recvCtrlMsg(ip, hdr, remote_worker_addr)) {
    logError("UCXClient::connect: recv remote worker addr failed");
    return nullptr;
  }
  if (remote_worker_addr.size() != remote.worker_addr_len) {
    logError("UCXClient::connect: remote worker addr length mismatch");
    return nullptr;
  }

  // 用对端 worker 地址创建 EP
  ucp_ep_params_t ep_params;
  std::memset(&ep_params, 0, sizeof(ep_params));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address =
      reinterpret_cast<const ucp_address_t *>(remote_worker_addr.data());

  ucp_ep_h ep = nullptr;
  ucs_status_t ucs_st =
      ucp_ep_create(ctx_->worker(), &ep_params, &ep);
  if (ucs_st != UCS_OK) {
    logError("ucp_ep_create failed: %d", ucs_st);
    return nullptr;
  }

  auto endpoint = std::make_unique<UCXEndpoint>(ctx_, ep);
  endpoint->role = EndpointType::Client;

  logInfo("UCXClient::connect success to %s:%u", ip.c_str(), port);
  return endpoint;
}

} // namespace hmc
