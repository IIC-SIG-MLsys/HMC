#include "net_ucx.h"
#include "../utils/log.h"

#include <arpa/inet.h>
#include <cstring>
#include <thread>
#include <chrono>

namespace hmc {

static status_t ucs_to_status(ucs_status_t st) {
  switch (st) {
    case UCS_OK:              return status_t::SUCCESS;
    case UCS_ERR_UNSUPPORTED: return status_t::UNSUPPORT;
    case UCS_ERR_TIMED_OUT:   return status_t::TIMEOUT;
    default:                  return status_t::ERROR;
  }
}

static bool make_sockaddr_v4(const std::string &ip, uint16_t port,
                             sockaddr_storage &ss, socklen_t &ss_len) {
  std::memset(&ss, 0, sizeof(ss));
  sockaddr_in *sin = reinterpret_cast<sockaddr_in *>(&ss);
  sin->sin_family = AF_INET;
  sin->sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &sin->sin_addr) != 1) return false;
  ss_len = sizeof(sockaddr_in);
  return true;
}

struct TagReqWrap {
  void *ucx_req{nullptr};
  UcxRequest *meta{nullptr};
};

// ============================================================
// UCXContext
// ============================================================

UCXContext::UCXContext() = default;

UCXContext::~UCXContext() { finalize(); }

status_t UCXContext::init() {
  if (initialized_) return status_t::SUCCESS;

  ucp_config_t *config = nullptr;
  ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
  if (st != UCS_OK) {
    logError("ucp_config_read failed: %d", (int)st);
    return ucs_to_status(st);
  }

  ucp_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;

  ucp_context_h ctx = nullptr;
  st = ucp_init(&params, config, &ctx);
  ucp_config_release(config);
  if (st != UCS_OK) {
    logError("ucp_init failed: %d", (int)st);
    return ucs_to_status(st);
  }

  ucp_worker_params_t wparams;
  std::memset(&wparams, 0, sizeof(wparams));
  wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_SINGLE;

  ucp_worker_h worker = nullptr;
  st = ucp_worker_create(ctx, &wparams, &worker);
  if (st != UCS_OK) {
    logError("ucp_worker_create failed: %d", (int)st);
    ucp_cleanup(ctx);
    return ucs_to_status(st);
  }

  context_ = ctx;
  worker_ = worker;
  initialized_ = true;

  logInfo("UCXContext initialized (TAG+RMA)");
  return status_t::SUCCESS;
}

void UCXContext::finalize() {
  stopListener();

  if (!initialized_) return;

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

void UCXContext::progress() {
  std::lock_guard<std::mutex> lk(worker_mutex_);
  if (worker_) (void)ucp_worker_progress(worker_);
}

status_t UCXContext::packWorkerAddress(std::vector<std::uint8_t> &out) {
  if (!initialized_) {
    status_t s = init();
    if (s != status_t::SUCCESS) return s;
  }

  ucp_address_t *addr = nullptr;
  size_t addr_len = 0;
  ucs_status_t st = ucp_worker_get_address(worker_, &addr, &addr_len);
  if (st != UCS_OK) {
    logError("ucp_worker_get_address failed: %d", (int)st);
    return ucs_to_status(st);
  }

  out.assign(reinterpret_cast<std::uint8_t *>(addr),
             reinterpret_cast<std::uint8_t *>(addr) + addr_len);
  ucp_worker_release_address(worker_, addr);
  return status_t::SUCCESS;
}

void UCXContext::onConnectRequest(ucp_conn_request_h conn_request, void *arg) {
  auto *self = static_cast<UCXContext *>(arg);
  if (!self || !conn_request) return;
  std::lock_guard<std::mutex> lk(self->cr_mutex_);
  self->pending_conn_reqs_.push_back(conn_request);
}

status_t UCXContext::startListener(const std::string &ip, std::uint16_t port) {
  if (!initialized_) {
    status_t s = init();
    if (s != status_t::SUCCESS) return s;
  }
  if (listener_) return status_t::SUCCESS;

  sockaddr_storage ss;
  socklen_t ss_len = 0;
  if (!make_sockaddr_v4(ip, port, ss, ss_len)) {
    logError("startListener invalid ip: %s", ip.c_str());
    return status_t::ERROR;
  }

  ucp_listener_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                      UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  params.sockaddr.addr = reinterpret_cast<const sockaddr *>(&ss);
  params.sockaddr.addrlen = ss_len;
  params.conn_handler.cb = &UCXContext::onConnectRequest;
  params.conn_handler.arg = this;

  ucp_listener_h l = nullptr;
  ucs_status_t st = ucp_listener_create(worker_, &params, &l);
  if (st != UCS_OK) {
    logError("ucp_listener_create failed: %d", (int)st);
    return ucs_to_status(st);
  }

  listener_ = l;
  logInfo("UCXContext listener started on %s:%u", ip.c_str(), (unsigned)port);
  return status_t::SUCCESS;
}

void UCXContext::stopListener() {
  if (listener_) {
    ucp_listener_destroy(listener_);
    listener_ = nullptr;
  }
  std::lock_guard<std::mutex> lk(cr_mutex_);
  pending_conn_reqs_.clear();
}

ucp_conn_request_h UCXContext::popConnRequest() {
  std::lock_guard<std::mutex> lk(cr_mutex_);
  if (pending_conn_reqs_.empty()) return nullptr;
  ucp_conn_request_h r = pending_conn_reqs_.front();
  pending_conn_reqs_.erase(pending_conn_reqs_.begin());
  return r;
}

// ============================================================
// UCXEndpoint
// ============================================================

UCXEndpoint::UCXEndpoint(std::shared_ptr<UCXContext> ctx,
                         ucp_ep_h ep,
                         std::shared_ptr<ConnBuffer> buffer)
    : ctx_(std::move(ctx)), ep_(ep), buffer_(std::move(buffer)) {
  if (buffer_) {
    local_base_ = reinterpret_cast<std::uint64_t>(buffer_->ptr);
    local_size_ = buffer_->buffer_size;
  }
}

UCXEndpoint::~UCXEndpoint() {
  closeEndpoint();
  if (local_mem_) ucp_mem_unmap(ctx_->context(), local_mem_);
  if (remote_rkey_) ucp_rkey_destroy(remote_rkey_);
}

ucp_tag_t UCXEndpoint::nextDefaultTag() {
  const uint32_t seq = tag_seq_.fetch_add(1, std::memory_order_relaxed);
  return UcxTagCodec::make(default_type_, default_channel_, seq);
}

void UCXEndpoint::onSendComplete(void *, ucs_status_t status, void *user_data) {
  auto *m = static_cast<UcxRequest *>(user_data);
  if (!m) return;
  m->status.store(status, std::memory_order_release);
  m->completed.store(true, std::memory_order_release);
}

void UCXEndpoint::onRecvComplete(void *, ucs_status_t status,
                                 const ucp_tag_recv_info_t *info,
                                 void *user_data) {
  auto *m = static_cast<UcxRequest *>(user_data);
  if (!m) return;
  if (status == UCS_OK && info)
    m->recv_len.store(info->length, std::memory_order_release);
  m->status.store(status, std::memory_order_release);
  m->completed.store(true, std::memory_order_release);
}

void UCXEndpoint::onRmaComplete(void *, ucs_status_t status, void *user_data) {
  auto *m = static_cast<UcxRequest *>(user_data);
  if (!m) return;
  m->status.store(status, std::memory_order_release);
  m->completed.store(true, std::memory_order_release);
}

status_t UCXEndpoint::tagSendAsync(void *buf, size_t len, void **out_req,
                                  uint16_t ch, uint32_t seq) {
  if (out_req) *out_req = nullptr;
  if (!ep_ || !buf || len == 0) return status_t::SUCCESS;

  auto *meta = new UcxRequest();
  auto *wrap = new TagReqWrap();
  wrap->meta = meta;

  ucp_tag_t tag = UcxTagCodec::make(UcxMsgType::kData, ch,
                                   seq ? seq : tag_seq_.fetch_add(1));

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));
  p.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
  p.cb.send = &UCXEndpoint::onSendComplete;
  p.user_data = meta;

  ucs_status_ptr_t req = ucp_tag_send_nbx(ep_, buf, len, tag, &p);
  if (UCS_PTR_IS_ERR(req)) {
    logError("ucp_tag_send_nbx failed: %d", UCS_PTR_STATUS(req));
    delete meta; delete wrap;
    return status_t::ERROR;
  }

  if (req == nullptr) {
    meta->status.store(UCS_OK);
    meta->completed.store(true);
    if (out_req) *out_req = wrap;
    return status_t::SUCCESS;
  }

  wrap->ucx_req = req;
  if (out_req) *out_req = wrap;
  return status_t::SUCCESS;
}

status_t UCXEndpoint::tagRecvAsync(void *buf, size_t capacity, void **out_req,
                                  size_t *recv_len,
                                  uint16_t channel, uint32_t seq) {
  if (out_req) *out_req = nullptr;
  if (recv_len) *recv_len = 0;
  if (!buf || capacity == 0) return status_t::ERROR;

  auto *meta = new UcxRequest();
  auto *wrap = new TagReqWrap();
  wrap->meta = meta;

  const ucp_tag_t tag = (seq == 0)
      ? UcxTagCodec::make(UcxMsgType::kData, channel, 0)
      : UcxTagCodec::make(UcxMsgType::kData, channel, seq);

  ucp_tag_t mask = UcxTagCodec::kMaskAll;
  if (seq == 0) {
    const ucp_tag_t seq_mask = (static_cast<ucp_tag_t>(0xFFFFFFFFu) << 8);
    mask = static_cast<ucp_tag_t>(~seq_mask); // 只匹配 type+channel，seq wildcard
  }

  ucp_request_param_t param;
  std::memset(&param, 0, sizeof(param));
  param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                       UCP_OP_ATTR_FIELD_USER_DATA;
  param.cb.recv = &UCXEndpoint::onRecvComplete;
  param.user_data = meta;

  ucs_status_ptr_t req =
      ucp_tag_recv_nbx(ctx_->worker(), buf, capacity, tag, mask, &param);

  if (UCS_PTR_IS_ERR(req)) {
    logError("ucp_tag_recv_nbx failed: %d", (int)UCS_PTR_STATUS(req));
    delete wrap;
    delete meta;
    return status_t::ERROR;
  }

  // 可能立即完成：长度会在 callback（或某些 UCX 实现里不回调）——这里统一靠 meta->recv_len
  if (req == nullptr) {
    meta->status.store(UCS_OK, std::memory_order_release);
    meta->completed.store(true, std::memory_order_release);
    wrap->ucx_req = nullptr;
    if (out_req) *out_req = wrap;
    return status_t::SUCCESS;
  }

  wrap->ucx_req = req;
  if (out_req) *out_req = wrap;
  return status_t::SUCCESS;
}

status_t UCXEndpoint::waitRequest(void *request) {
  if (!request) return status_t::SUCCESS;

  auto *wrap = static_cast<TagReqWrap*>(request);
  auto *meta = wrap->meta;

  if (wrap->ucx_req == nullptr) {
    status_t r = ucs_to_status(meta->status.load());
    delete meta; delete wrap;
    return r;
  }

  while (true) {
    ctx_->progress();
    if (meta->completed.load()) break;
    ucs_status_t st = ucp_request_check_status(wrap->ucx_req);
    if (st != UCS_INPROGRESS) {
      meta->status.store(st);
      meta->completed.store(true);
      break;
    }
  }

  ucp_request_free(wrap->ucx_req);
  wrap->ucx_req = nullptr;

  status_t r = ucs_to_status(meta->status.load());
  delete meta; delete wrap;
  return r;
}

status_t UCXEndpoint::tagSend(void *buf, size_t len, uint16_t ch, uint32_t seq) {
  void *req = nullptr;
  status_t st = tagSendAsync(buf, len, &req, ch, seq);
  if (st != status_t::SUCCESS) return st;
  return waitRequest(req);
}

status_t UCXEndpoint::tagRecv(void *buf, size_t capacity, size_t *recv_len,
                             uint16_t channel, uint32_t seq) {
  if (recv_len) *recv_len = 0;

  void *reqv = nullptr;
  status_t st = tagRecvAsync(buf, capacity, &reqv, recv_len, channel, seq);
  if (st != status_t::SUCCESS) return st;

  auto *wrap = static_cast<TagReqWrap *>(reqv);
  if (!wrap || !wrap->meta) return status_t::ERROR;

  UcxRequest *meta = wrap->meta;

  if (wrap->ucx_req == nullptr) {
    if (recv_len) *recv_len = meta->recv_len.load(std::memory_order_acquire);
    ucs_status_t ust = meta->status.load(std::memory_order_acquire);
    delete wrap;
    delete meta;
    return ucs_to_status(ust);
  }

  while (true) {
    ctx_->progress();

    if (meta->completed.load(std::memory_order_acquire)) break;

    ucs_status_t ust = ucp_request_check_status(wrap->ucx_req);
    if (ust != UCS_INPROGRESS) {
      meta->status.store(ust, std::memory_order_release);
      meta->completed.store(true, std::memory_order_release);
      break;
    }
  }

  ucp_request_free(wrap->ucx_req);
  wrap->ucx_req = nullptr;

  if (recv_len) *recv_len = meta->recv_len.load(std::memory_order_acquire);
  ucs_status_t ust = meta->status.load(std::memory_order_acquire);

  delete wrap;
  delete meta;
  return ucs_to_status(ust);
}

status_t UCXEndpoint::ensureLocalMemRegistered() {
  if (local_mem_ready_) return status_t::SUCCESS;
  if (!ctx_ || !buffer_ || !buffer_->ptr) return status_t::ERROR;
  if (local_size_ == 0) return status_t::ERROR;

  ucp_mem_map_params_t p;
  std::memset(&p, 0, sizeof(p));
  p.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                 UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  p.address = buffer_->ptr;
  p.length = local_size_;

  ucs_status_t st = ucp_mem_map(ctx_->context(), &p, &local_mem_);
  if (st != UCS_OK) {
    logError("ucp_mem_map failed: %d", (int)st);
    return ucs_to_status(st);
  }

  local_mem_ready_ = true;
  return status_t::SUCCESS;
}

status_t UCXEndpoint::exportLocalMemInfo(UcxRemoteMemInfo &hdr,
                                        std::vector<std::uint8_t> &rkey_bytes) {
  status_t st = ensureLocalMemRegistered();
  if (st != status_t::SUCCESS) return st;

  void *rkey_buf = nullptr;
  size_t rkey_len = 0;
  ucs_status_t ust = ucp_rkey_pack(ctx_->context(), local_mem_, &rkey_buf, &rkey_len);
  if (ust != UCS_OK) {
    logError("ucp_rkey_pack failed: %d", (int)ust);
    return ucs_to_status(ust);
  }

  hdr.base_addr = local_base_;
  hdr.size = local_size_;
  hdr.rkey_len = static_cast<std::uint32_t>(rkey_len);

  rkey_bytes.assign(reinterpret_cast<std::uint8_t*>(rkey_buf),
                    reinterpret_cast<std::uint8_t*>(rkey_buf) + rkey_len);
  ucp_rkey_buffer_release(rkey_buf);
  return status_t::SUCCESS;
}

status_t UCXEndpoint::setRemoteMemInfo(const UcxRemoteMemInfo &hdr,
                                       const std::vector<std::uint8_t> &rkey_bytes) {
  if (!ep_) return status_t::ERROR;
  if (hdr.base_addr == 0 || hdr.size == 0 || hdr.rkey_len == 0) return status_t::ERROR;
  if (rkey_bytes.size() != hdr.rkey_len) return status_t::ERROR;

  if (remote_rkey_) {
    ucp_rkey_destroy(remote_rkey_);
    remote_rkey_ = nullptr;
  }

  ucp_rkey_h rkey = nullptr;
  ucs_status_t ust = ucp_ep_rkey_unpack(ep_, rkey_bytes.data(), &rkey);
  if (ust != UCS_OK) {
    logError("ucp_ep_rkey_unpack failed: %d", (int)ust);
    return ucs_to_status(ust);
  }

  remote_base_ = hdr.base_addr;
  remote_size_ = hdr.size;
  remote_rkey_ = rkey;
  return status_t::SUCCESS;
}

status_t UCXEndpoint::writeData(size_t local_off, size_t remote_off, size_t size) {
  if (!ep_ || !remote_rkey_ || !buffer_ || !buffer_->ptr) return status_t::ERROR;
  if (local_off + size > local_size_) return status_t::ERROR;
  if (remote_off + size > remote_size_) return status_t::ERROR;

  void *src = static_cast<char*>(buffer_->ptr) + local_off;
  uint64_t dst = remote_base_ + remote_off;

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));

  ucs_status_ptr_t req = ucp_put_nbx(ep_, src, size, dst, remote_rkey_, &p);
  if (UCS_PTR_IS_ERR(req)) return status_t::ERROR;
  if (req == nullptr) return status_t::SUCCESS;

  while (ucp_request_check_status(req) == UCS_INPROGRESS) ctx_->progress();
  ucs_status_t st = ucp_request_check_status(req);
  ucp_request_free(req);
  return ucs_to_status(st);
}

status_t UCXEndpoint::readData(size_t local_off, size_t remote_off, size_t size) {
  if (!ep_ || !remote_rkey_ || !buffer_ || !buffer_->ptr) return status_t::ERROR;
  if (local_off + size > local_size_) return status_t::ERROR;
  if (remote_off + size > remote_size_) return status_t::ERROR;

  void *dst = static_cast<char*>(buffer_->ptr) + local_off;
  uint64_t src = remote_base_ + remote_off;

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));

  ucs_status_ptr_t req = ucp_get_nbx(ep_, dst, size, src, remote_rkey_, &p);
  if (UCS_PTR_IS_ERR(req)) return status_t::ERROR;
  if (req == nullptr) return status_t::SUCCESS;

  while (ucp_request_check_status(req) == UCS_INPROGRESS) ctx_->progress();
  ucs_status_t st = ucp_request_check_status(req);
  ucp_request_free(req);
  return ucs_to_status(st);
}

status_t UCXEndpoint::writeDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wrid) {
  if (wrid) *wrid = 0;
  if (!ep_ || !remote_rkey_ || !buffer_ || !buffer_->ptr) return status_t::ERROR;
  if (local_off + size > local_size_) return status_t::ERROR;
  if (remote_off + size > remote_size_) return status_t::ERROR;

  void *src = static_cast<char*>(buffer_->ptr) + local_off;
  uint64_t dst = remote_base_ + remote_off;

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));

  ucs_status_ptr_t req = ucp_put_nbx(ep_, src, size, dst, remote_rkey_, &p);
  if (UCS_PTR_IS_ERR(req)) return status_t::ERROR;

  if (req == nullptr) { // completed immediately
    if (wrid) *wrid = 0;
    return status_t::SUCCESS;
  }

  uint64_t id = next_wrid_.fetch_add(1, std::memory_order_relaxed) + 1; // avoid 0
  {
    std::lock_guard<std::mutex> lk(req_mutex_);
    inflight_[id] = req;
  }
  if (wrid) *wrid = id;
  return status_t::SUCCESS;
}

status_t UCXEndpoint::readDataNB(size_t local_off, size_t remote_off, size_t size, uint64_t *wrid) {
  if (wrid) *wrid = 0;
  if (!ep_ || !remote_rkey_ || !buffer_ || !buffer_->ptr) return status_t::ERROR;
  if (local_off + size > local_size_) return status_t::ERROR;
  if (remote_off + size > remote_size_) return status_t::ERROR;

  void *dst = static_cast<char*>(buffer_->ptr) + local_off;
  uint64_t src = remote_base_ + remote_off;

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));

  ucs_status_ptr_t req = ucp_get_nbx(ep_, dst, size, src, remote_rkey_, &p);
  if (UCS_PTR_IS_ERR(req)) return status_t::ERROR;

  if (req == nullptr) { // completed immediately
    if (wrid) *wrid = 0;
    return status_t::SUCCESS;
  }

  uint64_t id = next_wrid_.fetch_add(1, std::memory_order_relaxed) + 1; // avoid 0
  {
    std::lock_guard<std::mutex> lk(req_mutex_);
    inflight_[id] = req;
  }
  if (wrid) *wrid = id;
  return status_t::SUCCESS;
}

status_t UCXEndpoint::waitWrId(uint64_t wid) {
  void *req = nullptr;
  {
    std::lock_guard<std::mutex> lk(req_mutex_);
    auto it = inflight_.find(wid);
    if (it == inflight_.end()) return status_t::ERROR;
    req = it->second;
    inflight_.erase(it);
  }

  if (!req) return status_t::SUCCESS;

  while (ucp_request_check_status(req) == UCS_INPROGRESS) ctx_->progress();
  ucs_status_t st = ucp_request_check_status(req);
  ucp_request_free(req);
  return ucs_to_status(st);
}

status_t UCXEndpoint::waitWrIdMulti(
    const std::vector<uint64_t>& wrids,
    std::chrono::milliseconds timeout) {

  if (wrids.empty()) return status_t::SUCCESS;

  std::vector<std::pair<uint64_t, void*>> pending;
  pending.reserve(wrids.size());

  {
    std::lock_guard<std::mutex> lk(req_mutex_);
    for (auto id : wrids) {
      if (id == 0) continue; // treat as already done
      auto it = inflight_.find(id);
      if (it == inflight_.end()) {
        return status_t::ERROR;
      }
      pending.emplace_back(id, it->second);
      inflight_.erase(it);
    }
  }

  std::vector<void*> reqs;
  reqs.reserve(pending.size());
  for (auto &kv : pending) {
    if (kv.second != nullptr) reqs.push_back(kv.second);
  }
  if (reqs.empty()) return status_t::SUCCESS;

  auto start_ts = std::chrono::steady_clock::now();
  size_t last_left = reqs.size();

  // We will compact reqs in-place by swap-remove finished ones.
  while (!reqs.empty()) {
    // drive progress a bit
    ctx_->progress();

    bool made_progress = false;

    for (size_t i = 0; i < reqs.size(); /*increment inside*/) {
      void* req = reqs[i];
      auto st = ucp_request_check_status(req);
      if (st == UCS_INPROGRESS) {
        ++i;
        continue;
      }

      // finished (success or error)
      ucp_request_free(req);
      // remove by swap with last
      reqs[i] = reqs.back();
      reqs.pop_back();
      made_progress = true;
    }

    if (made_progress) {
      if (reqs.size() != last_left) {
        last_left = reqs.size();
        // logDebug("UCX waitWrIdMulti: remaining=%zu", last_left);
      }
      continue;
    }

    // backoff
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(50));

    auto now = std::chrono::steady_clock::now();
    if (now - start_ts > timeout) {
      return status_t::TIMEOUT;
    }
  }

  return status_t::SUCCESS;
}

status_t UCXEndpoint::closeEndpoint() {
  if (!ep_) return status_t::SUCCESS;

  ucp_request_param_t p;
  std::memset(&p, 0, sizeof(p));
  ucs_status_ptr_t req = ucp_ep_close_nbx(ep_, &p);
  if (UCS_PTR_IS_ERR(req)) {
    logError("ucp_ep_close_nbx failed: %d", UCS_PTR_STATUS(req));
  } else if (req != nullptr) {
    while (ucp_request_check_status(req) == UCS_INPROGRESS) ctx_->progress();
    ucp_request_free(req);
  }
  ep_ = nullptr;
  return status_t::SUCCESS;
}

} // namespace hmc
