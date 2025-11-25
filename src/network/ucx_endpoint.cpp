#include "net_ucx.h"

#include "../utils/log.h"

#include <cstring>

namespace hmc {

// 本文件局部的 ucs_to_status
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

UCXEndpoint::UCXEndpoint(std::shared_ptr<UCXContext> ctx, ucp_ep_h ep)
    : ctx_(std::move(ctx)), ep_(ep) {}

UCXEndpoint::~UCXEndpoint() {
  // 析构时尽量优雅关闭
  closeEndpoint();
}

status_t UCXEndpoint::waitRequest(void *request) {
  if (!request)
    return status_t::SUCCESS;

  while (true) {
    ctx_->progress();
    ucs_status_t st = ucp_request_check_status(request);
    if (st == UCS_INPROGRESS) {
      continue;
    }
    ucp_request_free(request);
    return ucs_to_status(st);
  }
}

status_t UCXEndpoint::uhm_send(void *input_buffer, const size_t send_size,
                               MemoryType mem_type) {
  (void)mem_type; // 目前让 UCX 自己做内存类型处理

  if (!ep_) {
    return status_t::INVALID_CONFIG;
  }
  if (!input_buffer || send_size == 0) {
    return status_t::SUCCESS;
  }

  ucp_request_param_t param;
  std::memset(&param, 0, sizeof(param));

  ucs_status_ptr_t req =
      ucp_tag_send_nbx(ep_, input_buffer, send_size, kTag, &param);

  if (UCS_PTR_IS_ERR(req)) {
    logError("ucp_tag_send_nbx failed: %d", UCS_PTR_STATUS(req));
    return status_t::ERROR;
  }

  if (req == nullptr) {
    // 立即完成
    return status_t::SUCCESS;
  }

  return waitRequest(req);
}

status_t UCXEndpoint::uhm_recv(void *output_buffer, const size_t buffer_size,
                               size_t *recv_flags, MemoryType mem_type) {
  (void)mem_type; // 目前让 UCX 自己做内存类型处理

  if (!output_buffer || buffer_size == 0) {
    if (recv_flags)
      *recv_flags = 0;
    return status_t::SUCCESS;
  }

  ucp_request_param_t param;
  std::memset(&param, 0, sizeof(param));

  ucs_status_ptr_t req = ucp_tag_recv_nbx(
      ctx_->worker(), output_buffer, buffer_size, kTag, (ucp_tag_t)-1, &param);

  if (UCS_PTR_IS_ERR(req)) {
    logError("ucp_tag_recv_nbx failed: %d", UCS_PTR_STATUS(req));
    return status_t::ERROR;
  }

  if (req == nullptr) {
    // 立即完成
    if (recv_flags)
      *recv_flags = buffer_size;
    return status_t::SUCCESS;
  }

  auto st = waitRequest(req);
  if (st == status_t::SUCCESS && recv_flags) {
    *recv_flags = buffer_size;
  }
  return st;
}

status_t UCXEndpoint::closeEndpoint() {
  if (ep_) {
    ucp_request_param_t param;
    std::memset(&param, 0, sizeof(param));
    ucs_status_ptr_t req = ucp_ep_close_nbx(ep_, &param);
    if (UCS_PTR_IS_ERR(req)) {
      logError("ucp_ep_close_nbx failed: %d", UCS_PTR_STATUS(req));
    } else if (req != nullptr) {
      // 等待 close 完成
      (void)waitRequest(req);
    }
    ep_ = nullptr;
  }

  return status_t::SUCCESS;
}

} // namespace hmc
