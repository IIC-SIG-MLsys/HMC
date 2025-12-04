/**
 * @copyright
 * Copyright (c) 2025,
 * SDU spgroup Holding Limited. All rights reserved.
 */

#include "net_rdma.h"
#include <unordered_set>

namespace hmc {

/* -------------------------------------------------------------------------- */
/*                          RDMAEndpoint Lifecycle                            */
/* -------------------------------------------------------------------------- */

RDMAEndpoint::RDMAEndpoint(std::shared_ptr<ConnBuffer> buffer, size_t num_qps)
    : buffer(buffer), num_qps_(num_qps) {}

RDMAEndpoint::~RDMAEndpoint() {
  if (role == EndpointType::Client && connStatus == status_t::SUCCESS) {
    closeEndpoint(); // only client actively closes
  }
  cleanRdmaResources();
}

/* -------------------------------------------------------------------------- */
/*                              QP Management                                 */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::setupQPs() {
  if (!pd || !cq || !cm_id) {
    logError("RDMAEndpoint::setupQPs: invalid PD/CQ/cm_id");
    return status_t::ERROR;
  }

  qps_.resize(num_qps_);

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.cap.max_send_wr = max_wr;
  qp_init_attr.cap.max_recv_wr = max_wr;
  qp_init_attr.cap.max_send_sge = max_sge;
  qp_init_attr.cap.max_recv_sge = max_sge;

  // Primary QP attached to cm_id
  if (rdma_create_qp(cm_id, pd, &qp_init_attr)) {
    logError("Failed to create primary QP[0]");
    return status_t::ERROR;
  }
  qps_[0] = cm_id->qp;

  // Additional QPs created directly from verbs
  for (size_t i = 1; i < num_qps_; ++i) {
    qps_[i] = ibv_create_qp(pd, &qp_init_attr);
    if (!qps_[i]) {
      logError("Failed to create extra QP[%zu]", i);
      return status_t::ERROR;
    }
  }

  // Record QP numbers into metadata
  local_metadata_attr.qp_nums = num_qps_;
  for (size_t i = 0; i < num_qps_; ++i) {
    local_metadata_attr.qp_num_list[i] = qps_[i]->qp_num;
  }

  return status_t::SUCCESS;
}

ibv_qp *RDMAEndpoint::getQP(size_t idx) {
  if (num_qps_ == 0 || qps_.empty()) {
    logError("getQP: QP not initialized");
    return nullptr;
  }
  if (num_qps_ == 1) return qps_[0];
  return qps_[idx % num_qps_];
}

status_t RDMAEndpoint::transitionExtraQPsToRTS() {
  if (num_qps_ <= 1) return status_t::SUCCESS;

  for (size_t i = 1; i < num_qps_; ++i) {
    ibv_qp* qp = qps_[i];
    if (!qp) continue;

    // ---- INIT ----
    {
      ibv_qp_attr attr{};
      attr.qp_state        = IBV_QPS_INIT;
      attr.pkey_index      = 0;
      attr.port_num        = port_num_;
      attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_WRITE;

      int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                  IBV_QP_PORT  | IBV_QP_ACCESS_FLAGS;
      if (ibv_modify_qp(qp, &attr, flags)) {
        logError("Failed to modify QP[%zu] to INIT: %s", i, strerror(errno));
        return status_t::ERROR;
      }
    }

    // ---- RTR (RoCEv2 / Global Route) ----
    {
      ibv_qp_attr attr{};
      attr.qp_state           = IBV_QPS_RTR;
      attr.path_mtu           = IBV_MTU_1024;
      attr.dest_qp_num        = remote_metadata_attr.qp_num_list[i];
      attr.rq_psn             = 0;
      attr.max_dest_rd_atomic = 1;
      attr.min_rnr_timer      = 12;

      // RoCEv2: 使用全局 GRH
      attr.ah_attr.is_global     = 1;
      attr.ah_attr.port_num      = port_num_;
      attr.ah_attr.sl            = 0;
      attr.ah_attr.src_path_bits = 0;
      attr.ah_attr.static_rate   = 0;
      memset(&attr.ah_attr.grh,  0, sizeof(attr.ah_attr.grh));

      // 关键：对端 GID -> dgid；本端 sgid index
      memcpy(&attr.ah_attr.grh.dgid, remote_metadata_attr.gid, 16);
      attr.ah_attr.grh.sgid_index = sgid_index_;     // 本端生效的 sgid 索引
      attr.ah_attr.grh.hop_limit  = grh_hop_limit_;  // 常用 1

      int rtr_flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV |
                      IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

      if (ibv_modify_qp(qp, &attr, rtr_flags)) {
        logError("Failed to modify QP[%zu] to RTR (RoCEv2): %s",
                 i, strerror(errno));
        return status_t::ERROR;
      }

      logInfo("QP[%zu] RTR (RoCEv2): dgid[0..3]=%02x%02x%02x%02x, sgid_index=%u",
              i,
              attr.ah_attr.grh.dgid.raw[0], attr.ah_attr.grh.dgid.raw[1],
              attr.ah_attr.grh.dgid.raw[2], attr.ah_attr.grh.dgid.raw[3],
              attr.ah_attr.grh.sgid_index);
    }

    // ---- RTS ----
    {
      ibv_qp_attr attr{};
      attr.qp_state      = IBV_QPS_RTS;
      attr.timeout       = 14;
      attr.retry_cnt     = 7;
      attr.rnr_retry     = 7;
      attr.sq_psn        = 0;
      attr.max_rd_atomic = 1;

      int rts_flags = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;

      if (ibv_modify_qp(qp, &attr, rts_flags)) {
        logError("Failed to modify QP[%zu] to RTS: %s", i, strerror(errno));
        return status_t::ERROR;
      }

      logInfo("QP[%zu] modified to RTS", i);
    }
  }

  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                              Connection Ops                                */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::closeEndpoint() {
  struct rdma_cm_event *cm_event = nullptr;
  if (rdma_disconnect(this->cm_id)) {
    logError("RDMAEndpoint: Failed to disconnect");
    return status_t::ERROR;
  }
  int ret = rdma_get_cm_event(this->cm_event_channel, &cm_event);
  if (ret) {
    logError("RDMAEndpoint: Failed to get disconnect event");
    return status_t::ERROR;
  } else {
    rdma_ack_cm_event(cm_event);
  }
  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                          RDMA Read/Write API                               */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::writeData(size_t local_off, size_t remote_off, size_t size) {
  // local range check
  if (local_off + size > buffer->buffer_size) {
    logError("writeData: local range invalid, local_off=%zu size=%zu buf=%zu",
             local_off, size, buffer->buffer_size);
    return status_t::ERROR;
  }
  // remote range check (needs remote_metadata_attr.length valid)
  if (remote_off + size > (size_t)remote_metadata_attr.length) {
    logError("writeData: remote range invalid, remote_off=%zu size=%zu remote_len=%u",
             remote_off, size, remote_metadata_attr.length);
    return status_t::ERROR;
  }

  uint64_t wr_id;
  if (writeDataNB(local_off, remote_off, size, &wr_id) != status_t::SUCCESS) {
    logError("writeData: post_write failed");
    return status_t::ERROR;
  }
  if (waitWrId(wr_id) != status_t::SUCCESS) {
    logError("writeData: waitWrId failed");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::writeDataNB(size_t local_off, size_t remote_off, size_t size,
                                  uint64_t *wr_id) {
  if (!wr_id) return status_t::ERROR;

  if (local_off + size > buffer->buffer_size) {
    logError("writeDataNB: local range invalid, local_off=%zu size=%zu buf=%zu",
             local_off, size, buffer->buffer_size);
    return status_t::ERROR;
  }
  if (remote_off + size > (size_t)remote_metadata_attr.length) {
    logError("writeDataNB: remote range invalid, remote_off=%zu size=%zu remote_len=%u",
             remote_off, size, remote_metadata_attr.length);
    return status_t::ERROR;
  }

  void *localAddr  = static_cast<char *>(buffer->ptr) + local_off;
  void *remoteAddr = reinterpret_cast<char *>(remote_metadata_attr.address) + remote_off;

  *wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
  size_t qp_idx = (*wr_id) % num_qps_;

  return postWrite(localAddr, remoteAddr, size, buffer_mr,
                   remote_metadata_attr.key, *wr_id, /*signaled=*/true, qp_idx);
}

status_t RDMAEndpoint::readData(size_t local_off, size_t remote_off, size_t size) {
  // local range check
  if (local_off + size > buffer->buffer_size) {
    logError("readData: local range invalid, local_off=%zu size=%zu buf=%zu",
             local_off, size, buffer->buffer_size);
    return status_t::ERROR;
  }
  // remote range check
  if (remote_off + size > (size_t)remote_metadata_attr.length) {
    logError("readData: remote range invalid, remote_off=%zu size=%zu remote_len=%u",
             remote_off, size, remote_metadata_attr.length);
    return status_t::ERROR;
  }

  uint64_t wr_id;
  if (readDataNB(local_off, remote_off, size, &wr_id) != status_t::SUCCESS) {
    logError("readData: post_read failed");
    return status_t::ERROR;
  }
  if (waitWrId(wr_id) != status_t::SUCCESS) {
    logError("readData: waitWrId failed");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::readDataNB(size_t local_off, size_t remote_off, size_t size,
                                 uint64_t *wr_id) {
  if (!wr_id) return status_t::ERROR;

  if (local_off + size > buffer->buffer_size) {
    logError("readDataNB: local range invalid, local_off=%zu size=%zu buf=%zu",
             local_off, size, buffer->buffer_size);
    return status_t::ERROR;
  }
  if (remote_off + size > (size_t)remote_metadata_attr.length) {
    logError("readDataNB: remote range invalid, remote_off=%zu size=%zu remote_len=%u",
             remote_off, size, remote_metadata_attr.length);
    return status_t::ERROR;
  }

  void *localAddr  = static_cast<char *>(buffer->ptr) + local_off;
  void *remoteAddr = reinterpret_cast<char *>(remote_metadata_attr.address) + remote_off;

  *wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
  size_t qp_idx = (*wr_id) % num_qps_;

  return postRead(localAddr, remoteAddr, size, buffer_mr,
                  remote_metadata_attr.key, *wr_id, /*signaled=*/true, qp_idx);
}

/* -------------------------------------------------------------------------- */
/*                           UHM Send / Recv                                  */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::uhm_send(void *input_buffer, const size_t send_flags,
                                MemoryType mem_type) {
  status_t ret;

  const size_t half_buffer_size = buffer->buffer_size / 2;
  const size_t num_send_chunks =
      (send_flags + half_buffer_size - 1) / half_buffer_size;
  size_t current_chunk = 0;
  size_t chunk_index = 0;
  size_t send_size = std::min(half_buffer_size, send_flags);

  // set flags
  uhm_buffer_state.state[0] = UHM_BUFFER_CAN_WRITE;
  uhm_buffer_state.state[1] = UHM_BUFFER_CAN_WRITE;
  uhm_buffer_state.length = send_flags;

  // write buffer_state (use qp 0 for flag bootstrap)
  void *localAddr = reinterpret_cast<char *>(&uhm_buffer_state);
  void *remoteAddr =
      reinterpret_cast<char *>(remote_metadata_attr.uhm_buffer_state_address);
  uint64_t wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
  ret = postWrite(localAddr, remoteAddr, sizeof(uhm_buffer_state),
                  uhm_buffer_state_mr,
                  remote_metadata_attr.uhm_buffer_state_key, wr_id, true, 0);
  if (ret != status_t::SUCCESS) {
    logError("Client::Send: Failed to post write buffer state");
    return ret;
  }

  // pre-fill first half buffer
  mem_type == MemoryType::CPU
      ? buffer->writeFromCpu(input_buffer, send_size, 0)
      : buffer->writeFromGpu(input_buffer, send_size, 0);

  if (waitWrId(wr_id) != status_t::SUCCESS) {
    logError("Client::Send: Failed to poll completion for state write");
    return status_t::ERROR;
  }

  while (current_chunk < num_send_chunks) {
    chunk_index = current_chunk % 2;
    size_t next_chunk_index = (current_chunk + 1) % 2;

    // spin until remote allows write on this half
    const int SPIN_COUNT = 100;
    while (uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
      int spin = 0;
      while (spin++ < SPIN_COUNT &&
             uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
      }
      if (uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
        std::this_thread::yield();
      }
    }

    // this chunk size
    size_t remaining = send_flags - current_chunk * half_buffer_size;
    send_size = std::min(half_buffer_size, remaining);

    // choose qp for this chunk to keep ordering between data and flag
    size_t qp_idx = current_chunk % num_qps_;

    // post data write
    void *data_local = static_cast<char *>(buffer->ptr) + chunk_index * half_buffer_size;
    void *data_remote =
        reinterpret_cast<char *>(remote_metadata_attr.address) + chunk_index * half_buffer_size;
    wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
    ret = postWrite(data_local, data_remote, send_size, buffer_mr,
                    remote_metadata_attr.key, wr_id, true, qp_idx);
    if (ret != status_t::SUCCESS) {
      logError("Client::Send: Failed to post write data for chunk %zu",
               current_chunk);
      return ret;
    }

    // mark this half as readable on the same QP to preserve order
    uhm_buffer_state.state[chunk_index] = UHM_BUFFER_CAN_READ;
    void *flag_local = reinterpret_cast<char *>(&uhm_buffer_state) +
                       chunk_index * sizeof(UHM_STATE_TYPE);
    void *flag_remote =
        reinterpret_cast<char *>(remote_metadata_attr.uhm_buffer_state_address) +
        chunk_index * sizeof(UHM_STATE_TYPE);
    ret = postWrite(flag_local, flag_remote, sizeof(UHM_STATE_TYPE),
                    uhm_buffer_state_mr,
                    remote_metadata_attr.uhm_buffer_state_key, wr_id, true, qp_idx);
    if (ret != status_t::SUCCESS) {
      logError("Client::Send: Failed to post write buffer state");
      return ret;
    }

    // pre-fill next half if any
    if (current_chunk + 1 < num_send_chunks) {
      size_t next_remaining =
          send_flags - (current_chunk + 1) * half_buffer_size;
      size_t next_size = std::min(half_buffer_size, next_remaining);
      size_t bias = next_chunk_index * half_buffer_size;
      mem_type == MemoryType::CPU
          ? buffer->writeFromCpu(input_buffer, next_size, bias)
          : buffer->writeFromGpu(input_buffer, next_size, bias);
    }

    // wait completion for data+flag
    if (waitWrId(wr_id) != status_t::SUCCESS) {
      logError("Client::Send: Failed to poll completion for chunk %zu",
               current_chunk);
      return status_t::ERROR;
    }

    current_chunk++;
  }

  return status_t::SUCCESS;
}

status_t RDMAEndpoint::uhm_recv(void *output_buffer, const size_t buffer_size,
                                size_t *recv_flags, MemoryType mem_type) {
  status_t ret;
  size_t current_chunk = 0;
  size_t chunk_index = 0;
  size_t accumulated_size = 0;
  size_t recv_size = 0;
  const size_t half_buffer_size = buffer->buffer_size / 2;
  size_t num_recv_chunks = 0;

  // init states so sender can see we're ready
  uhm_buffer_state.state[0] = UHM_BUFFER_CAN_READ;
  uhm_buffer_state.state[1] = UHM_BUFFER_CAN_READ;
  uhm_buffer_state.length = 0;

  // wait for length + initial states
  const int SPIN_COUNT = 100;
  while (true) {
    auto tmp = uhm_buffer_state;
    if (tmp.state[chunk_index] == UHM_BUFFER_CAN_WRITE) {
      *recv_flags = tmp.length;
      if (*recv_flags == 0) {
        logError("Server::Recv: Invalid receive size is 0");
        return status_t::ERROR;
      }
      num_recv_chunks = (*recv_flags + half_buffer_size - 1) / half_buffer_size;
      this->uhm_buffer_state.state[0] = UHM_BUFFER_CAN_WRITE;
      this->uhm_buffer_state.state[1] = UHM_BUFFER_CAN_WRITE;
      break;
    }

    int spin = 0;
    while (spin++ < SPIN_COUNT &&
           this->uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
#if defined(__x86_64__) || defined(_M_X64)
      __builtin_ia32_pause();
#elif defined(__aarch64__)
      asm volatile("yield");
#endif
    }
  }

  // receive chunks
  uint64_t wr_id;
  while (current_chunk < num_recv_chunks) {
    chunk_index = current_chunk % 2;
    // choose qp for this chunk (for ordering of flag write-back)
    size_t qp_idx = current_chunk % num_qps_;

    if (this->uhm_buffer_state.state[chunk_index] == UHM_BUFFER_CAN_READ) {
      // immediately grant write permission back to sender (same QP)
      this->uhm_buffer_state.state[chunk_index] = UHM_BUFFER_CAN_WRITE;
      void *flag_local = reinterpret_cast<char *>(&uhm_buffer_state) +
                         chunk_index * sizeof(UHM_STATE_TYPE);
      void *flag_remote =
          reinterpret_cast<char *>(remote_metadata_attr.uhm_buffer_state_address) +
          chunk_index * sizeof(UHM_STATE_TYPE);
      wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
      ret = postWrite(flag_local, flag_remote, sizeof(UHM_STATE_TYPE),
                      uhm_buffer_state_mr,
                      remote_metadata_attr.uhm_buffer_state_key, wr_id, true, qp_idx);
      if (ret != status_t::SUCCESS) {
        logError("Server::Recv: Failed to post write buffer state");
        return ret;
      }

      // compute receive size
      if (current_chunk == num_recv_chunks - 1) {
        recv_size = *recv_flags - accumulated_size;
      } else {
        recv_size = half_buffer_size;
      }
      // boundary checks
      if (recv_size == 0) {
        logError("Server::Recv: Invalid receive size is 0");
        return status_t::ERROR;
      } else if (recv_size > buffer_size) {
        logError("Server::Recv: Invalid receive size %zu > buffer_size %zu",
                 recv_size, buffer_size);
        return status_t::ERROR;
      } else if (accumulated_size + recv_size > buffer_size) {
        logError("Server::Recv: accumulated_size + recv_size > buffer_size");
        return status_t::ERROR;
      }

      // copy to output
      size_t bias = chunk_index * half_buffer_size;
      void *dest = static_cast<char *>(output_buffer) + accumulated_size;
      mem_type == MemoryType::CPU ? buffer->readToCpu(dest, recv_size, bias)
                                  : buffer->readToGpu(dest, recv_size, bias);

      // ensure flag write completion processed
      if (waitWrId(wr_id) != status_t::SUCCESS) {
        logError("Failed to poll completion queue");
        return status_t::ERROR;
      }

      accumulated_size += recv_size;
      current_chunk++;
    }
  };

  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                           Memory Registration                              */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::registerMemory(void *addr, size_t length,
                                      struct ibv_mr **mr) {
  if (!addr || !mr) {
    return status_t::ERROR;
  }
  *mr = ibv_reg_mr(pd, addr, length,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE);
  if (!*mr) {
    logError("Failed to register memory region");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::deRegisterMemory(struct ibv_mr *mr) {
  if (!mr) return status_t::ERROR;
  if (ibv_dereg_mr(mr)) {
    logError("Failed to deregister memory region");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                           Buffer Initialization                            */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::setupBuffers() {
  status_t ret = registerMemory(
      &local_metadata_attr, sizeof(local_metadata_attr), &local_metadata_mr);
  if (ret != status_t::SUCCESS) return ret;

  ret = registerMemory(&remote_metadata_attr, sizeof(remote_metadata_attr),
                       &remote_metadata_mr);
  if (ret != status_t::SUCCESS) return ret;

  if (!buffer) {
    logError("Error while register buffer, buffer is NULL");
    return status_t::ERROR;
  }

  ret = registerMemory(buffer->ptr, buffer->buffer_size, &buffer_mr);
  if (ret != status_t::SUCCESS) {
    logError("Error while register buffer %p", buffer.get());
    return ret;
  }

  local_metadata_attr.address = (uint64_t)buffer->ptr;
  local_metadata_attr.length = buffer->buffer_size;
  local_metadata_attr.key = buffer_mr->lkey;

  ret = registerMemory(&uhm_buffer_state, sizeof(uhm_buffer_state),
                       &uhm_buffer_state_mr);
  if (ret != status_t::SUCCESS) {
    logError("Error while register uhm_buffer_state");
    return ret;
  }

  local_metadata_attr.uhm_buffer_state_address = (uint64_t)&uhm_buffer_state;
  local_metadata_attr.uhm_buffer_state_key = uhm_buffer_state_mr->lkey;

  // RoCEv2: query GID 并写入 metadata
  {
    ibv_gid gid{};
    int rc = ibv_query_gid(pd->context, port_num_, sgid_index_, &gid);
    if (rc == 0) {
      memcpy(local_metadata_attr.gid, &gid, 16);
      local_metadata_attr.sgid_index = sgid_index_;
    } else {
      memset(local_metadata_attr.gid, 0, 16);
      local_metadata_attr.sgid_index = sgid_index_;
      logError("ibv_query_gid failed on port=%u index=%u (rc=%d)",
               port_num_, sgid_index_, rc);
    }
  }

  // Fill QP info after QP setup
  local_metadata_attr.qp_nums = num_qps_;
  for (size_t i = 0; i < num_qps_; ++i)
    local_metadata_attr.qp_num_list[i] = qps_[i]->qp_num;

  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                          RDMA Post Operations                              */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::postSend(void *addr, size_t length, struct ibv_mr *mr,
                                uint64_t wr_id, bool signaled, size_t qp_idx) {
  struct ibv_send_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)addr;
  sge.length = length;
  sge.lkey = mr->lkey;

  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;

  ibv_qp *target_qp = getQP(qp_idx);
  if (ibv_post_send(target_qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA send");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postRecv(void *addr, size_t length, struct ibv_mr *mr,
                                uint64_t wr_id, size_t qp_idx) {
  struct ibv_recv_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)addr;
  sge.length = length;
  sge.lkey = mr->lkey;

  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  ibv_qp *target_qp = getQP(qp_idx);
  if (ibv_post_recv(target_qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA recv");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postWrite(void *local_addr, void *remote_addr,
                                 size_t length, struct ibv_mr *local_mr,
                                 uint32_t remote_key, uint64_t wr_id,
                                 bool signaled, size_t qp_idx) {
  struct ibv_send_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)local_addr;
  sge.length = length;
  sge.lkey = local_mr->lkey;

  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = (uint64_t)remote_addr;
  wr.wr.rdma.rkey = remote_key;

  ibv_qp *target_qp = getQP(qp_idx);
  if (ibv_post_send(target_qp, &wr, &bad_wr)) {
    int e = errno;
    logError("ibv_post_send failed on QP[%zu]: errno=%d (%s)", qp_idx, e, strerror(e));
    if (bad_wr)
      fprintf(stderr, "Failed wr_id: %lu\n", bad_wr->wr_id);

    ibv_qp_attr attr;
    ibv_qp_init_attr init_attr;
    if (ibv_query_qp(target_qp, &attr, IBV_QP_STATE, &init_attr) == 0) {
      logError("QP[%zu] state after failure: %d", qp_idx, attr.qp_state);
    }
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postRead(void *local_addr, void *remote_addr,
                                size_t length, struct ibv_mr *local_mr,
                                uint32_t remote_key, uint64_t wr_id,
                                bool signaled, size_t qp_idx) {
  struct ibv_send_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)local_addr;
  sge.length = length;
  sge.lkey = local_mr->lkey;

  wr.wr_id = wr_id;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = (uint64_t)remote_addr;
  wr.wr.rdma.rkey = remote_key;

  ibv_qp *target_qp = getQP(qp_idx);
  if (ibv_post_send(target_qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA read on QP[%zu]", qp_idx);
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

/* -------------------------------------------------------------------------- */
/*                           Debug helper                                     */
/* -------------------------------------------------------------------------- */

void RDMAEndpoint::showRdmaBufferAttr(const struct rdma_buffer_attr *attr) {
  logInfo("Buffer Attr:");
  logInfo("  address: 0x%lx", attr->address);
  logInfo("  length: %u", attr->length);
  logInfo("  key: 0x%x", attr->key);
  logInfo("  uhm_buffer_state address: 0x%lx", attr->uhm_buffer_state_address);
  logInfo("  uhm_buffer_state key: 0x%x", attr->uhm_buffer_state_key);
  logInfo("  QPs: %u", attr->qp_nums);
  for (uint32_t i = 0; i < attr->qp_nums; ++i) {
    logInfo("    qp_num_list[%u] = %u", i, attr->qp_num_list[i]);
  }
}

/* -------------------------------------------------------------------------- */
/*                            Resource Cleanup                                */
/* -------------------------------------------------------------------------- */

void RDMAEndpoint::cleanRdmaResources() {
  for (auto &qp_ptr : qps_) {
    if (qp_ptr) {
      ibv_destroy_qp(qp_ptr);
      qp_ptr = nullptr;
    }
  }
  qps_.clear();

  if (cq) {
    ibv_destroy_cq(cq);
    cq = nullptr;
  }
  if (pd) {
    ibv_dealloc_pd(pd);
    pd = nullptr;
  }
  if (completion_channel) {
    ibv_destroy_comp_channel(completion_channel);
    completion_channel = nullptr;
  }
  if (local_metadata_mr) {
    deRegisterMemory(local_metadata_mr);
    local_metadata_mr = nullptr;
  }
  if (remote_metadata_mr) {
    deRegisterMemory(remote_metadata_mr);
    remote_metadata_mr = nullptr;
  }
  if (uhm_buffer_state_mr) {
    deRegisterMemory(uhm_buffer_state_mr);
    uhm_buffer_state_mr = nullptr;
  }
  if (cm_event_channel) {
    rdma_destroy_event_channel(cm_event_channel);
    cm_event_channel = nullptr;
  }
  if (cm_id) {
    rdma_destroy_id(cm_id);
    cm_id = nullptr;
  }
}

/* -------------------------------------------------------------------------- */
/*                          Completion Queue Polling                          */
/* -------------------------------------------------------------------------- */

status_t RDMAEndpoint::waitWrId(uint64_t target_wr_id) {
  if (!cq) {
    logError("waitForWrId: CQ is null");
    return status_t::ERROR;
  }

  const int max_wcs = cq_capacity;
  std::vector<ibv_wc> wcs(max_wcs);

  while (true) {
    int n = ibv_poll_cq(cq, max_wcs, wcs.data());
    if (n < 0) {
      logError("waitWrId: poll CQ failed");
      return status_t::ERROR;
    }
    if (n == 0) continue;

    for (int i = 0; i < n; ++i) {
      if (wcs[i].status != IBV_WC_SUCCESS) {
        logError("waitWrId: CQE error, status %d", wcs[i].status);
        return status_t::ERROR;
      }
      if (wcs[i].wr_id == target_wr_id) {
        return status_t::SUCCESS;
      }
    }
  }
}

status_t RDMAEndpoint::waitWrIdMulti(const std::vector<uint64_t>& target_wr_ids,
                                     std::chrono::milliseconds timeout) {
  if (!cq) {
    logError("waitWrIdMulti: CQ is null");
    return status_t::ERROR;
  }
  if (target_wr_ids.empty()) {
    logDebug("waitWrIdMulti: empty wr_id list");
    return status_t::SUCCESS;
  }

  const int max_wcs = cq_capacity;
  std::vector<ibv_wc> wcs(max_wcs);

  std::unordered_set<uint64_t> pending(target_wr_ids.begin(), target_wr_ids.end());
  auto start_ts   = std::chrono::steady_clock::now();
  auto last_prog  = start_ts;
  size_t last_left = pending.size();

  while (!pending.empty()) {
    int n = ibv_poll_cq(cq, max_wcs, wcs.data());
    if (n < 0) {
      logError("waitWrIdMulti: poll CQ failed (n=%d, errno=%d %s)", n, errno, strerror(errno));
      return status_t::ERROR;
    }

    bool made_progress = false;

    if (n > 0) {
      for (int i = 0; i < n; ++i) {
        const auto &cqe = wcs[i];

        if (cqe.status != IBV_WC_SUCCESS) {
          logError("waitWrIdMulti: CQE error: wr_id=%lu status=%d vendor_err=0x%x",
                   cqe.wr_id, cqe.status, cqe.vendor_err);
          return status_t::ERROR;
        }

        auto it = pending.find(cqe.wr_id);
        if (it != pending.end()) {
          pending.erase(it);
          made_progress = true;

          if (pending.size() != last_left) {
            logDebug("waitWrIdMulti: WR %lu done, remaining=%zu", cqe.wr_id, pending.size());
            last_left = pending.size();
          }
        }
      }
    }

    if (made_progress) {
      last_prog = std::chrono::steady_clock::now();
      continue;
    }

#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(50));

    auto now = std::chrono::steady_clock::now();
    if (now - start_ts > timeout) {
      logError("waitWrIdMulti: timeout after %lld ms, still pending=%zu",
               (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - start_ts).count(),
               pending.size());

      for (size_t i = 0; i < qps_.size(); ++i) {
        if (!qps_[i]) continue;
        ibv_qp_attr attr;
        ibv_qp_init_attr init_attr;
        if (ibv_query_qp(qps_[i], &attr, IBV_QP_STATE, &init_attr) == 0) {
          logError("QP[%zu] state=%d (INIT=0, RTR=3, RTS=4, SQD=5, SQE=6, ERR=7)",
                   i, attr.qp_state);
        } else {
          logError("QP[%zu] ibv_query_qp failed errno=%d %s", i, errno, strerror(errno));
        }
      }

      for (auto wr : pending) {
        logError("Pending WR still not completed: %lu", wr);
      }
      return status_t::TIMEOUT;
    }
  }

  return status_t::SUCCESS;
}

} // namespace hmc
