/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_rdma.h"

namespace hmc {

RDMAEndpoint::RDMAEndpoint(std::shared_ptr<ConnBuffer> buffer)
    : buffer(buffer){};

RDMAEndpoint::~RDMAEndpoint() {
  if (role == EndpointType::Client && connStatus == status_t::SUCCESS) {
    closeEndpoint(); // 目前只有在连接正常时，在client侧进行断开操作
  }
  cleanRdmaResources();
};

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

status_t RDMAEndpoint::writeData(size_t data_bias, size_t size) {
  // 实现写入远端数据逻辑
  if (data_bias + size > buffer->buffer_size) {
    logError("Invalid data bias and size, buffer_size %ld, data_bias+size %ld", buffer->buffer_size, data_bias+size);
    return status_t::ERROR;
  }
  uint64_t wr_id;
  if (writeDataNB(data_bias, size, &wr_id) != status_t::SUCCESS) {
    logError("Error for post_write");
    return status_t::ERROR;
  }
  if (waitWrId(wr_id) != status_t::SUCCESS) {
    logError("Error for waiting writeData finished");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::writeDataNB(size_t data_bias, size_t size, uint64_t* wr_id) {
  void *localAddr = static_cast<char *>(buffer->ptr) + data_bias;
  void *remoteAddr =
      reinterpret_cast<char *>(remote_metadata_attr.address) + data_bias;
  *wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
  return postWrite(localAddr, remoteAddr, size, buffer_mr,
                   remote_metadata_attr.key, *wr_id, true);
}

status_t RDMAEndpoint::readData(size_t data_bias, size_t size) {
  // 实现读取远端数据逻辑
  if (data_bias + size > buffer->buffer_size) {
    logError("Invalid data bias and size, buffer_size %ld, data_bias+size %ld", buffer->buffer_size, data_bias+size);
    return status_t::ERROR;
  }
  uint64_t wr_id;
  if (readDataNB(data_bias, size, &wr_id) != status_t::SUCCESS) {
    logError("Error for post_read");
    return status_t::ERROR;
  }
  if (waitWrId(wr_id) != status_t::SUCCESS) {
    logError("Error for waiting writeData finished");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::readDataNB(size_t data_bias, size_t size, uint64_t* wr_id) {
  void *localAddr = static_cast<char *>(buffer->ptr) + data_bias;
  void *remoteAddr =
      reinterpret_cast<char *>(remote_metadata_attr.address) + data_bias;
  *wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
  return postRead(localAddr, remoteAddr, size, buffer_mr,
                  remote_metadata_attr.key, *wr_id, true);
}

status_t RDMAEndpoint::registerMemory(void *addr, size_t length,
                                      struct ibv_mr **mr) {
  if (!addr || !mr) {
    return status_t::ERROR;
  }
  *mr = ibv_reg_mr(pd, addr, length,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_WRITE);
  if (!*mr) {
    std::cout << pd << " " << addr << " " << length << std::endl;
    logError("Failed to register memory region");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::deRegisterMemory(struct ibv_mr *mr) {
  if (!mr) {
    return status_t::ERROR;
  }

  if (ibv_dereg_mr(mr)) {
    logError("Failed to deregister memory region");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::uhm_send(void *input_buffer, const size_t send_flags, MemoryType mem_type) {
  status_t ret;

  const size_t half_buffer_size = buffer->buffer_size / 2;
  const size_t num_send_chunks = (send_flags + half_buffer_size - 1) / half_buffer_size;
  size_t current_chunk = 0;
  size_t chunk_index = 0;
  size_t send_size = std::min(half_buffer_size, send_flags);

  // 发送标志位，通知对面要发送多少数据
  
  uhm_buffer_state.state[0] = UHM_BUFFER_CAN_WRITE;
  uhm_buffer_state.state[1] = UHM_BUFFER_CAN_WRITE;
  // logDebug("send flags is %ld", send_flags); // 有概率触发读取不到最新值的情况，主动赋值或者打印会更新缓存
  uhm_buffer_state.length = send_flags;

  void *localAddr = reinterpret_cast<char *>(&uhm_buffer_state);
  void *remoteAddr = reinterpret_cast<char *>(remote_metadata_attr.uhm_buffer_state_address);
  uint64_t wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
  ret = postWrite(localAddr, remoteAddr, sizeof(uhm_buffer_state), uhm_buffer_state_mr, remote_metadata_attr.uhm_buffer_state_key, wr_id, true);
  if (ret != status_t::SUCCESS) {
    logError("Client::Send: Failed to post write buffer state");
    return ret;
  }
  // 首次缓冲区拷贝
  mem_type == MemoryType::CPU ? buffer->writeFromCpu(input_buffer, send_size, 0) : buffer->writeFromGpu(input_buffer, send_size, 0);
  // 处理发送事件
  if (waitWrId(wr_id) != status_t::SUCCESS) {
    logError("Client::Send: Failed to poll completion for chunk %zu",
              current_chunk);
    return status_t::ERROR;
  }

  while (current_chunk < num_send_chunks) {
    chunk_index = current_chunk % 2;
    size_t next_chunk_index = (current_chunk + 1) % 2;
    // logDebug(
    //     "Client::Send: current_chunk %ld, chunk_index %ld, num_send_chunks %ld",
    //     current_chunk, chunk_index, num_send_chunks);

    // 远端会通过write改变本地uhm_buffer_state对应位置的状态
    // 首先检查状态能不能写入远端
    const int SPIN_COUNT =
        100; // 3ghz的cpu，4000次大约相当于200us，100相当于5us

    while (uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
      int spin = 0;
      while (spin++ < SPIN_COUNT &&
             uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
        // 使用 CPU pause 指令，减少功耗并优化自旋等待
        #if defined(__x86_64__) || defined(_M_X64)
        __builtin_ia32_pause();
        #elif defined(__aarch64__)
        asm volatile("yield");
        #endif
      }
      if (uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
        // 超过自旋次数后短暂让出CPU
        std::this_thread::yield();
      }
    }

    // 计算当前块的实际大小
    size_t remaining = send_flags - current_chunk * half_buffer_size;
    send_size = std::min(half_buffer_size, remaining);

    // 可写，写当前块到远端
    ret = writeDataNB(chunk_index * half_buffer_size, send_size, &wr_id);
    if (ret != status_t::SUCCESS) {
      logError("Client::Send: Failed to post write data for chunk %zu", current_chunk);
      return ret;
    }
    // logDebug("Client::Send success");

    // 发送标志位
    uhm_buffer_state.state[chunk_index] = UHM_BUFFER_CAN_READ;
    void *localAddr = reinterpret_cast<char *>(&uhm_buffer_state) + chunk_index * sizeof(UHM_STATE_TYPE);
    void *remoteAddr = reinterpret_cast<char *>(remote_metadata_attr.uhm_buffer_state_address) + chunk_index * sizeof(UHM_STATE_TYPE);
    ret = postWrite(localAddr, remoteAddr, sizeof(UHM_STATE_TYPE), uhm_buffer_state_mr, remote_metadata_attr.uhm_buffer_state_key, wr_id, true);
    if (ret != status_t::SUCCESS) {
      logError("Client::Send: Failed to post write buffer state");
      return ret;
    }

    // rdma对一个qp内的事件做保序
    // 只有当还有下一块数据时才进行拷贝
    if (current_chunk + 1 < num_send_chunks) {
      size_t next_remaining = send_flags - (current_chunk + 1) * half_buffer_size;
      size_t next_size = std::min(half_buffer_size, next_remaining);

      size_t bias = next_chunk_index * half_buffer_size;
      // void* src = static_cast<char *>(input_buffer) + (current_chunk + 1) * half_buffer_size;
      mem_type == MemoryType::CPU ? buffer->writeFromCpu(input_buffer, next_size, bias) : buffer->writeFromGpu(input_buffer, next_size, bias);
    }

    // 等待向远端写操作完成
    if (waitWrId(wr_id) != status_t::SUCCESS) {
      logError("Client::Send: Failed to poll completion for chunk %zu", current_chunk);
      return status_t::ERROR;
    }

    // logDebug("Client::Send: sent chunk %zu with size %zu", current_chunk, send_size);
    current_chunk++;
  }

  // logDebug("Client::Send: sent finished");
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

  // 初始化uhm_buffer_state，为了接受消息，先置可读
  uhm_buffer_state.state[0] = UHM_BUFFER_CAN_READ;
  uhm_buffer_state.state[1] = UHM_BUFFER_CAN_READ;
  uhm_buffer_state.length = 0;
  // logDebug("Server::Recv: uhm_buffer_state %d", this->uhm_buffer_state.state[0]);

  // 接收要传入的块数量
  const int SPIN_COUNT = 100; // 3ghz的cpu，4000次大约相当于200us，10:0.05us
  while (true) {
      auto tmp = uhm_buffer_state;
      if (tmp.state[chunk_index] == UHM_BUFFER_CAN_WRITE) {
        // 检查接收大小的合法性
        *recv_flags  = tmp.length;
        if (*recv_flags == 0) {
          logError("Server::Recv: Invalid receive size is 0");
          return status_t::ERROR;
        }
        num_recv_chunks = (*recv_flags + half_buffer_size - 1) / half_buffer_size;
        this->uhm_buffer_state.state[0] = UHM_BUFFER_CAN_WRITE; // 正确获取到flag，开始等对方写
        this->uhm_buffer_state.state[1] = UHM_BUFFER_CAN_WRITE;
        break;
      }

      // hygon cpu has bug，which need a small time block.
      int spin = 0;
      while (spin++ < SPIN_COUNT &&
             this->uhm_buffer_state.state[chunk_index] != UHM_BUFFER_CAN_WRITE) {
      // 使用 CPU pause 指令，减少功耗并优化自旋等待
      #if defined(__x86_64__) || defined(_M_X64)
              __builtin_ia32_pause();
      #elif defined(__aarch64__)
              asm volatile("yield");
      #endif
      }
  }
  // logDebug("recv_flags %zu, current chunk %zu, num_recv_chunks %zu\n", *recv_flags, current_chunk, num_recv_chunks);
  uint64_t wr_id;
  while (current_chunk < num_recv_chunks) {
    chunk_index = current_chunk % 2;
    // logDebug("current_chunk %zu, chunk_index %zu, uhm_buffer_state.state[chunk_index] %u\n", current_chunk, chunk_index, uhm_buffer_state.state[chunk_index]);
    // logDebug会触发缓存刷新，当前方案采用volite对state加缓存屏障解决此问题
    if (this->uhm_buffer_state.state[chunk_index] == UHM_BUFFER_CAN_READ) {
      // 直接同步通知对面可写
      this->uhm_buffer_state.state[chunk_index] = UHM_BUFFER_CAN_WRITE; 
      void *localAddr = reinterpret_cast<char *>(&uhm_buffer_state) + chunk_index * sizeof(UHM_STATE_TYPE);
      void *remoteAddr = reinterpret_cast<char *>(remote_metadata_attr.uhm_buffer_state_address) + chunk_index * sizeof(UHM_STATE_TYPE);
      wr_id = next_wr_id_.fetch_add(1, std::memory_order_relaxed);
      ret = postWrite(localAddr, remoteAddr, sizeof(UHM_STATE_TYPE), uhm_buffer_state_mr, remote_metadata_attr.uhm_buffer_state_key, wr_id, true);   
      if (ret != status_t::SUCCESS) {
        logError("Server::Recv: Failed to post write buffer state");
        return ret;
      }

      // 接收大小
      if(current_chunk == num_recv_chunks - 1){ // 最后一次接收
        recv_size = *recv_flags - accumulated_size;
        logDebug("Server::Recv: Receive flag is %ld, accumulated size is %ld", *recv_flags, accumulated_size);
      } else {
        recv_size = half_buffer_size;
      }
      // 检查接收大小的合法性
      if (recv_size == 0) {
        logError("Server::Recv: Invalid receive size is 0");
        return status_t::ERROR;
      } else if (recv_size > buffer_size)
      {
        logError("Server::Recv: Invalid receive size %zu is bigger than buffer size %zu", recv_size, buffer_size);
        return status_t::ERROR;
      } else if (accumulated_size + recv_size > buffer_size)
      {
        logError("Server::Recv: Invalid accumulated_size + recv_size > buffer_size");
        return status_t::ERROR;
      }

      // 拷贝到输出缓冲区
      size_t bias = chunk_index * half_buffer_size;
      void* dest = static_cast<char *>(output_buffer) + accumulated_size;
      mem_type == MemoryType::CPU ? buffer->readToCpu(dest, recv_size, bias) : buffer->readToGpu(dest, recv_size, bias);
    
      // 处理通知完成消息，防止漏掉结束消息
      if (waitWrId(wr_id) != status_t::SUCCESS) {
        logError("Failed to poll completion queue");
        return status_t::ERROR;
      }

      // logDebug("current chunk %zu, total chunk %zu\n", current_chunk, num_recv_chunks);
      accumulated_size += recv_size;
      current_chunk++;
    }
  };

  return status_t::SUCCESS;
}

status_t RDMAEndpoint::setupBuffers() {
  /** buffer 的相关数据准备 **/
  // 注册本地元数据缓冲区
  status_t ret = registerMemory(
      &local_metadata_attr, sizeof(local_metadata_attr), &local_metadata_mr);
  if (ret != status_t::SUCCESS) {
    logError("Error while register local_metadata_attr");
    return ret;
  }
  // 注册远程元数据缓冲区
  ret = registerMemory(&remote_metadata_attr, sizeof(remote_metadata_attr),
                       &remote_metadata_mr);
  if (ret != status_t::SUCCESS) {
    logError("Error while register remote_metadata_attr");
    return ret;
  }
  if (buffer == NULL) {
    logError("Error while register buffer, buffer is NULL");
    return ret;
  }
  // 注册缓冲区
  ret = registerMemory(buffer->ptr, buffer->buffer_size, &buffer_mr);
  if (ret != status_t::SUCCESS) {
    logError("Error while register buffer %p", buffer.get());
    return ret;
  }
  // 设置本地元数据属性
  local_metadata_attr.address = (uint64_t)buffer->ptr;
  local_metadata_attr.length = buffer->buffer_size;
  local_metadata_attr.key = buffer_mr->lkey;

  /** TODO：在这里增加，支持更多的数据buffer; 在metadata里面存好相关地址和key
   * **/
  ret = registerMemory(&uhm_buffer_state, sizeof(uhm_buffer_state),
                       &uhm_buffer_state_mr);
  if (ret != status_t::SUCCESS) {
    logError("Error while register uhm_buffer_state");
    return ret;
  }
  local_metadata_attr.uhm_buffer_state_address = (uint64_t)&uhm_buffer_state;
  local_metadata_attr.uhm_buffer_state_key = uhm_buffer_state_mr->lkey;

  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postSend(void *addr, size_t length, struct ibv_mr *mr, uint64_t wr_id,
                                bool signaled) {
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

  if (ibv_post_send(qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA send");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postRecv(void *addr, size_t length, struct ibv_mr *mr, uint64_t wr_id) {
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

  if (ibv_post_recv(qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA recv");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postWrite(void *local_addr, void *remote_addr,
                                 size_t length, struct ibv_mr *local_mr,
                                 uint32_t remote_key, uint64_t wr_id, bool signaled) {
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
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0; // 带信号，则后续处理cq即可
  wr.wr.rdma.remote_addr = (uint64_t)remote_addr;
  wr.wr.rdma.rkey = remote_key;

  if (ibv_post_send(qp, &wr, &bad_wr)) {
    int e = errno;
    logError("ibv_post_send failed: errno=%d (%s)", e, strerror(e));
    if (bad_wr) fprintf(stderr, "Failed wr_id: %lu\n", bad_wr->wr_id);

    ibv_qp_attr attr; ibv_qp_init_attr init_attr;
    if (ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr) == 0) {
      logError("QP state after failure: %d", attr.qp_state);
    }
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postRead(void *local_addr, void *remote_addr,
                                size_t length, struct ibv_mr *local_mr,
                                uint32_t remote_key, uint64_t wr_id, bool signaled) {
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

  if (ibv_post_send(qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA read");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

void RDMAEndpoint::showRdmaBufferAttr(const struct rdma_buffer_attr *attr) {
  logInfo("Buffer Attr:");
  logInfo("  address: 0x%lx\n", attr->address);
  logInfo("  length: %u\n", attr->length);
  logInfo("  key: 0x%x\n", attr->key);
  logInfo("  uhm_buffer_state address: 0x%lx\n", attr->uhm_buffer_state_address);
}

void RDMAEndpoint::cleanRdmaResources() {
  if (qp) {
    ibv_destroy_qp(qp);
    qp = nullptr;
  }
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
  if (cm_id &&
      cm_id
          ->context) { // 清理到这里，context失效了，没走rdma_destroy_id,可能有泄漏风险
    rdma_destroy_id(cm_id);
    cm_id = nullptr;
  }

  // 注意 buffer 的清理工作由外部的ConnMemory维护，Endpoint不对buffer进行处理
}

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


} // namespace hmc