/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_rdma.h"

namespace hmc {

RDMAEndpoint::RDMAEndpoint(std::shared_ptr<ConnBuffer> buffer)
    : buffer(buffer){};

RDMAEndpoint::~RDMAEndpoint() {
  if (role == EndpointType::Client) {
    closeEndpoint(); // 目前只有client进行断开操作
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
    logError("Invalid data bias and size");
    return status_t::ERROR;
  }
  if (writeDataNB(data_bias, size) != status_t::SUCCESS) {
    logError("Error for post_write");
    return status_t::ERROR;
  }
  if (pollCompletion(1) != status_t::SUCCESS) {
    logError("Error for waiting writeData finished");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::readData(size_t data_bias, size_t size) {
  // 实现读取远端数据逻辑
  if (data_bias + size > buffer->buffer_size) {
    logError("Invalid data bias and size");
    return status_t::ERROR;
  }
  if (readDataNB(data_bias, size) != status_t::SUCCESS) {
    logError("Error for post_read");
    return status_t::ERROR;
  }
  if (pollCompletion(1) != status_t::SUCCESS) {
    logError("Error for waiting writeData finished");
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::writeDataNB(size_t data_bias, size_t size) {
  void *localAddr = static_cast<char *>(buffer->ptr) + data_bias;
  void *remoteAddr =
      reinterpret_cast<char *>(remote_metadata_attr.address) + data_bias;
  return postWrite(localAddr, remoteAddr, size, buffer_mr,
                   remote_metadata_attr.key, true);
}

status_t RDMAEndpoint::readDataNB(size_t data_bias, size_t size) {
  void *localAddr = static_cast<char *>(buffer->ptr) + data_bias;
  void *remoteAddr =
      reinterpret_cast<char *>(remote_metadata_attr.address) + data_bias;
  return postRead(localAddr, remoteAddr, size, buffer_mr,
                  remote_metadata_attr.key, true);
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

status_t RDMAEndpoint::pollCompletion(int num_completions_to_process) {
  // 检查 cq 是否为空指针
  if (cq == nullptr) {
    logError("Completion queue is null, check if endpoint is ok?");
    return status_t::ERROR;
  }

  // 分配足够的空间来存储多个完成事件
  const int max_wcs = cq_capacity; // 可根据实际情况调整大小
  std::vector<struct ibv_wc> wcs(max_wcs);

  int total_processed = 0;

  while (total_processed < num_completions_to_process) {
    // 尽可能多地获取完成事件
    int num_completions = ibv_poll_cq(cq, max_wcs, wcs.data());

    if (num_completions < 0) {
      logError("Failed to poll completion queue");
      return status_t::ERROR;
    }

    for (int i = 0;
         i < num_completions && total_processed < num_completions_to_process;
         ++i) {
      struct ibv_wc &wc = wcs[i];

      if (wc.status != IBV_WC_SUCCESS) {
        switch (wc.status) {
        case IBV_WC_LOC_LEN_ERR:
          logError("WC status: Local Length Error");
          break;
        case IBV_WC_LOC_QP_OP_ERR:
          logError("WC status: Local QP Operation Error");
          break;
        case IBV_WC_LOC_EEC_OP_ERR:
          logError("WC status: Local EE Context Operation Error");
          break;
        case IBV_WC_LOC_PROT_ERR:
          logError("WC status: Local Protection Error");
          break;
        case IBV_WC_WR_FLUSH_ERR:
          logError("WC status: Work Request Flushed Error");
          break;
        case IBV_WC_MW_BIND_ERR:
          logError("WC status: Memory Window Binding Error");
          break;
        case IBV_WC_REM_ACCESS_ERR:
          logError("WC status: Remote Access Error");
          break;
        case IBV_WC_REM_OP_ERR:
          logError("WC status: Remote Operation Error");
          break;
        case IBV_WC_RNR_RETRY_EXC_ERR:
          logError("WC status: RNR Retry Counter Exceeded");
          break;
        case IBV_WC_RETRY_EXC_ERR:
          logError("WC status: Transport Retry Counter Exceeded");
          break;
        default:
          logError("WC status: Unknown error (%d)", wc.status);
          break;
        }
        return status_t::ERROR;
      } else {
        // 处理成功的完成事件
        logDebug("Completion event processed successfully");
      }

      total_processed++;
    }

    // 如果没有更多的完成事件，则退出循环
    if (num_completions == 0) {
      break;
    }
  }

  return status_t::SUCCESS;
}

status_t RDMAEndpoint::uhm_send(void *input_buffer, const size_t send_flags, MemoryType mem_type) {
  const size_t half_buffer_size = buffer->buffer_size / 2;
  const size_t num_send_chunks =
      (send_flags + half_buffer_size - 1) / half_buffer_size;

  // 标志位初始化, send使用remote_metadata_attr
  remote_metadata_attr.uhm_buffer_state.state[0] = UHM_BUFFER_CAN_WRITE;
  remote_metadata_attr.uhm_buffer_state.state[1] = UHM_BUFFER_CAN_WRITE;

  size_t current_chunk = 0;
  status_t ret = status_t::SUCCESS;

  // 预先拷贝第一个chunk的数据
  size_t chunk_index = 0;
  size_t send_size = std::min(half_buffer_size, send_flags);

  mem_type == MemoryType::CPU ? buffer->writeFromCpu(input_buffer, send_size) : buffer->writeFromGpu(input_buffer, send_size);

  while (current_chunk < num_send_chunks) {
    chunk_index = current_chunk % 2;
    size_t next_chunk_index = (current_chunk + 1) % 2;
    logDebug(
        "Client::Send: current_chunk %ld, chunk_index %ld, num_send_chunks %ld",
        current_chunk, chunk_index, num_send_chunks);

    // todo
  
  }

  return status_t::SUCCESS;
}
status_t RDMAEndpoint::uhm_recv(void *output_buffer, const size_t buffer_size,
                      size_t *recv_flags, MemoryType mem_type) {
  // 标志位初始化, recv使用local_metadata_attr

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

  /** TODO：在这里增加，支持更多的数据buffer; 在exchangeMetadata完成数据信息交换
   * **/

  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postSend(void *addr, size_t length, struct ibv_mr *mr,
                                bool signaled) {
  struct ibv_send_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)addr;
  sge.length = length;
  sge.lkey = mr->lkey;

  wr.wr_id = 0;
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

status_t RDMAEndpoint::postRecv(void *addr, size_t length, struct ibv_mr *mr) {
  struct ibv_recv_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)addr;
  sge.length = length;
  sge.lkey = mr->lkey;

  wr.wr_id = 0;
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
                                 uint32_t remote_key, bool signaled) {
  struct ibv_send_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)local_addr;
  sge.length = length;
  sge.lkey = local_mr->lkey;

  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0; // 带信号，则后续处理cq即可
  wr.wr.rdma.remote_addr = (uint64_t)remote_addr;
  wr.wr.rdma.rkey = remote_key;

  if (ibv_post_send(qp, &wr, &bad_wr)) {
    logError("Failed to post RDMA write");
    if (bad_wr) {
      fprintf(stderr, "Failed wr_id: %lu\n", bad_wr->wr_id);
    }
    return status_t::ERROR;
  }
  return status_t::SUCCESS;
}

status_t RDMAEndpoint::postRead(void *local_addr, void *remote_addr,
                                size_t length, struct ibv_mr *local_mr,
                                uint32_t remote_key, bool signaled) {
  struct ibv_send_wr wr, *bad_wr = nullptr;
  struct ibv_sge sge;

  memset(&wr, 0, sizeof(wr));
  memset(&sge, 0, sizeof(sge));

  sge.addr = (uint64_t)local_addr;
  sge.length = length;
  sge.lkey = local_mr->lkey;

  wr.wr_id = 0;
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

} // namespace hmc