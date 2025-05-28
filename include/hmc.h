/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_H
#define HMC_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
// #include <cuda.h>
#endif
#ifdef ENABLE_ROCM
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif
#ifdef ENABLE_MUSA
#include <musa.h>
#include <musa_runtime.h>
#endif
#ifdef ENABLE_NEUWARE
#include "cn_api.h" // CNresult
#include "cnrt.h"
#include "mlu_op.h"
#endif

#include "mem.h"
#include "status.h"

#include <atomic>
#include <chrono>
#include <csignal> // For signal
#include <cstdlib>
#include <iostream>
#include <memory>
#include <queue>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility> // For std::pair

namespace hmc {

/* communicator */
class ConnManager;
class Endpoint;

class ConnBuffer {
public:
  void *ptr = nullptr;
  size_t buffer_size;
  Memory *mem_ops = nullptr;

  ConnBuffer(int device_id, size_t buffer_size,
             MemoryType mem_type = MemoryType::DEFAULT);
  // buffer必须在首次分配，并且不允许重新分配，否则指针发生改变，会导致通信的缓冲区失效

  status_t writeFromCpu(void *src, size_t size, size_t bias = 0);
  status_t readToCpu(void *dest, size_t size, size_t bias = 0);
  status_t writeFromGpu(void *src, size_t size, size_t bias = 0);
  status_t readToGpu(void *src, size_t size, size_t bias = 0);

  ~ConnBuffer();
};

enum class ConnType { RDMA, UCX };

class Communicator {
private:
  std::unordered_map<uint32_t, std::pair<std::string, uint16_t>> rank_addr_map; // rank: ip, port
  std::shared_ptr<ConnBuffer> buffer;
  std::shared_ptr<ConnManager>
      conn_manager; // must be shared, enable shared obj

public:
  Communicator(std::shared_ptr<ConnBuffer> buffer);

  status_t writeTo(uint32_t node_rank, size_t ptr_bias, size_t size,
                   ConnType connType = ConnType::RDMA);
  status_t readFrom(uint32_t node_rank, size_t ptr_bias, size_t size,
                    ConnType connType = ConnType::RDMA);

  status_t sendDataTo(uint32_t node_rank, void *send_buf, size_t buf_size, MemoryType buf_type); // uhm interface, for big data
  status_t recvDataFrom(uint32_t node_rank, void *recv_buf, size_t buf_size, MemoryType buf_type, size_t *flag); // flag, recv size

  status_t initServer(std::string ip, uint16_t port, ConnType serverType);
  status_t closeServer();
  status_t connectTo(uint32_t node_rank, ConnType connType);
  status_t disConnect(uint32_t node_rank, ConnType connType);
  status_t checkConn(uint32_t node_rank, ConnType connType);
  status_t checkOrNewConn(uint32_t node_rank, ConnType connType, std::string *ip);

  status_t addNewRankAddr(uint32_t rank, std::string ip, uint16_t port);
  status_t delRankAddr(uint32_t rank);

  ~Communicator();

private:
  const std::pair<std::string, uint16_t> *_getAddrByRank(uint32_t node_rank);
};

} // namespace hmc

#endif