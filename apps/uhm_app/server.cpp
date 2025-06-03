#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <mutex>
#include <future>
#include <string>

using namespace hmc;
using namespace std;
const size_t buffer_size = 2048ULL * 32; // 寒武纪最大：2048ULL * 1024 * 16  // 1024ULL * 256 -> 2^8
struct RecvContext {
  int id;
  Communicator* comm;
  uint8_t* dest_ptr;
  size_t size;
  std::mutex* log_mutex;
};
Memory *mem_op  = new Memory(0);
// UHM
void recv_channel_slice_uhm(RecvContext ctx) {
  size_t flags;
  if (ctx.comm->recvDataFrom(0, ctx.dest_ptr, ctx.size, MemoryType::MOORE_GPU, &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[Channel " << ctx.id << "] Receive failed.";
    return;
  }
}

// Serial: 分段接收（每个 chunk 最大 half_buffer_size）
void recv_channel_slice_serial(RecvContext ctx) {
  const size_t half_buffer_size = buffer_size / 2; // 来自全局变量或 main 中定义
  size_t remaining = ctx.size;
  uint8_t* ptr = ctx.dest_ptr;

  while (remaining > 0) {
    size_t chunk_size = std::min(half_buffer_size, remaining);
    size_t flags;

    if (ctx.comm->recvDataFrom(0, ptr, chunk_size, MemoryType::MOORE_GPU, &flags) != status_t::SUCCESS) {
      std::lock_guard<std::mutex> lock(*ctx.log_mutex);
      LOG(ERROR) << "[Channel " << ctx.id << "] Serial Receive failed.";
      return;
    }

    remaining -= chunk_size;
    ptr += chunk_size;
  }
}

// G2H2G: 使用 GPU 内存接收
void recv_channel_slice_g2h2g(RecvContext ctx) {
  size_t flags;
  if (ctx.comm->recvDataFrom(0, ctx.dest_ptr, ctx.size, MemoryType::CPU, &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[Channel " << ctx.id << "] Receive failed.";
    return;
  }
}

// RDMA over CPU memory
void recv_channel_slice_rdma_cpu(RecvContext ctx) {
  size_t flags;
  if (ctx.comm->recvDataFrom(0, ctx.dest_ptr, ctx.size, MemoryType::CPU, &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[Channel " << ctx.id << "] RDMA (CPU) Receive failed.";
    return;
  }
}

std::string get_mode_from_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      std::string mode = argv[++i];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" || mode == "rdma_cpu") {
        return mode;
      } else {
        std::cerr << "Invalid mode: " << mode << ". "
                  << "Supported modes: uhm / serial / g2h2g / rdma_cpu\n";
        exit(1);
      }
    }
  }
  return "uhm"; // 默认模式
}

int main(int argc, char* argv[]) {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  // ./server --mode=serial/uhm/g2h2g/rdma_cpu
  std::string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  const int channel_count = 1;
  const std::string server_ip = "192.168.2.238";
  const int base_port = 2025;

  const int device_id = 0;
  

  std::vector<std::shared_ptr<ConnBuffer>> buffers;
  std::vector<Communicator*> communicators;
  std::mutex log_mutex;

  // 设置接收函数
  void (*recv_func)(RecvContext) = recv_channel_slice_uhm; // 默认是 UHM
  if (mode == "serial") {
    recv_func = recv_channel_slice_serial;
  } else if (mode == "g2h2g") {
    recv_func = recv_channel_slice_g2h2g;
  } else if (mode == "rdma_cpu") {
    recv_func = recv_channel_slice_rdma_cpu;
  } else if (mode == "uhm") {
    recv_func = recv_channel_slice_uhm;
  } else {
    LOG(ERROR) << "Invalid mode: " << mode
               << ". Supported modes: uhm / serial / g2h2g / rdma_cpu";
    return -1;
  }

  for (int i = 0; i < channel_count; ++i) {
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
    Communicator* comm = new Communicator(buffer);
    if (comm->initServer(server_ip, base_port + i, ConnType::RDMA) != status_t::SUCCESS) {
      LOG(ERROR) << "Channel " << i << " server init failed.";
      return -1;
    }
    comm->addNewRankAddr(0, server_ip, 0);
    communicators.push_back(comm);
    buffers.push_back(buffer);
  }

  for (int i = 0; i < channel_count; ++i) {
    while (communicators[i]->checkConn(0, ConnType::RDMA) != status_t::SUCCESS) {
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    LOG(INFO) << "Channel " << i << " connected.";
  }

  LOG(INFO) << "All channels connected. Ready to receive.";

  for (int power = 3; power <= 28; ++power) {
    const size_t total_size = std::pow(2, power);
    size_t slice_size = total_size / channel_count;

    const int repeat = 1;
    double total_MBps = 0.0;

    for (int r = 0; r < repeat; ++r) {
      std::vector<uint8_t> full_data(total_size);
      std::vector<std::future<void>> futures;

      auto start_time = std::chrono::high_resolution_clock::now();
      void* slice_ptr;
      for (int i = 0; i < channel_count; ++i) {
        
        mem_op->allocateBuffer(&slice_ptr, total_size);
         
        size_t actual_size = (i == channel_count - 1)
                                ? total_size - i * slice_size
                                : slice_size;

        RecvContext ctx = {
            .id = i,
            .comm = communicators[i],
            .dest_ptr = (uint8_t*)slice_ptr,
            .size = actual_size,
            .log_mutex = &log_mutex
        };

        futures.emplace_back(std::async(std::launch::async, recv_func, ctx));
      }

      for (auto& f : futures) f.get();

      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end_time - start_time;
      double seconds = duration.count();

      double throughput_MBps = (total_size / 1024.0 / 1024.0) / seconds;
      double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

      total_MBps += throughput_MBps;
      mem_op->copyDeviceToHost(full_data.data(), slice_ptr, total_size);
      bool valid = true;
      for (size_t i = 0; i < std::min<size_t>(10, total_size); ++i) {
        if (full_data[i] != 'A') {
          valid = false;
          break;
        }
      }

      LOG(INFO) << "[Trial " << r + 1 << "] "
                << total_size / (1024 * 1024) << " MB received in "
                << seconds << " s, "
                << throughput_MBps << " MB/s ("
                << throughput_Gbps << " Gbps), "
                << "Data Integrity: " << (valid ? "PASS" : "FAIL");

      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    double avg_MBps = total_MBps / repeat;
    double avg_Gbps = avg_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    LOG(INFO) << ">>> Average over " << repeat << " trials: "
              << avg_MBps << " MB/s ("
              << avg_Gbps << " Gbps)";
    LOG(INFO) << "--------------------------------------------";

    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  for (int i = 0; i < channel_count; ++i) {
    communicators[i]->closeServer();
    delete communicators[i];
  }

  std::cout << "Server shutdown complete." << std::endl;
  return 0;
}