#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <future>
#include <fstream>
#include <string>

using namespace hmc;
using namespace std;
using namespace std::chrono;

struct ChannelContext {
  int id;
  Communicator* comm;
  void* data_ptr;
  size_t size;
  std::mutex* log_mutex;  // 用于线程安全的日志输出
};

// UHM
void send_channel_slice_uhm(ChannelContext ctx) {
  auto status = ctx.comm->sendDataTo(0, ctx.data_ptr, ctx.size, MemoryType::CPU);
  if (status != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[Channel " << ctx.id << "] Send failed.";
  }
}

// Serial 模式：分块发送
void send_channel_slice_serial(ChannelContext ctx) {
  // const size_t chunk_size = 2 * 1024 * 1024; // 2MB chunks
  // size_t remaining = ctx.size;
  // char* ptr = reinterpret_cast<char*>(ctx.data_ptr);

  // while (remaining > 0) {
  //   size_t send_size = min(chunk_size, remaining);
  //   auto status = ctx.comm->sendDataTo(0, ptr, send_size, MemoryType::AMD_GPU);
  //   if (status != status_t::SUCCESS) {
  //     std::lock_guard<std::mutex> lock(*ctx.log_mutex);
  //     LOG(ERROR) << "[Channel " << ctx.id << "] Serial Send failed at offset " << (ctx.size - remaining);
  //     return;
  //   }
  //   ptr += send_size;
  //   remaining -= send_size;
  // }
}

// G2H2G 模式：先拷贝到 Host，再发送 GPU 内存
void send_channel_slice_g2h2g(ChannelContext ctx) {
  // Memory mem(0);
  // void* host_buffer = nullptr;
  // mem.allocateBuffer(&host_buffer, ctx.size);
  // mem.copyDeviceToHost(host_buffer, ctx.data_ptr, ctx.size);

  // auto status = ctx.comm->sendDataTo(0, host_buffer, ctx.size, MemoryType::CPU);
  // if (status != status_t::SUCCESS) {
  //   std::lock_guard<std::mutex> lock(*ctx.log_mutex);
  //   LOG(ERROR) << "[Channel " << ctx.id << "] G2H2G Send failed.";
  // }

  // mem.freeBuffer(host_buffer);
}

// RDMA over CPU 内存
void send_channel_slice_rdma_cpu(ChannelContext ctx) {
  // auto status = ctx.comm->sendDataTo(0, ctx.data_ptr, ctx.size, MemoryType::CPU);
  // if (status != status_t::SUCCESS) {
  //   std::lock_guard<std::mutex> lock(*ctx.log_mutex);
  //   LOG(ERROR) << "[Channel " << ctx.id << "] RDMA-CPU Send failed.";
  // }
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

  // ./client --mode=serial/uhm/g2h2g/rdma_cpu
  std::string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  const int channel_count = 1;
  const std::string server_ip = "192.168.2.248";
  const int base_port = 2025;

  const int device_id = 0;
  const size_t buffer_size = 2048ULL * 1024 * 16; // 寒武纪最大：2048ULL * 1024 * 16; // 1024ULL * 256 -> 2^8

  std::vector<std::shared_ptr<ConnBuffer>> buffers;
  std::vector<Communicator*> communicators;
  std::mutex log_mutex;

  // 设置发送函数
  void (*send_func)(ChannelContext) = send_channel_slice_uhm;
  if (mode == "serial") {
    send_func = send_channel_slice_serial;
  } else if (mode == "g2h2g") {
    send_func = send_channel_slice_g2h2g;
  } else if (mode == "rdma_cpu") {
    send_func = send_channel_slice_rdma_cpu;
  } else if (mode == "uhm") {
    send_func = send_channel_slice_uhm;
  } else {
    LOG(ERROR) << "Invalid mode: " << mode
               << ". Supported modes: uhm / serial / g2h2g / rdma_cpu";
    return -1;
  }

  for (int i = 0; i < channel_count; ++i) {
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
    Communicator* comm = new Communicator(buffer);
    comm->addNewRankAddr(0, server_ip, base_port + i);
    comm->connectTo(0, ConnType::RDMA);

    buffers.push_back(buffer);
    communicators.push_back(comm);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  ofstream csv_file("performanceTest_client.csv", ios::app);
  if (csv_file.tellp() == 0) {
    csv_file << "Mode,Data Size (MB),Avg Throughput (MB/s),Avg Bandwidth (Gbps)\n";
  }
  csv_file.close();

  for (int power = 20; power <= 28; ++power) {
  const size_t total_size = std::pow(2, power);
  std::vector<uint8_t> host_data(total_size, 'A');
  size_t slice_size = total_size / channel_count;

  const int repeat = 1;
  double total_MBps = 0.0;

  for (int r = 0; r < repeat; ++r) {
    std::vector<std::future<void>> futures;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < channel_count; ++i) {
      void* slice_ptr = host_data.data() + i * slice_size;
      size_t actual_size = (i == channel_count - 1)
                              ? total_size - i * slice_size
                              : slice_size;

      ChannelContext ctx = {
          .id = i,
          .comm = communicators[i],
          .data_ptr = slice_ptr,
          .size = actual_size};

      futures.emplace_back(std::async(std::launch::async, send_func, ctx));
    }

    for (auto& f : futures) f.get();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;
    double seconds = duration.count();

    double throughput_MBps = (total_size / 1024.0 / 1024.0) / seconds;
    double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    total_MBps += throughput_MBps;

    LOG(INFO) << "[Trial " << r + 1 << "] "
              << total_size << " B "
              << total_size / (1024) << " KB "
              << total_size / (1024 * 1024) << " MB sent in "
              << seconds << " s, "
              << throughput_MBps << " MB/s ("
              << throughput_Gbps << " Gbps)";
    std::this_thread::sleep_for(std::chrono::seconds(3));
  }

  double avg_MBps = total_MBps / repeat;
  double avg_Gbps = avg_MBps * 1024.0 * 1024.0 * 8 / 1e9;

  LOG(INFO) << ">>> Average over " << repeat << " trials: "
            << avg_MBps << " MB/s ("
            << avg_Gbps << " Gbps)";
  LOG(INFO) << "--------------------------------------------";

  // 写入 CSV
  ofstream file("performanceTest_client.csv", ios::app);
  if (file.is_open()) {
    file << mode << ","
         << total_size / (1024 * 1024) << ","
         << avg_MBps << ","
         << avg_Gbps << "\n";
    file.close();
  }

  std::this_thread::sleep_for(std::chrono::seconds(3));
}

  for (int i = 0; i < channel_count; ++i) {
    communicators[i]->disConnect(0, ConnType::RDMA);
    delete communicators[i];
  }

  std::cout << "Client finished all transfers" << std::endl;
  return 0;
}