#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <thread>
#include <cmath>
#include <vector>

using namespace hmc;

struct ChannelContext {
  int id;
  Communicator* comm;
  void* data_ptr;
  size_t size;
};

void send_channel_slice(ChannelContext ctx) {
  auto status = ctx.comm->sendDataTo(0, ctx.data_ptr, ctx.size, MemoryType::CPU);
  if (status != status_t::SUCCESS) {
    LOG(ERROR) << "[Channel " << ctx.id << "] Send failed.";
  }
}

int main() {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  const int channel_count = 1;
  const std::string server_ip = "192.168.2.241";
  const int base_port = 2025;

  const int device_id = 1;
  const size_t buffer_size =  32 * 8 * 1024;

  std::vector<std::shared_ptr<ConnBuffer>> buffers;
  std::vector<Communicator*> communicators;

  for (int i = 0; i < channel_count; ++i) {
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
    Communicator* comm = new Communicator(buffer);
    comm->addNewRankAddr(0, server_ip, base_port + i);
    comm->connectTo(0, ConnType::RDMA);

    buffers.push_back(buffer);
    communicators.push_back(comm);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  for (int power = 20; power <= 30; ++power) {
    const size_t total_size = std::pow(2, power);
    std::vector<uint8_t> host_data(total_size, 'A');

    size_t slice_size = total_size / channel_count;
    std::vector<std::thread> threads;

    LOG(INFO) << "=== Sending " << total_size / (1024 * 1024)
              << " MB via " << channel_count << " channels ===";

    // 🌟 总开始时间
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

      LOG(INFO) << "=== slice size " << slice_size;
      LOG(INFO) << "=== actual size " << actual_size;

      threads.emplace_back(send_channel_slice, ctx);
    }

    for (auto& t : threads) t.join();

    // 🌟 总结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double seconds = duration.count();

    double throughput_MBps = (total_size / 1024.0 / 1024.0) / seconds;
    double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    LOG(INFO) << ">>> Total Size: " << total_size / (1024 * 1024) << " MB"
              << ", Time: " << seconds << " s"
              << ", Aggregate Throughput: "
              << throughput_MBps << " MB/s ("
              << throughput_Gbps << " Gbps)";

    std::this_thread::sleep_for(std::chrono::seconds(5)); // 间隔太短，有可能发串
  }

  for (int i = 0; i < channel_count; ++i) {
    communicators[i]->disConnect(0, ConnType::RDMA);
    delete communicators[i];
  }

  std::cout << "Client finished all transfers" << std::endl;
  return 0;
}

/*
8Gbps
|
v
4.85Gbps
|
v

*/