#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <future>

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
  const std::string server_ip = "192.168.2.248";
  const int base_port = 2025;

  const int device_id = 0;
  const size_t buffer_size = 2048ULL * 1024 * 16; // 32MB

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

  for (int power = 24; power <= 26; ++power) {
  const size_t total_size = std::pow(2, power);
  std::vector<uint8_t> host_data(total_size, 'A');
  size_t slice_size = total_size / channel_count;

  const int repeat = 3;
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

      futures.emplace_back(std::async(std::launch::async, send_channel_slice, ctx));
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

  std::this_thread::sleep_for(std::chrono::seconds(3));
}

  for (int i = 0; i < channel_count; ++i) {
    communicators[i]->disConnect(0, ConnType::RDMA);
    delete communicators[i];
  }

  std::cout << "Client finished all transfers" << std::endl;
  return 0;
}