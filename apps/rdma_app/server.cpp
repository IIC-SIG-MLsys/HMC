#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <mutex>
#include <future>

using namespace hmc;

struct RecvContext {
  int id;
  Communicator* comm;
  uint8_t* dest_ptr;
  size_t size;
  std::mutex* log_mutex;
};

void recv_channel_slice(RecvContext ctx) {
  size_t flags;
  if (ctx.comm->recvDataFrom(0, ctx.dest_ptr, ctx.size, MemoryType::CPU, &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[Channel " << ctx.id << "] Receive failed.";
    return;
  }
}

int main() {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  const int channel_count = 1;
  const std::string server_ip = "192.168.2.241";
  const int base_port = 2025;

  const int device_id = 1;
  const size_t buffer_size = 2048ULL * 1024 * 16; // 32MB

  std::vector<std::shared_ptr<ConnBuffer>> buffers;
  std::vector<Communicator*> communicators;
  std::mutex log_mutex;

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

  for (int power = 25; power <= 30; ++power) {
  const size_t total_size = std::pow(2, power);
  size_t slice_size = total_size / channel_count;

  const int repeat = 3;
  double total_MBps = 0.0;

  for (int r = 0; r < repeat; ++r) {
    std::vector<uint8_t> full_data(total_size);
    std::vector<std::future<void>> futures;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < channel_count; ++i) {
      uint8_t* slice_ptr = full_data.data() + i * slice_size;
      size_t actual_size = (i == channel_count - 1)
                              ? total_size - i * slice_size
                              : slice_size;

      RecvContext ctx = {
          .id = i,
          .comm = communicators[i],
          .dest_ptr = slice_ptr,
          .size = actual_size,
          .log_mutex = &log_mutex
      };

      futures.emplace_back(std::async(std::launch::async, recv_channel_slice, ctx));
    }

    for (auto& f : futures) f.get();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double seconds = duration.count();

    double throughput_MBps = (total_size / 1024.0 / 1024.0) / seconds;
    double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    total_MBps += throughput_MBps;

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
