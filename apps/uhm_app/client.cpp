#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <mutex>
#include "../src/memories/mem_type.h"
#include "../src/resource_manager/gpu_interface.h"

using namespace hmc;
using namespace std;
using namespace std::chrono;

const std::string server_ip = "192.168.2.248";
const std::string client_ip = "192.168.2.248";
size_t buffer_size = 2048ULL * 32;
const int device_id = 0;
const int gpu_port = 2025;
const int cpu_port = 2026;
Communicator *gpu_comm;
Communicator *cpu_comm;
std::shared_ptr<ConnBuffer> gpu_buffer;
std::shared_ptr<ConnBuffer> cpu_buffer;
Memory* gpu_mem_op = new Memory(device_id);
Memory* cpu_mem_op = new Memory(0, MemoryType::CPU);

struct Context {
  void* cpu_data_ptr;
  void* gpu_data_ptr;
  size_t size;
  std::mutex* log_mutex;
};

long long total_time = 0;
std::mutex log_mutex;

void send_channel_slice_serial(Context ctx) {
  // total_time = 0;
  // const size_t chunk_size = buffer_size / 2;
  // size_t remaining = ctx.size;
  // char* ptr = reinterpret_cast<char*>(ctx.data_ptr);
  // std::vector<uint8_t> host_data1(chunk_size, 'A');

  // while (remaining > 0) {
  //   size_t send_size = min(chunk_size, remaining);
  //   auto start = high_resolution_clock::now();

  //   mem_op->copyHostToDevice(ptr, host_data1.data(), send_size);
  //   auto status = ctx.comm->sendDataTo(0, ptr, send_size, MemoryType::DEFAULT);
  //   auto end = high_resolution_clock::now();

  //   if (status != status_t::SUCCESS) {
  //     std::lock_guard<std::mutex> lock(*ctx.log_mutex);
  //     LOG(ERROR) << "[Serial] Send failed at offset " << (ctx.size - remaining);
  //     return;
  //   }

  //   total_time += duration_cast<microseconds>(end - start).count();
  //   ptr += send_size;
  //   remaining -= send_size;
  // }
}

void send_channel_slice_uhm(Context ctx) {
  auto start = high_resolution_clock::now();
  auto status = gpu_comm->sendDataTo(server_ip, ctx.gpu_data_ptr, ctx.size, MemoryType::DEFAULT);
  auto end = high_resolution_clock::now();
  total_time = duration_cast<microseconds>(end - start).count();

  if (status != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UHM] Send failed.";
  }
}

void send_channel_slice_g2h2g(Context ctx) {
  // Memory mem_cpu(0, MemoryType::CPU);
  // void* host_buffer;
  // mem_cpu.allocateBuffer(&host_buffer, ctx.size);

  // auto start = high_resolution_clock::now();
  // mem_op->copyDeviceToHost(host_buffer, ctx.data_ptr, ctx.size);
  // auto status = ctx.comm->sendDataTo(0, host_buffer, ctx.size, MemoryType::CPU);
  // auto end = high_resolution_clock::now();

  // total_time = duration_cast<microseconds>(end - start).count();

  // if (status != status_t::SUCCESS) {
  //   std::lock_guard<std::mutex> lock(*ctx.log_mutex);
  //   LOG(ERROR) << "[G2H2G] Send failed.";
  // }

  // mem_cpu.freeBuffer(host_buffer);
}

void send_channel_slice_rdma_cpu(Context ctx) {
  // const size_t chunk_size = buffer_size / 2;
  // size_t remaining = ctx.size;
  // size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;

  // std::vector<uint8_t> host_data(ctx.size, 'A');

  // auto start = high_resolution_clock::now();
  // for (size_t i = 0; i < num_chunks; ++i) {
  //   size_t send_size = min(chunk_size, remaining);
  //   buffer->writeFromCpu(host_data.data(), send_size, 0);
  //   ctx.comm->writeTo(0, 0, send_size, ConnType::RDMA);
  //   remaining -= send_size;
  // }
  // auto end = high_resolution_clock::now();
  // total_time = duration_cast<microseconds>(end - start).count();
}

std::string get_mode_from_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" || mode == "rdma_cpu") return mode;
      cerr << "Invalid mode: " << mode << "\n";
      exit(1);
    }
  }
  return "uhm";
}

int main(int argc, char* argv[]) {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  // ./client --mode=serial/uhm/g2h2g/rdma_cpu
  string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  cpu_buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);
  gpu_comm = new Communicator(gpu_buffer);
  cpu_comm = new Communicator(cpu_buffer);
  gpu_comm->connectTo(server_ip, gpu_port, ConnType::RDMA);
  cpu_comm->connectTo(server_ip, cpu_port, ConnType::RDMA);

  void (*send_func)(Context) = nullptr;
  if (mode == "serial") send_func = send_channel_slice_serial;
  else if (mode == "g2h2g") send_func = send_channel_slice_g2h2g;
  else if (mode == "rdma_cpu") send_func = send_channel_slice_rdma_cpu;
  else send_func = send_channel_slice_uhm;

  ofstream csv_file("performanceTest_client.csv", ios::app);
  if (csv_file.tellp() == 0) {
    csv_file << "Mode,Data Size (MB),Time(us),Bandwidth(Gbps)\n";
  }
  csv_file.close();

  for (int power = 3; power <= 26; ++power) {
    size_t total_size = pow(2, power);
    std::vector<uint8_t> host_data(total_size, 'A');

    void* device_ptr;
    gpu_mem_op->allocateBuffer(&device_ptr, total_size);
    gpu_mem_op->copyHostToDevice(device_ptr, host_data.data(), total_size);

    Context ctx = {.cpu_data_ptr=host_data.data(), .gpu_data_ptr = device_ptr, .size = total_size, .log_mutex = &log_mutex};
    
    send_func(ctx);

    double throughput_MBps = (total_size / 1024.0 / 1024.0) / (total_time / 1e6);
    double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    LOG(INFO) << "[Data Size " << (total_size / (1024 * 1024)) << " MB] "
              << total_time << " us, "
              << throughput_MBps << " MB/s, "
              << throughput_Gbps << " Gbps";

    ofstream file("performanceTest_client.csv", ios::app);
    if (file.is_open()) {
      file << mode << "," << (total_size / (1024 * 1024)) << "," << total_time << "," << throughput_Gbps << "\n";
      file.close();
    }

    gpu_mem_op->freeBuffer(device_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  gpu_comm->disConnect(server_ip, ConnType::RDMA);
  cpu_comm->disConnect(server_ip, ConnType::RDMA);
  delete gpu_comm;
  delete cpu_comm;

  std::cout << "Client finished all transfers\n";
  return 0;
}
