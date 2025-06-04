#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <mutex>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "../src/memories/mem_type.h"
#include "../src/resource_manager/gpu_interface.h"

using namespace hmc;
using namespace std;
using namespace std::chrono;
const std::string DEFAULT_SERVER_IP = "192.168.2.236";
const std::string DEFAULT_CLIENT_IP = "192.168.2.248";
const std::string DEFAULT_TCP_IP = "10.102.0.241";
std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;
size_t buffer_size = 2048ULL * 32;
const int device_id = 0;
const int gpu_port = 2025;
const int cpu_port = 2026;
const int ctrl_port = 2027;

int ctrl_sock = -1; // TCP 控制sock

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

// 使用函数封装环境变量读取逻辑
std::string get_env_or_default(const char* var_name, const std::string& default_val) {
  const char* val = getenv(var_name);
  return (val != nullptr) ? std::string(val) : default_val;
}

// ===== TCP 连接 =====
bool connect_control_server(const std::string& server_ip, int ctrl_port = 9099) {
  ctrl_sock = socket(AF_INET, SOCK_STREAM, 0);
  if (ctrl_sock < 0) {
      perror("Socket creation error");
      return false;
  }

  sockaddr_in serv_addr{};
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(ctrl_port);

  if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0) {
      perror("Invalid address / Address not supported");
      close(ctrl_sock);
      ctrl_sock = -1;
      return false;
  }

  if (connect(ctrl_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
      perror("Connection to control server failed");
      close(ctrl_sock);
      ctrl_sock = -1;
      return false;
  }

  std::cout << "Connected to control server." << std::endl;
  return true;
}

bool send_control_message(const std::string& msg) {
  if (ctrl_sock < 0) {
      std::cerr << "Control socket not connected." << std::endl;
      return false;
  }
  ssize_t sent = send(ctrl_sock, msg.c_str(), msg.size(), 0);
  if (sent < 0) {
      perror("Send failed");
      return false;
  }
  return true;
}

void close_control_connection() {
  if (ctrl_sock >= 0) {
      close(ctrl_sock);
      ctrl_sock = -1;
      std::cout << "Control connection closed." << std::endl;
  }
}

void send_channel_slice_uhm(Context ctx) {
  total_time = 0;
  auto start = high_resolution_clock::now();
  auto status = gpu_comm->sendDataTo(server_ip, ctx.gpu_data_ptr, ctx.size, MemoryType::DEFAULT);
  auto end = high_resolution_clock::now();
  total_time = duration_cast<microseconds>(end - start).count();

  if (status != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UHM] Send failed.";
  }
}

void send_channel_slice_serial(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    auto start1 = high_resolution_clock::now();
    size_t send_size = min(chunk_size, remaining);
    gpu_buffer->writeFromGpu(static_cast<char*>(ctx.gpu_data_ptr) + (ctx.size - remaining), send_size, 0); // 从gpu直接拷贝到gpu buffer
    auto end1 = high_resolution_clock::now();
    auto start2 = high_resolution_clock::now();
    // LOG(INFO) << ctx.size << " " << ctx.size-remaining << " " << send_size;
    gpu_comm->writeTo(server_ip, 0, send_size, ConnType::RDMA); // 从gpu buffer 发送到对方 gpu buffer
    remaining -= send_size;
    auto end2 = high_resolution_clock::now();

    if (!send_control_message("Finished")) {
      std::cerr << "Send control message failed" << std::endl;
    }

    total_time += (duration_cast<microseconds>(end1 - start1).count() + duration_cast<microseconds>(end2 - start2).count()); // 拷贝时延+传输时延
  }
}

// 分块的gpu->cpu->cpu->gpu
void send_channel_slice_g2h2g(Context ctx) {
  void* host_buffer;
  cpu_mem_op->allocateBuffer(&host_buffer, ctx.size);

  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    auto start1 = high_resolution_clock::now();
    size_t send_size = min(chunk_size, remaining);
    gpu_mem_op->copyDeviceToHost(cpu_buffer->ptr, static_cast<char*>(ctx.gpu_data_ptr) + (ctx.size - remaining), send_size); // 从gpu直接拷贝到cpu buffer
    auto end1 = high_resolution_clock::now();
    auto start2 = high_resolution_clock::now();
    // LOG(INFO) << ctx.size << " " << ctx.size-remaining << " " << send_size;
    cpu_comm->writeTo(server_ip, 0, send_size, ConnType::RDMA); // 从cpu buffer 发送到对方 cpu buffer
    remaining -= send_size;
    auto end2 = high_resolution_clock::now();

    if (!send_control_message("Finished")) {
      std::cerr << "Send control message failed" << std::endl;
    }

    total_time += (duration_cast<microseconds>(end1 - start1).count() + duration_cast<microseconds>(end2 - start2).count()); // 拷贝时延+传输时延
  }

  cpu_mem_op->freeBuffer(host_buffer);
}

// 主动 RDMA Write
void send_channel_slice_rdma_cpu(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;
  
  for (size_t i = 0; i < num_chunks; ++i) {
    size_t send_size = min(chunk_size, remaining);
    // gpu_buffer->writeFromGpu(ctx.gpu_data_ptr, send_size, ctx.size-remaining);

    auto start = high_resolution_clock::now();
    gpu_comm->writeTo(server_ip, 0, send_size, ConnType::RDMA);
    remaining -= send_size;
    auto end = high_resolution_clock::now();

    total_time += duration_cast<microseconds>(end - start).count();
  }

  if (!send_control_message("Finished")) {
    std::cerr << "Send control message failed" << std::endl;
  }
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

  // ./client --mode serial/uhm/g2h2g/rdma_cpu
  string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  server_ip = get_env_or_default("SERVER_IP", DEFAULT_SERVER_IP);
  client_ip = get_env_or_default("CLIENT_IP", DEFAULT_CLIENT_IP);
  tcp_server_ip = get_env_or_default("TCP_SERVER_IP", DEFAULT_TCP_IP);

  gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  cpu_buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);
  gpu_comm = new Communicator(gpu_buffer);
  cpu_comm = new Communicator(cpu_buffer);

  gpu_comm->connectTo(server_ip, gpu_port, ConnType::RDMA);
  cpu_comm->connectTo(server_ip, cpu_port, ConnType::RDMA);

  // wait server start
  std::this_thread::sleep_for(std::chrono::seconds(1));

  int retry_count = 0;
  while (!connect_control_server(tcp_server_ip, ctrl_port)) {
    if (retry_count > 5) {
      std::cerr << "Failed to connect control server :"<< tcp_server_ip << std::endl;
      return -1;
    }
    retry_count++;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

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

  for (int power = 2; power <= 26; ++power) { // 暂时有bug，发送端多发一次，即接受端的缓冲区大一些做测试，这里是2，服务端为3
    size_t total_size = pow(2, power);
    std::vector<uint8_t> host_data(total_size, 'A');

    void* device_ptr;
    gpu_mem_op->allocateBuffer(&device_ptr, total_size);
    gpu_mem_op->copyHostToDevice(device_ptr, host_data.data(), total_size);

    Context ctx = {.cpu_data_ptr = host_data.data(), .gpu_data_ptr = device_ptr, .size = total_size, .log_mutex = &log_mutex};
    
    send_func(ctx);

    double throughput_MBps = (total_size / 1024.0 / 1024.0) / (total_time / 1e6);
    double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    LOG(INFO) << "[Data Size " << (total_size / (1024 * 1024)) << " MB] "
              << total_time << " us, "
              << throughput_MBps << " MB/s, "
              << throughput_Gbps << " Gbps";

    ofstream file("performanceTest_client.csv", ios::app);
    if (file.is_open()) {
      file << mode << "," << power << "," << total_time << "," << throughput_Gbps << "\n";
      file.close();
    }

    gpu_mem_op->freeBuffer(device_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  gpu_comm->disConnect(server_ip, ConnType::RDMA);
  cpu_comm->disConnect(server_ip, ConnType::RDMA);
  close_control_connection();
  delete gpu_comm;
  delete cpu_comm;

  std::cout << "Client finished all transfers\n";
  return 0;
}
