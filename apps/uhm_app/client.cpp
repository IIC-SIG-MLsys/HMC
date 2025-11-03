#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../src/memories/mem_type.h"
#include "../src/resource_manager/gpu_interface.h"

using namespace hmc;
using namespace std;

const std::string DEFAULT_SERVER_IP = "192.168.2.236";
const std::string DEFAULT_CLIENT_IP = "192.168.2.248";
const std::string DEFAULT_TCP_IP = "10.102.0.241";

std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;

size_t buffer_size = 1024ULL * 1024 * 4; // 4 MB
const int device_id = 0;
const int gpu_port = 2025;
const int cpu_port = 2026;
const int ctrl_port = 2027;

int ctrl_sock = -1;

Communicator *gpu_comm;
Communicator *cpu_comm;
std::shared_ptr<ConnBuffer> gpu_buffer;
std::shared_ptr<ConnBuffer> cpu_buffer;

Memory *gpu_mem_op = new Memory(device_id);
Memory *cpu_mem_op = new Memory(0, MemoryType::CPU);

struct Context {
  void *cpu_data_ptr;
  void *gpu_data_ptr;
  size_t size;
  std::mutex *log_mutex;
};

long long total_time = 0;
std::mutex log_mutex;

using steady_clock_t = std::chrono::steady_clock;

// -------------------- 控制通道 --------------------
std::string get_env_or_default(const char *var_name,
                               const std::string &default_val) {
  const char *val = getenv(var_name);
  return (val != nullptr) ? std::string(val) : default_val;
}

bool connect_control_server(const std::string &server_ip,
                            int ctrl_port = 9099) {
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

  if (connect(ctrl_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
      0) {
    perror("Connection to control server failed");
    close(ctrl_sock);
    ctrl_sock = -1;
    return false;
  }

  std::cout << "Connected to control server." << std::endl;
  return true;
}

bool send_control_message(const std::string &msg) {
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

// -------------------- 发送模式 --------------------
void send_channel_slice_uhm(Context ctx) {
  total_time = 0;
  auto start = steady_clock_t::now();
  auto status = gpu_comm->sendDataTo(server_ip, ctx.gpu_data_ptr, ctx.size,
                                     MemoryType::DEFAULT);
  auto end = steady_clock_t::now();
  total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

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
    size_t send_size = std::min(chunk_size, remaining);
    auto start = steady_clock_t::now();
    gpu_buffer->writeFromGpu(static_cast<char *>(ctx.gpu_data_ptr) +
                                 (ctx.size - remaining),
                             send_size, 0);
    gpu_comm->writeTo(server_ip, 0, send_size, ConnType::RDMA);
    auto end = steady_clock_t::now();

    remaining -= send_size;
    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    send_control_message("Finished");
  }
}

void send_channel_slice_g2h2g(Context ctx) {
  void *host_buffer;
  cpu_mem_op->allocateBuffer(&host_buffer, ctx.size);

  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t send_size = std::min(chunk_size, remaining);
    auto start = steady_clock_t::now();
    gpu_mem_op->copyDeviceToHost(cpu_buffer->ptr,
                                 static_cast<char *>(ctx.gpu_data_ptr) +
                                     (ctx.size - remaining),
                                 send_size);
    cpu_comm->writeTo(server_ip, 0, send_size, ConnType::RDMA);
    auto end = steady_clock_t::now();

    remaining -= send_size;
    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    send_control_message("Finished");
  }

  cpu_mem_op->freeBuffer(host_buffer);
}

void send_channel_slice_rdma_cpu(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  size_t sent_offset = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t send_size = std::min(chunk_size, remaining);

    gpu_buffer->writeFromGpu(
        static_cast<char *>(ctx.gpu_data_ptr) + sent_offset, send_size, 0);

    auto start = steady_clock_t::now();
    gpu_comm->send(server_ip, 0, send_size, ConnType::RDMA);
    auto end = steady_clock_t::now();

    sent_offset += send_size;
    remaining -= send_size;
    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }
  send_control_message("Finished");
}

// -------------------- 主程序 --------------------
std::string get_mode_from_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" ||
          mode == "rdma_cpu")
        return mode;
      cerr << "Invalid mode: " << mode << "\n";
      exit(1);
    }
  }
  return "uhm";
}

int main(int argc, char *argv[]) {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  server_ip = get_env_or_default("SERVER_IP", DEFAULT_SERVER_IP);
  client_ip = get_env_or_default("CLIENT_IP", DEFAULT_CLIENT_IP);
  tcp_server_ip = get_env_or_default("TCP_SERVER_IP", DEFAULT_TCP_IP);

  gpu_buffer =
      std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  cpu_buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);

  int num_channels = 1;
  gpu_comm = new Communicator(gpu_buffer, num_channels);
  cpu_comm = new Communicator(cpu_buffer, num_channels);

  gpu_comm->connectTo(server_ip, gpu_port, ConnType::RDMA);
  cpu_comm->connectTo(server_ip, cpu_port, ConnType::RDMA);

  std::this_thread::sleep_for(std::chrono::seconds(1));

  int retry_count = 0;
  while (!connect_control_server(tcp_server_ip, ctrl_port)) {
    if (retry_count > 5) {
      std::cerr << "Failed to connect control server :" << tcp_server_ip
                << std::endl;
      return -1;
    }
    retry_count++;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  void (*send_func)(Context) = nullptr;
  if (mode == "serial")
    send_func = send_channel_slice_serial;
  else if (mode == "g2h2g")
    send_func = send_channel_slice_g2h2g;
  else if (mode == "rdma_cpu")
    send_func = send_channel_slice_rdma_cpu;
  else
    send_func = send_channel_slice_uhm;

  ofstream csv_file("performanceTest_client.csv", ios::app);
  if (csv_file.tellp() == 0)
    csv_file << "Mode,Data Size (MB),Time(us),Bandwidth(Gbps)\n";
  csv_file.close();

  sleep(3);

  for (int power = 10; power <= 30; ++power) {
    size_t total_size = (size_t)1 << power;
    std::vector<uint8_t> host_data(total_size, 'A');

    void *device_ptr;
    gpu_mem_op->allocateBuffer(&device_ptr, total_size);
    gpu_mem_op->copyHostToDevice(device_ptr, host_data.data(), total_size);

    Context ctx = {.cpu_data_ptr = host_data.data(),
                   .gpu_data_ptr = device_ptr,
                   .size = total_size,
                   .log_mutex = &log_mutex};

    send_func(ctx);

    double time_s = total_time / 1e6;
    double gbps = (total_size * 8.0) / time_s / 1e9;
    double MBps = (total_size / time_s) / (1024.0 * 1024.0);

    LOG(INFO) << "[Mode: " << mode << "] "
              << (mode == "uhm" || mode == "rdma_cpu" ? "[Network only]"
                                                      : "[End-to-end]")
              << " Data Size: " << total_size << " B, "
              << "Time: " << total_time << " us, " << MBps << " MiB/s, " << gbps
              << " Gbps";

    ofstream file("performanceTest_client.csv", ios::app);
    if (file.is_open()) {
      double size_MB = total_size / 1024.0 / 1024.0;
      file << mode << "," << size_MB << "," << total_time << "," << gbps
           << "\n";
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
