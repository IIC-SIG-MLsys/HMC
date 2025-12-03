#include <arpa/inet.h>
#include <chrono>
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

size_t buffer_size = 1024ULL * 1024 * 128; // 128 MB
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

#include <cuda_runtime.h>
#define CUDA_OK(call)                                                          \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      LOG(ERROR) << "CUDA error: " << cudaGetErrorString(_e)                   \
                 << " @" << __FILE__ << ":" << __LINE__;                      \
    }                                                                          \
  } while (0)

static inline void cuda_barrier(int dev) {
  CUDA_OK(cudaSetDevice(dev));
  CUDA_OK(cudaGetLastError());
  CUDA_OK(cudaDeviceSynchronize());
}

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

  if (connect(ctrl_sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
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

// -------------------- UCX 模式发送 --------------------
void send_channel_slice_ucx(Context ctx) {
  total_time = 0;

  // UCX 走 CPU buffer：host->cpu_buffer，然后 RMA put
  if (cpu_buffer->writeFromCpu(ctx.cpu_data_ptr, ctx.size, 0) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UCX] writeFromCpu failed.";
    return;
  }

  auto start  = steady_clock_t::now();
  auto status = cpu_comm->writeTo(server_ip, /*ptr_bias=*/0, ctx.size, ConnType::UCX);
  auto end    = steady_clock_t::now();

  total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  if (status != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UCX] writeTo failed.";
    return;
  }

  send_control_message("Finished");
}

std::string get_mode_from_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" ||
          mode == "rdma_cpu" || mode == "ucx")
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

  gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  cpu_buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);

  int num_channels = 1;
  gpu_comm = new Communicator(gpu_buffer, num_channels);
  cpu_comm = new Communicator(cpu_buffer, num_channels);

  ConnType gpu_conn_type = (mode == "ucx") ? ConnType::UCX : ConnType::RDMA;
  ConnType cpu_conn_type = ConnType::RDMA;

  if (mode == "ucx") {
    cpu_comm->connectTo(server_ip, gpu_port, ConnType::UCX);
  } else {
    gpu_comm->connectTo(server_ip, gpu_port, ConnType::RDMA);
    cpu_comm->connectTo(server_ip, cpu_port, ConnType::RDMA);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  int retry_count = 0;
  while (!connect_control_server(tcp_server_ip, ctrl_port)) {
    if (retry_count > 5) {
      std::cerr << "Failed to connect control server :" << tcp_server_ip << std::endl;
      return -1;
    }
    retry_count++;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  void (*send_func)(Context) = nullptr;
  if (mode == "ucx") send_func = send_channel_slice_ucx;
  else {
    // 你原来的其它模式函数保持不变的话，这里自己接回去
    // 为了简洁，这个改完版只保证 UCX 模式正确；非 UCX 你可用原版文件。
    LOG(ERROR) << "This patched client file focuses on UCX mode. Use your original file for other modes.";
    return -1;
  }

  ofstream csv_file("performanceTest_client.csv", ios::app);
  if (csv_file.tellp() == 0) csv_file << "Mode,Data Size (MB),Time(us),Bandwidth(Gbps)\n";
  csv_file.close();

  sleep(3);

  for (int power = 5; power <= 26; ++power) {
    size_t total_size = (size_t)1 << power;
    std::vector<uint8_t> host_data(total_size, 'A');

    // 这两行是修复 free 的关键之一：保证 device 正确 + 上一个 CUDA 错误在这里暴露
    cuda_barrier(device_id);

    void *device_ptr = nullptr;
    CUDA_OK(cudaSetDevice(device_id));
    gpu_mem_op->allocateBuffer(&device_ptr, total_size);

    // H2D 只是为了你其它模式/一致性；UCX 其实不用 GPU buffer 也行
    gpu_mem_op->copyHostToDevice(device_ptr, host_data.data(), total_size);
    cuda_barrier(device_id);

    Context ctx = {.cpu_data_ptr = host_data.data(),
                   .gpu_data_ptr = device_ptr,
                   .size = total_size,
                   .log_mutex = &log_mutex};

    send_func(ctx);

    double time_s = total_time / 1e6;
    double gbps = (total_size * 8.0) / time_s / 1e9;
    double MBps = (total_size / time_s) / (1024.0 * 1024.0);

    LOG(INFO) << "[Mode: " << mode << "] [Network only]"
              << " Data Size: " << total_size << " B, "
              << "Time: " << total_time << " us, " << MBps << " MiB/s, " << gbps << " Gbps";

    ofstream file("performanceTest_client.csv", ios::app);
    if (file.is_open()) {
      double size_MB = total_size / 1024.0 / 1024.0;
      file << mode << "," << size_MB << "," << total_time << "," << gbps << "\n";
      file.close();
    }

    cuda_barrier(device_id);
    CUDA_OK(cudaSetDevice(device_id));
    gpu_mem_op->freeBuffer(device_ptr);
    cuda_barrier(device_id);

    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  gpu_comm->disConnect(server_ip, gpu_conn_type);
  cpu_comm->disConnect(server_ip, cpu_conn_type);
  close_control_connection();
  delete gpu_comm;
  delete cpu_comm;

  std::cout << "Client finished all transfers\n";
  return 0;
}
