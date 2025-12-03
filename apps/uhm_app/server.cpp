#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace hmc;
using namespace std;

const std::string DEFAULT_SERVER_IP = "192.168.2.248";
const std::string DEFAULT_CLIENT_IP = "192.168.2.248";
const std::string DEFAULT_TCP_IP = "192.168.2.248";

std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;

const size_t buffer_size = 1024ULL * 1024 * 128; // 128 MB
const int device_id = 0;
const int gpu_port = 2025;
const int cpu_port = 2026;
const int ctrl_port = 2027;

int ctrl_socket_fd = -1;

std::shared_ptr<ConnBuffer> gpu_buffer;
std::shared_ptr<ConnBuffer> cpu_buffer;
Communicator *gpu_comm;
Communicator *cpu_comm;

Memory *gpu_mem_op = new Memory(device_id);
Memory *cpu_mem_op = new Memory(0, MemoryType::CPU);

struct Context {
  void *cpu_data_ptr;
  void *gpu_data_ptr;
  size_t size;
  std::mutex *log_mutex;
};

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

// 环境变量读取
std::string get_env_or_default(const char *var_name,
                               const std::string &default_val) {
  const char *val = getenv(var_name);
  return (val != nullptr) ? std::string(val) : default_val;
}

// ---------- TCP 控制通道 ----------
int setup_tcp_control_socket(int port, const std::string &bind_ip) {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == -1) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

  sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (inet_pton(AF_INET, bind_ip.c_str(), &address.sin_addr) <= 0) {
    perror("Invalid bind IP address");
    exit(EXIT_FAILURE);
  }

  if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  if (listen(server_fd, 1) < 0) {
    perror("listen failed");
    exit(EXIT_FAILURE);
  }

  LOG(INFO) << "Waiting for TCP control connection on " << bind_ip << ":" << port;
  int addrlen = sizeof(address);
  int new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
  if (new_socket < 0) {
    perror("accept failed");
    exit(EXIT_FAILURE);
  }

  close(server_fd);
  LOG(INFO) << "TCP control connection established.";
  return new_socket;
}

void close_control_socket(int &sock) {
  if (sock >= 0) {
    close(sock);
    sock = -1;
    LOG(INFO) << "Control socket closed.";
  }
}

bool wait_for_control_message(int socket_fd) {
  char buffer[16] = {0};
  int valread = read(socket_fd, buffer, sizeof(buffer));
  if (valread <= 0) return false;
  return std::string(buffer).find("Finished") != std::string::npos;
}

// ---------- UCX recv ----------
void recv_channel_slice_ucx(Context ctx) {
  if (!wait_for_control_message(ctrl_socket_fd)) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UCX] control message timeout.";
    return;
  }

  if (cpu_buffer->readToCpu(ctx.cpu_data_ptr, ctx.size, 0) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UCX] readToCpu failed.";
    return;
  }

  cuda_barrier(device_id);

  // H2D：这一步若是 async/或曾经非法，会在 barrier 处暴露，不会拖到 free
  gpu_mem_op->copyHostToDevice(ctx.gpu_data_ptr, ctx.cpu_data_ptr, ctx.size);

  cuda_barrier(device_id);

  LOG(INFO) << "[UCX] recv head=" << std::hex
            << (int)((unsigned char*)ctx.cpu_data_ptr)[0] << " "
            << (int)((unsigned char*)ctx.cpu_data_ptr)[1] << " "
            << (int)((unsigned char*)ctx.cpu_data_ptr)[2] << " "
            << (int)((unsigned char*)ctx.cpu_data_ptr)[3]
            << std::dec;
}

std::string get_mode_from_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" ||
          mode == "rdma_cpu" || mode == "ucx")
        return mode;
      cerr << "Invalid mode: " << mode << endl;
      exit(1);
    }
  }
  return "uhm";
}

int main(int argc, char *argv[]) {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  std::string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  server_ip = get_env_or_default("SERVER_IP", DEFAULT_SERVER_IP);
  client_ip = get_env_or_default("CLIENT_IP", DEFAULT_CLIENT_IP);
  tcp_server_ip = get_env_or_default("TCP_SERVER_IP", DEFAULT_TCP_IP);

  std::mutex log_mutex;
  gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  cpu_buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);

  int num_channels = 1;
  gpu_comm = new Communicator(gpu_buffer, num_channels);
  cpu_comm = new Communicator(cpu_buffer, num_channels);

  if (mode == "ucx") {
    if (cpu_comm->initServer(server_ip, gpu_port, ConnType::UCX) != status_t::SUCCESS) {
      LOG(ERROR) << "[Server] UCX initServer failed.";
      return -1;
    }
  } else {
    LOG(ERROR) << "This patched server file focuses on UCX mode. Use your original file for other modes.";
    return -1;
  }

  ctrl_socket_fd = setup_tcp_control_socket(ctrl_port, tcp_server_ip);

  for (int power = 5; power <= 26; ++power) {
    size_t total_size = size_t(1) << power;
    std::vector<uint8_t> host_data(total_size, 0);

    // 关键：每轮开始都把 device 锁死 + 清 error
    cuda_barrier(device_id);

    void *gpu_ptr = nullptr;
    CUDA_OK(cudaSetDevice(device_id));
    gpu_mem_op->allocateBuffer(&gpu_ptr, total_size);
    cuda_barrier(device_id);

    Context ctx = {host_data.data(), gpu_ptr, total_size, &log_mutex};
    recv_channel_slice_ucx(ctx);

    // D2H 校验（这一步也可能是 async，所以必须同步）
    gpu_mem_op->copyDeviceToHost(host_data.data(), gpu_ptr, total_size);
    cuda_barrier(device_id);

    bool valid = true;
    for (size_t i = 0; i < std::min<size_t>(10, total_size); ++i) {
      if (host_data[i] != 'A') { valid = false; break; }
    }

    LOG(INFO) << "[Size " << total_size << " B] Data Integrity: " << (valid ? "PASS" : "FAIL");

    cuda_barrier(device_id);
    CUDA_OK(cudaSetDevice(device_id));
    gpu_mem_op->freeBuffer(gpu_ptr);
    cuda_barrier(device_id);

    std::this_thread::sleep_for(std::chrono::seconds(1));
    LOG(INFO) << "--------------------------------------------";
  }

  gpu_comm->closeServer();
  cpu_comm->closeServer();
  close_control_socket(ctrl_socket_fd);
  delete gpu_comm;
  delete cpu_comm;

  std::cout << "Server shutdown complete." << std::endl;
  return 0;
}
