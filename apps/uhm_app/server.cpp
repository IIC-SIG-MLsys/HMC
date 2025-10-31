#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <mutex>
#include <string>
#include <thread>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace hmc;
using namespace std;

const std::string DEFAULT_SERVER_IP = "192.168.2.248";
const std::string DEFAULT_CLIENT_IP = "192.168.2.248";
const std::string DEFAULT_TCP_IP    = "192.168.2.248";

std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;

const size_t buffer_size = 2048ULL * 32 * 1024;
const int device_id = 0;
const int gpu_port  = 2025;
const int cpu_port  = 2026;
const int ctrl_port = 2027;

int ctrl_socket_fd = -1;

std::shared_ptr<ConnBuffer> gpu_buffer;
std::shared_ptr<ConnBuffer> cpu_buffer;
Communicator* gpu_comm;
Communicator* cpu_comm;

Memory* gpu_mem_op = new Memory(device_id);
Memory* cpu_mem_op = new Memory(0, MemoryType::CPU);

struct Context {
  void* cpu_data_ptr;
  void* gpu_data_ptr;
  size_t size;
  std::mutex* log_mutex;
};

using steady_clock_t = std::chrono::steady_clock;

// 环境变量读取
std::string get_env_or_default(const char* var_name, const std::string& default_val) {
  const char* val = getenv(var_name);
  return (val != nullptr) ? std::string(val) : default_val;
}

// ---------- TCP 控制通道 ----------
int setup_tcp_control_socket(int port, const std::string& bind_ip) {
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

  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  if (listen(server_fd, 1) < 0) {
    perror("listen failed");
    exit(EXIT_FAILURE);
  }

  LOG(INFO) << "Waiting for TCP control connection on " << bind_ip << ":" << port;
  int addrlen = sizeof(address);
  int new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
  if (new_socket < 0) {
    perror("accept failed");
    exit(EXIT_FAILURE);
  }

  close(server_fd); // 监听完即可关闭
  LOG(INFO) << "TCP control connection established.";
  return new_socket;
}

void close_control_socket(int& sock) {
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

// ---------- 各模式接收 ----------

// ✅ uhm：GPU直连接收
void recv_channel_slice_uhm(Context ctx) {
  size_t flags = 0;
  if (gpu_comm->recvDataFrom(client_ip, ctx.gpu_data_ptr, ctx.size, MemoryType::DEFAULT, &flags)
      != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UHM] Receive failed.";
  }
}

// ✅ serial：分段接收
void recv_channel_slice_serial(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;

  for (size_t i = 0; i < num_chunks; ++i) {
    if (!wait_for_control_message(ctrl_socket_fd)) {
      LOG(ERROR) << "Serial mode: control message timeout.";
      return;
    }
  }
}

// ✅ g2h2g：分段接收
void recv_channel_slice_g2h2g(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;

  for (size_t i = 0; i < num_chunks; ++i) {
    if (!wait_for_control_message(ctrl_socket_fd)) {
      LOG(ERROR) << "G2H2G mode: control message timeout.";
      return;
    }
  }
}

// ✅ rdma_cpu：被动接收 RDMA write + control 信号
void recv_channel_slice_rdma_cpu(Context ctx) {
  gpu_comm->recv(client_ip, 0, ctx.size);
  wait_for_control_message(ctrl_socket_fd);
}

// ---------- 模式选择 ----------
std::string get_mode_from_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" || mode == "rdma_cpu")
        return mode;
      cerr << "Invalid mode: " << mode << endl;
      exit(1);
    }
  }
  return "uhm";
}

// ---------- 主程序 ----------
int main(int argc, char* argv[]) {
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
  gpu_comm = new Communicator(gpu_buffer);
  cpu_comm = new Communicator(cpu_buffer);

  if (gpu_comm->initServer(server_ip, gpu_port, ConnType::RDMA) != status_t::SUCCESS ||
      cpu_comm->initServer(server_ip, cpu_port, ConnType::RDMA) != status_t::SUCCESS) {
    LOG(ERROR) << "Failed to init RDMA servers.";
    return -1;
  }

  // 建立 TCP 控制连接
  ctrl_socket_fd = setup_tcp_control_socket(ctrl_port, tcp_server_ip);

  // 模式函数选择
  void (*recv_func)(Context) = nullptr;
  if (mode == "serial") recv_func = recv_channel_slice_serial;
  else if (mode == "g2h2g") recv_func = recv_channel_slice_g2h2g;
  else if (mode == "rdma_cpu") recv_func = recv_channel_slice_rdma_cpu;
  else recv_func = recv_channel_slice_uhm;

  // 主循环
  for (int power = 3; power <= 26; ++power) {
    size_t total_size = size_t(1) << power;
    std::vector<uint8_t> host_data(total_size, 0);
    void* gpu_ptr;
    gpu_mem_op->allocateBuffer(&gpu_ptr, total_size);

    Context ctx = { host_data.data(), gpu_ptr, total_size, &log_mutex };
    recv_func(ctx);

    // 数据完整性验证（可选）
    if (mode == "rdma_cpu")
      gpu_buffer->readToCpu(host_data.data(), total_size, 0);
    else if (mode != "g2h2g")
      gpu_mem_op->copyDeviceToHost(host_data.data(), gpu_ptr, total_size);

    bool valid = true;
    for (size_t i = 0; i < std::min<size_t>(10, total_size); ++i) {
      if (host_data[i] != 'A') {
        valid = false;
        break;
      }
    }

    LOG(INFO) << "[Size " << total_size << " B] "
              << "Transfer done. Data Integrity: "
              << (valid ? "PASS" : "FAIL");

    gpu_mem_op->freeBuffer(gpu_ptr);
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
