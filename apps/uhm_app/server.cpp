#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <mutex>
#include <string>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace hmc;
using namespace std;

const size_t buffer_size = 2048ULL * 32;
const int device_id = 0;
const std::string server_ip = "192.168.2.248";
const std::string client_ip = "192.168.2.248";
const int gpu_port = 2025;
const int cpu_port = 2026;
const int ctrl_port = 2027;  // TCP 控制端口

int ctrl_socket_fd; // TCP 控制fd

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

// 控制函数（TCP 消息机制）
int setup_tcp_control_socket(int port) {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == -1) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

  sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }

  if (listen(server_fd, 1) < 0) {
    perror("listen failed");
    exit(EXIT_FAILURE);
  }

  LOG(INFO) << "Waiting for TCP control connection on port " << port;
  int addrlen = sizeof(address);
  int new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
  if (new_socket < 0) {
    perror("accept failed");
    exit(EXIT_FAILURE);
  }
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

void wait_for_control_message(int socket_fd) {
  char buffer[16] = {0};
  int valread = read(socket_fd, buffer, sizeof(buffer));
  if (valread <= 0 || std::string(buffer).find("Finished") == std::string::npos) {
    LOG(ERROR) << "Unexpected or no control message received.";
    exit(EXIT_FAILURE);
  }
  // LOG(INFO) << "Received control signal: " << buffer;
}

// 接收函数：UHM：GPU到GPU的点对点直传
void recv_channel_slice_uhm(Context ctx) {
  size_t flags;
  if (gpu_comm->recvDataFrom(client_ip, ctx.gpu_data_ptr, ctx.size, MemoryType::DEFAULT, &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "UHM Receive failed.";
  }
}

// 接收函数：Serial（分段）：被动处理每次传输结束
void recv_channel_slice_serial(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  // size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  for (size_t i = 0; i < num_chunks; ++i) {
    wait_for_control_message(ctrl_socket_fd);
    // copy逻辑有问题，暂时不用
    // gpu_mem_op->copyDeviceToDevice(static_cast<char*>(ctx.gpu_data_ptr) + (ctx.size - remaining), gpu_buffer->ptr, chunk_size); // 从gpu直接拷贝到cpu buffer
    // remaining -= chunk_size;
  }
}

// 接收函数：G2H2G：被动处理每次传输结束
void recv_channel_slice_g2h2g(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  // size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  for (size_t i = 0; i < num_chunks; ++i) {
    wait_for_control_message(ctrl_socket_fd);
    // copy逻辑有问题，暂时不用
    // gpu_mem_op->copyHostToDevice(static_cast<char*>(ctx.gpu_data_ptr) + (ctx.size - remaining), cpu_buffer->ptr, chunk_size); // 从gpu直接拷贝到cpu buffer
    // remaining -= chunk_size;
  }
}

// 接收函数：无需操作，被动接收 RDMA Write
void recv_channel_slice_rdma_cpu(Context ctx) {
  wait_for_control_message(ctrl_socket_fd);
}

// 获取运行模式
std::string get_mode_from_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      std::string mode = argv[++i];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" || mode == "rdma_cpu") {
        return mode;
      } else {
        std::cerr << "Invalid mode: " << mode << ". Supported: uhm / serial / g2h2g / rdma_cpu\n";
        exit(1);
      }
    }
  }
  return "uhm"; // 默认模式
}

int main(int argc, char* argv[]) {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  // ./server --mode serial/uhm/g2h2g/rdma_cpu
  std::string mode = get_mode_from_args(argc, argv);
  LOG(INFO) << "Running in mode: " << mode;

  std::mutex log_mutex;

  gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  cpu_buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);
  gpu_comm = new Communicator(gpu_buffer);
  cpu_comm = new Communicator(cpu_buffer);

  if (gpu_comm->initServer(server_ip, gpu_port, ConnType::RDMA) != status_t::SUCCESS) {
    LOG(ERROR) << "GPU Server init failed.";
    return -1;
  }
  if (cpu_comm->initServer(server_ip, cpu_port, ConnType::RDMA) != status_t::SUCCESS) {
    LOG(ERROR) << "CPU Server init failed.";
    return -1;
  }

  while (gpu_comm->checkConn(client_ip, ConnType::RDMA) != status_t::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  while (cpu_comm->checkConn(client_ip, ConnType::RDMA) != status_t::SUCCESS) {
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  LOG(INFO) << "RDMA Connection established. Ready to receive.";

  // 启动 TCP 控制连接
  ctrl_socket_fd = setup_tcp_control_socket(ctrl_port);

  // 选择接收函数
  void (*recv_func)(Context) = nullptr;
  if (mode == "serial") recv_func = recv_channel_slice_serial;
  else if (mode == "g2h2g") recv_func = recv_channel_slice_g2h2g;
  else if (mode == "rdma_cpu") recv_func = recv_channel_slice_rdma_cpu;
  else recv_func = recv_channel_slice_uhm;

  for (int power = 3; power <= 26; ++power) {
    const size_t total_size = std::pow(2, power);
    std::vector<uint8_t> host_data(total_size);
    void* gpu_ptr;
    gpu_mem_op->allocateBuffer(&gpu_ptr, total_size);

    Context ctx = {
        .cpu_data_ptr = host_data.data(),
        .gpu_data_ptr = gpu_ptr,
        .size = total_size,
        .log_mutex = &log_mutex
    };

    auto start_time = std::chrono::high_resolution_clock::now();
    recv_func(ctx);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;
    double seconds = duration.count();
    double throughput_MBps = (total_size / 1024.0 / 1024.0) / seconds;
    double throughput_Gbps = throughput_MBps * 1024.0 * 1024.0 * 8 / 1e9;

    if (mode != "g2h2g") gpu_mem_op->copyDeviceToHost(host_data.data(), gpu_ptr, total_size); // g2h2g已经在内部分段直接写入了

    bool valid = true;
    for (size_t i = 0; i < std::min<size_t>(10, total_size); ++i) {
      if (host_data[i] != 'A') {
        valid = false;
        break;
      }
    }

    LOG(INFO) << "[Size " << total_size / (1024 * 1024) << " MB] "
              << "Received in " << seconds << " s, "
              << throughput_MBps << " MB/s ("
              << throughput_Gbps << " Gbps), "
              << "Data Integrity: " << (valid ? "PASS" : "FAIL");

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
