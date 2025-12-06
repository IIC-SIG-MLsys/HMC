#include <arpa/inet.h>
#include <chrono>
#include <cmath>
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

#include "../src/memories/mem_type.h"
#include "../src/resource_manager/gpu_interface.h"

using namespace hmc;
using namespace std;

const std::string DEFAULT_SERVER_IP = "192.168.2.248";
const std::string DEFAULT_CLIENT_IP = "192.168.2.248";
const std::string DEFAULT_TCP_IP = "192.168.2.248";

std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;

const size_t buffer_size = 1024ULL * 1024 * 128; // max 32 for MLU
const int device_id = 0;
const int g_port = 2025;
const int ctrl_port = 2027;

int ctrl_socket_fd = -1;

std::shared_ptr<ConnBuffer> buffer;
Communicator *comm = nullptr;

Memory *gpu_mem_op = new Memory(device_id);
Memory *cpu_mem_op = new Memory(0, MemoryType::CPU);

struct Context {
  void *cpu_data_ptr;
  void *gpu_data_ptr;
  size_t size;
  std::mutex *log_mutex;
};

static Communicator::CtrlId self_rank = 0;
static Communicator::CtrlId peer_rank = 1;

std::string get_env_or_default(const char *var_name, const std::string &default_val) {
  const char *val = getenv(var_name);
  return (val != nullptr) ? std::string(val) : default_val;
}

static uint32_t get_env_u32_or_default(const char* var_name, uint32_t def) {
  const char* v = getenv(var_name);
  if (!v) return def;
  char* end = nullptr;
  unsigned long x = std::strtoul(v, &end, 10);
  if (end == v) return def;
  return static_cast<uint32_t>(x);
}

int setup_tcp_control_socket(int port, const std::string &bind_ip) {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd == -1) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));

  sockaddr_in address{};
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
  socklen_t addrlen = sizeof(address);
  int new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen);
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
  char buffer_local[16] = {0};
  int valread = read(socket_fd, buffer_local, sizeof(buffer_local));
  if (valread <= 0) return false;
  return std::string(buffer_local).find("Finished") != std::string::npos;
}

void recv_channel_slice_uhm(Context ctx) {
  size_t flags = 0;

  // NOTE: per your new Communicator, recvDataFrom is still recvDataFrom(ip, ...)
  // and is intended for RDMA-only cross-IP p2p legacy UHM path;
  // so we DO NOT add port here.

  // 如果client也init了server，那么此处用client的rdma port，否则是0
  if (comm->recvDataFrom(client_ip, 0, ctx.gpu_data_ptr, ctx.size, MemoryType::DEFAULT,
                         &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UHM] Receive failed.";
  }
  wait_for_control_message(ctrl_socket_fd);
}

void recv_channel_slice_serial(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;

  uint64_t tag = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    auto r = comm->ctrlRecv(peer_rank, &tag);
    if (r != status_t::SUCCESS || tag != 1) {
      LOG(ERROR) << "Serial mode: control message timeout, tag is " << tag;
      return;
    }
  }
  wait_for_control_message(ctrl_socket_fd);
}

void recv_channel_slice_g2h2g(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;

  uint64_t tag = 0;
  for (size_t i = 0; i < num_chunks; ++i) {
    auto r = comm->ctrlRecv(peer_rank, &tag);
    if (r != status_t::SUCCESS) {
      LOG(ERROR) << "G2H2G mode: control message timeout";
      return;
    }
    if (tag != 1) {
      LOG(ERROR) << "G2H2G mode: control tag error is " << tag;
      return;
    }
  }
  wait_for_control_message(ctrl_socket_fd);
}

void recv_channel_slice_rdma_cpu(Context ctx) {
  (void)ctx;
  wait_for_control_message(ctrl_socket_fd);
}

void recv_channel_slice_ucx(Context ctx) {
  if (!wait_for_control_message(ctrl_socket_fd)) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UCX] control message timeout.";
    return;
  }

  if (buffer->readToCpu(ctx.cpu_data_ptr, ctx.size, 0) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    LOG(ERROR) << "[UCX] readToCpu failed.";
    return;
  }

  gpu_mem_op->copyHostToDevice(ctx.gpu_data_ptr, ctx.cpu_data_ptr, ctx.size);
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

  self_rank = get_env_u32_or_default("SELF_RANK", 0);
  peer_rank = get_env_u32_or_default("PEER_RANK", 1);
  LOG(INFO) << "Ranks: self=" << self_rank << " peer=" << peer_rank;

  std::mutex log_mutex;

  const bool use_cpu_buffer = (mode == "g2h2g" || mode == "ucx");

  if (use_cpu_buffer) {
    buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);
  } else {
    buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::DEFAULT);
  }

  int num_channels = 1;
  comm = new Communicator(buffer, num_channels);

  ConnType conn_type = (mode == "ucx") ? ConnType::UCX : ConnType::RDMA;
  int port = g_port;

  // ---- initServer(bind_ip, data_port, ctrl_tcp_port, ctrl_uds_path, serverType) ----
  const bool same_host = (server_ip == client_ip);

  std::string ctrl_uds_path;
  uint16_t ctrl_tcp_port = static_cast<uint16_t>(ctrl_port + 1);

  if (same_host) {
    std::string uds_dir = get_env_or_default("CTRL_UDS_DIR", "/tmp");
    ctrl_uds_path = Communicator::udsPathFor(uds_dir, self_rank);
    LOG(INFO) << "Ctrl transport=UDS (same_host) listen_path=" << ctrl_uds_path;
  } else {
    ctrl_uds_path = "";
    LOG(INFO) << "Ctrl transport=TCP listen " << server_ip << ":" << ctrl_tcp_port;
  }

  if (comm->initServer(server_ip,
                       static_cast<uint16_t>(port),
                       ctrl_tcp_port,
                       ctrl_uds_path,
                       conn_type) != status_t::SUCCESS) {
    LOG(ERROR) << "Failed to init server.";
    return -1;
  }

  ctrl_socket_fd = setup_tcp_control_socket(ctrl_port, tcp_server_ip);

  void (*recv_func)(Context) = nullptr;
  if (mode == "serial") recv_func = recv_channel_slice_serial;
  else if (mode == "g2h2g") recv_func = recv_channel_slice_g2h2g;
  else if (mode == "rdma_cpu") recv_func = recv_channel_slice_rdma_cpu;
  else if (mode == "ucx") recv_func = recv_channel_slice_ucx;
  else recv_func = recv_channel_slice_uhm;

  for (int power = 5; power <= 26; ++power) {
    size_t total_size = size_t(1) << power;
    std::vector<uint8_t> host_data(total_size, 0);

    void *gpu_ptr = nullptr;
    gpu_mem_op->allocateBuffer(&gpu_ptr, total_size);

    Context ctx = {host_data.data(), gpu_ptr, total_size, &log_mutex};
    recv_func(ctx);

    if (mode == "rdma_cpu") {
      gpu_mem_op->freeBuffer(gpu_ptr);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      LOG(INFO) << "--------------------------------------------";
      continue;
    }

    if (mode == "g2h2g") {
      if (buffer->readToCpu(host_data.data(), total_size, 0) != status_t::SUCCESS) {
        LOG(ERROR) << "[G2H2G] readToCpu failed.";
      }
    } else {
      gpu_mem_op->copyDeviceToHost(host_data.data(), gpu_ptr, total_size);
    }

    bool valid = true;
    for (size_t i = 0; i < std::min<size_t>(10, total_size); ++i) {
      if (host_data[i] != 'A') {
        valid = false;
        break;
      }
    }

    LOG(INFO) << "[Size " << total_size << " B] Data Integrity: "
              << (valid ? "PASS" : "FAIL");

    gpu_mem_op->freeBuffer(gpu_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    LOG(INFO) << "--------------------------------------------";
  }

  comm->closeServer();
  close_control_socket(ctrl_socket_fd);
  delete comm;
  comm = nullptr;

  std::cout << "Server shutdown complete." << std::endl;
  return 0;
}
