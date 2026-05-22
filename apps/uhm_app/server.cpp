#include <arpa/inet.h>
#include <chrono>
#include <cmath>
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

const std::string DEFAULT_SERVER_IP = "192.168.2.243";
const std::string DEFAULT_CLIENT_IP = "192.168.2.243";
const std::string DEFAULT_TCP_IP = "192.168.2.243";

std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;

size_t buffer_size = 1024ULL * 1024 * 128; // max 32 for MLU
int device_id = 0;
const int g_port = 2025;
const int ctrl_port = 2027;

int ctrl_socket_fd = -1;

std::shared_ptr<ConnBuffer> buffer;
Communicator *comm = nullptr;

Memory *gpu_mem_op = nullptr;
Memory *cpu_mem_op = nullptr;

struct Context {
  void *cpu_data_ptr;
  void *gpu_data_ptr;
  size_t size;
  std::mutex *log_mutex;
};

static Communicator::CtrlId self_rank = 0;
static Communicator::CtrlId peer_rank = 1;

std::string get_env_or_default(const char *var_name,
                               const std::string &default_val) {
  const char *val = getenv(var_name);
  return (val != nullptr) ? std::string(val) : default_val;
}

static uint32_t get_env_u32_or_default(const char *var_name, uint32_t def) {
  const char *v = getenv(var_name);
  if (!v) return def;
  char *end = nullptr;
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
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
             sizeof(opt));

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

  std::cout << "Waiting for TCP control connection on " << bind_ip << ":" << port
            << std::endl;

  socklen_t addrlen = sizeof(address);
  int new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen);
  if (new_socket < 0) {
    perror("accept failed");
    exit(EXIT_FAILURE);
  }

  close(server_fd);
  std::cout << "TCP control connection established." << std::endl;
  return new_socket;
}

void close_control_socket(int &sock) {
  if (sock >= 0) {
    close(sock);
    sock = -1;
    std::cout << "Control socket closed." << std::endl;
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

  const bool same_host = (server_ip == client_ip);
  if (same_host) {
    // P2P mode: data already in ConnBuffer via client's putP2p
    wait_for_control_message(ctrl_socket_fd);
    return;
  }

  if (comm->recvDataFrom(client_ip, 0, ctx.gpu_data_ptr, ctx.size,
                         MemoryType::DEFAULT, &flags) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[UHM] Receive failed." << std::endl;
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
      std::cerr << "Serial mode: control message timeout, tag is " << tag
                << std::endl;
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
      std::cerr << "G2H2G mode: control message timeout" << std::endl;
      return;
    }
    if (tag != 1) {
      std::cerr << "G2H2G mode: control tag error is " << tag << std::endl;
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
    std::cerr << "[UCX] control message timeout." << std::endl;
    return;
  }

  if (buffer->readToCpu(ctx.cpu_data_ptr, ctx.size, 0) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[UCX] readToCpu failed." << std::endl;
    return;
  }

  gpu_mem_op->copyHostToDevice(ctx.gpu_data_ptr, ctx.cpu_data_ptr, ctx.size);
}

static size_t pipeline_chunk_size = 4 * 1024 * 1024;
static size_t pipeline_max_inflight = 64;

void recv_channel_slice_pipeline(Context ctx) {
  (void)ctx;
  wait_for_control_message(ctrl_socket_fd);
}

std::string get_mode_from_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" ||
          mode == "rdma_cpu" || mode == "ucx" || mode == "pipeline" ||
          mode == "write" || mode == "write_cpu" || mode == "write_stage")
        return mode;
      cerr << "Invalid mode: " << mode << endl;
      exit(1);
    }
  }
  return "uhm";
}

int main(int argc, char *argv[]) {
  std::string mode = get_mode_from_args(argc, argv);
  std::cout << "Running in mode: " << mode << std::endl;

  const std::string default_server_ip =
      get_env_or_default("DEFAULT_SERVER_IP", DEFAULT_SERVER_IP);
  const std::string default_client_ip =
      get_env_or_default("DEFAULT_CLIENT_IP", DEFAULT_CLIENT_IP);
  const std::string default_tcp_ip =
      get_env_or_default("DEFAULT_TCP_IP", DEFAULT_TCP_IP);

  server_ip = get_env_or_default("SERVER_IP", default_server_ip);
  client_ip = get_env_or_default("CLIENT_IP", default_client_ip);
  tcp_server_ip = get_env_or_default("TCP_SERVER_IP", default_tcp_ip);

  self_rank = get_env_u32_or_default("SELF_RANK", 0);
  peer_rank = get_env_u32_or_default("PEER_RANK", 1);
  std::cout << "Ranks: self=" << self_rank << " peer=" << peer_rank
            << std::endl;

  const bool same_host = (server_ip == client_ip);
  if (same_host) device_id = static_cast<int>(self_rank);

  if (!gpu_mem_op) gpu_mem_op = new Memory(device_id);
  if (!cpu_mem_op) cpu_mem_op = new Memory(0, MemoryType::CPU);

  std::mutex log_mutex;

  const bool use_cpu_buffer =
      (mode == "g2h2g" || mode == "ucx" || mode == "write_cpu" ||
       mode == "write_stage");

  const uint32_t default_buf_mb = use_cpu_buffer ? 128 : 32;
  uint32_t buf_mb = get_env_u32_or_default("BUFFER_SIZE_MB", default_buf_mb);
  if (buf_mb == 0) buf_mb = default_buf_mb;
  buffer_size = static_cast<size_t>(buf_mb) * 1024ULL * 1024ULL;
  std::cout << "ConnBuffer size: " << (buffer_size / 1024 / 1024) << "MB"
            << std::endl;

  if (use_cpu_buffer) {
    buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);
  } else {
    buffer = std::make_shared<ConnBuffer>(device_id, buffer_size,
                                          MemoryType::DEFAULT);
  }

  int num_channels = get_env_u32_or_default("NUM_CHANNELS", 1);
  std::cout << "Using " << num_channels << " QPs" << std::endl;

  if (mode == "pipeline" || mode == "write" || mode == "write_cpu" ||
      mode == "write_stage") {
    pipeline_chunk_size = get_env_u32_or_default("PIPELINE_CHUNK", 4 * 1024 * 1024);
    pipeline_max_inflight = get_env_u32_or_default("PIPELINE_INFLIGHT", 64);
    std::cout << "Pipeline: chunk=" << (pipeline_chunk_size / 1024 / 1024) 
              << "MB, max_inflight=" << pipeline_max_inflight << std::endl;
  }

  comm = new Communicator(buffer, num_channels);

  ConnType   conn_type = (mode == "ucx") ? ConnType::UCX : ConnType::RDMA;
  int port = g_port;

  if (same_host) {
    std::string uds_dir = get_env_or_default("CTRL_UDS_DIR", "/tmp");
    std::string uds_path = Communicator::udsPathFor(uds_dir, self_rank);
    comm->initCtrlServer(server_ip, 0, uds_path);

    // Wait for client via TCP sync before connecting UDS
    ctrl_socket_fd = setup_tcp_control_socket(ctrl_port, tcp_server_ip);
    {
      char msg[8];
      ssize_t n = read(ctrl_socket_fd, msg, sizeof(msg)); (void)n;
    }

    Communicator::CtrlLink link;
    link.transport = Communicator::CtrlTransport::UDS;
    link.uds_path = Communicator::udsPathFor(uds_dir, peer_rank);
    comm->connectCtrl(peer_rank, self_rank, link);

    comm->connectP2p(peer_rank, self_rank, device_id,
                     buffer->mem_ops ? buffer->mem_ops->getMemoryType() : MemoryType::DEFAULT);
    std::cout << "P2P connection established (same-host IPC)" << std::endl;
  } else {
    // ---- Original RDMA init ----
    std::string ctrl_uds_path;
    uint16_t ctrl_tcp_port = static_cast<uint16_t>(ctrl_port + 1);

    std::string uds_dir = get_env_or_default("CTRL_UDS_DIR", "/tmp");
    ctrl_uds_path = Communicator::udsPathFor(uds_dir, self_rank);
    std::cout << "Ctrl transport=TCP listen " << server_ip << ":"
              << ctrl_tcp_port << std::endl;

    if (comm->initServer(server_ip, static_cast<uint16_t>(port), ctrl_tcp_port,
                         ctrl_uds_path, conn_type) != status_t::SUCCESS) {
      std::cerr << "Failed to init server." << std::endl;
      return -1;
    }

    ctrl_socket_fd = setup_tcp_control_socket(ctrl_port, tcp_server_ip);
  }

  void (*recv_func)(Context) = nullptr;
  if (mode == "serial")
    recv_func = recv_channel_slice_serial;
  else if (mode == "g2h2g")
    recv_func = recv_channel_slice_g2h2g;
  else if (mode == "rdma_cpu")
    recv_func = recv_channel_slice_rdma_cpu;
  else if (mode == "ucx")
    recv_func = recv_channel_slice_ucx;
  else if (mode == "pipeline" || mode == "write" || mode == "write_cpu" ||
           mode == "write_stage")
    recv_func = recv_channel_slice_pipeline;
  else
    recv_func = recv_channel_slice_uhm;

  int min_power = static_cast<int>(get_env_u32_or_default("MIN_POWER", 5));
  int max_power = static_cast<int>(get_env_u32_or_default(
      "MAX_POWER",
      (mode == "write" || mode == "pipeline" || mode == "write_cpu" ||
       mode == "write_stage") ? 29 : 26));
  if (min_power < 1) min_power = 1;
  if (max_power < min_power) max_power = min_power;

  for (int power = min_power; power <= max_power; ++power) {
    size_t total_size = size_t(1) << power;
    const bool verify_from_buffer =
        (mode == "g2h2g" || mode == "pipeline" || mode == "write" ||
         mode == "write_cpu" || mode == "write_stage") || same_host;
    const size_t buffer_window =
        (buffer && buffer->buffer_size > 0) ? buffer->buffer_size : buffer_size;
    size_t check_size = verify_from_buffer ? std::min(total_size, buffer_window)
                                           : total_size;
    std::vector<uint8_t> host_data(check_size, 0);

    void *gpu_ptr = nullptr;
    if (!verify_from_buffer) {
      if (gpu_mem_op->allocateBuffer(&gpu_ptr, total_size) != status_t::SUCCESS ||
          !gpu_ptr) {
        std::cerr << "allocateBuffer failed, size=" << total_size << std::endl;
        break;
      }
    }

    Context ctx = {host_data.data(), gpu_ptr, total_size, &log_mutex};
    recv_func(ctx);

    if (mode == "rdma_cpu") {
      if (gpu_ptr) gpu_mem_op->freeBuffer(gpu_ptr);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      std::cout << "--------------------------------------------"
                << std::endl;
      continue;
    }

    if (verify_from_buffer) {
      if (buffer->readToCpu(host_data.data(), check_size, 0) !=
          status_t::SUCCESS) {
        std::cerr << "[" << mode << "] readToCpu failed." << std::endl;
      }
    } else {
      gpu_mem_op->copyDeviceToHost(host_data.data(), gpu_ptr, check_size);
    }

    bool valid = true;
    for (size_t i = 0; i < std::min<size_t>(10, check_size); ++i) {
      if (host_data[i] != 'A') {
        valid = false;
        break;
      }
    }

    std::cout << "[Size " << total_size << " B] Data Integrity: "
              << (valid ? "PASS" : "FAIL") << std::endl;

    if (gpu_ptr) gpu_mem_op->freeBuffer(gpu_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "--------------------------------------------" << std::endl;
  }

  comm->closeServer();
  close_control_socket(ctrl_socket_fd);
  delete comm;
  comm = nullptr;

  std::cout << "Server shutdown complete." << std::endl;
  return 0;
}
