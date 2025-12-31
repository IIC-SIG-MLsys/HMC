#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

const std::string DEFAULT_SERVER_IP = "192.168.2.244";
const std::string DEFAULT_CLIENT_IP = "192.168.2.244";
const std::string DEFAULT_TCP_IP = "192.168.2.244";

std::string server_ip;
std::string client_ip;
std::string tcp_server_ip;

size_t buffer_size = 1024ULL * 1024 * 128; // max 32 for MLU
const int device_id = 0;
const int g_port = 2025;
const int ctrl_port = 2027;

int ctrl_sock = -1;

Communicator *comm = nullptr;
std::shared_ptr<ConnBuffer> buffer;
std::shared_ptr<ConnBuffer> gpu_buffer;

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

static Communicator::CtrlId self_rank = 1;
static Communicator::CtrlId peer_rank = 0;

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

bool connect_control_server(const std::string &server_ip, int ctrl_port = 9099) {
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

void send_channel_slice_uhm(Context ctx) {
  total_time = 0;
  auto start = steady_clock_t::now();

  auto status =
      comm->sendDataTo(server_ip, static_cast<uint16_t>(g_port),
                       ctx.gpu_data_ptr, ctx.size, MemoryType::DEFAULT);

  auto end = steady_clock_t::now();
  total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  if (status != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[UHM] Send failed." << std::endl;
  }
  send_control_message("Finished");
}

void send_channel_slice_serial(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t send_size = std::min(chunk_size, remaining);
    auto start = steady_clock_t::now();

    buffer->writeFromGpu(
        static_cast<char *>(ctx.gpu_data_ptr) + (ctx.size - remaining),
        send_size, 0);

    comm->put(server_ip, static_cast<uint16_t>(g_port),
              /*local_off=*/0, /*remote_off=*/0,
              /*size=*/send_size, ConnType::RDMA);

    auto end = steady_clock_t::now();
    remaining -= send_size;
    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    comm->ctrlSend(peer_rank, 1);
  }
  send_control_message("Finished");
}

void send_channel_slice_g2h2g(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  size_t sent_offset = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t send_size = std::min(chunk_size, remaining);
    auto start = steady_clock_t::now();

    gpu_mem_op->copyDeviceToHost(
        buffer->ptr, static_cast<char *>(ctx.gpu_data_ptr) + sent_offset,
        send_size);

    comm->put(server_ip, static_cast<uint16_t>(g_port),
              /*local_off=*/0, /*remote_off=*/0,
              /*size=*/send_size, ConnType::RDMA);

    auto end = steady_clock_t::now();
    sent_offset += send_size;
    remaining -= send_size;
    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();

    auto r = comm->ctrlSend(peer_rank, 1);
    if (r != status_t::SUCCESS)
      std::cout << "send error " << server_ip << std::endl;
  }

  send_control_message("Finished");
}

void send_channel_slice_rdma_cpu(Context ctx) {
  const size_t chunk_size = buffer_size / 2;
  size_t remaining = ctx.size;
  size_t num_chunks = (ctx.size + chunk_size - 1) / chunk_size;
  total_time = 0;

  size_t sent_offset = 0;

  for (size_t i = 0; i < num_chunks; ++i) {
    size_t send_size = std::min(chunk_size, remaining);

    buffer->writeFromGpu(static_cast<char *>(ctx.gpu_data_ptr) + sent_offset,
                         send_size, 0);

    auto start = steady_clock_t::now();

    comm->put(server_ip, static_cast<uint16_t>(g_port),
              /*local_off=*/0, /*remote_off=*/0,
              /*size=*/send_size, ConnType::RDMA);

    auto end = steady_clock_t::now();

    sent_offset += send_size;
    remaining -= send_size;
    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }

  send_control_message("Finished");
}

void send_channel_slice_ucx(Context ctx) {
  total_time = 0;

  auto start = steady_clock_t::now();

  // just for time calcu
  if (gpu_buffer->writeFromCpu(ctx.cpu_data_ptr, ctx.size, 0) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[UCX] writeFromCpu failed." << std::endl;
    return;
  }

  if (buffer->writeFromCpu(ctx.cpu_data_ptr, ctx.size, 0) != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[UCX] writeFromCpu failed." << std::endl;
    return;
  }

  auto status = comm->put(server_ip, static_cast<uint16_t>(g_port),
                          /*local_off=*/0, /*remote_off=*/0,
                          /*size=*/ctx.size, ConnType::UCX);

  auto end = steady_clock_t::now();

  total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  if (status != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[UCX] put failed." << std::endl;
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
  string mode = get_mode_from_args(argc, argv);
  std::cout << "Running in mode: " << mode << std::endl;

  server_ip = get_env_or_default("SERVER_IP", DEFAULT_SERVER_IP);
  client_ip = get_env_or_default("CLIENT_IP", DEFAULT_CLIENT_IP);
  tcp_server_ip = get_env_or_default("TCP_SERVER_IP", DEFAULT_TCP_IP);

  self_rank = get_env_u32_or_default("SELF_RANK", 1);
  peer_rank = get_env_u32_or_default("PEER_RANK", 0);
  std::cout << "Ranks: self=" << self_rank << " peer=" << peer_rank << std::endl;

  const bool use_cpu_buffer = (mode == "g2h2g" || mode == "ucx");

  if (use_cpu_buffer) {
    buffer = std::make_shared<ConnBuffer>(0, buffer_size, MemoryType::CPU);
    gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size,
                                              MemoryType::DEFAULT);
  } else {
    buffer = std::make_shared<ConnBuffer>(device_id, buffer_size,
                                          MemoryType::DEFAULT);
  }

  int num_channels = 1;
  comm = new Communicator(buffer, num_channels);

  ConnType conn_type = (mode == "ucx") ? ConnType::UCX : ConnType::RDMA;
  int port = g_port;

  // ---- ctrl link: same-host => UDS, cross-host => TCP ----
  Communicator::CtrlLink ctrl_link;
  const bool same_host = (server_ip == client_ip);

  if (same_host) {
    ctrl_link.transport = Communicator::CtrlTransport::UDS;
    std::string uds_dir = get_env_or_default("CTRL_UDS_DIR", "/tmp");
    ctrl_link.uds_path =
        Communicator::udsPathFor(uds_dir, peer_rank); // server rank path
    std::cout << "Ctrl transport=UDS (same_host) path=" << ctrl_link.uds_path
              << std::endl;
  } else {
    ctrl_link.transport = Communicator::CtrlTransport::TCP;
    ctrl_link.ip = server_ip;
    ctrl_link.port = static_cast<uint16_t>(ctrl_port + 1);
    std::cout << "Ctrl transport=TCP " << ctrl_link.ip << ":" << ctrl_link.port
              << std::endl;
  }

  {
    auto r = comm->connectTo(peer_rank, self_rank, server_ip,
                             static_cast<uint16_t>(port), ctrl_link, conn_type);
    if (r != status_t::SUCCESS) {
      std::cerr << "HMC connectTo failed" << std::endl;
      return -1;
    }
  }

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
  else if (mode == "ucx")
    send_func = send_channel_slice_ucx;
  else
    send_func = send_channel_slice_uhm;

  ofstream csv_file("performanceTest_client.csv", ios::app);
  if (csv_file.tellp() == 0)
    csv_file << "Mode,Data Size (MB),Time(us),Bandwidth(Gbps)\n";
  csv_file.close();

  sleep(3);

  for (int power = 5; power <= 26; ++power) {
    size_t total_size = (size_t)1 << power;
    std::vector<uint8_t> host_data(total_size, 'A');

    void *device_ptr = nullptr;
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

    std::cout << "[Mode: " << mode << "] "
              << (mode == "uhm" || mode == "rdma_cpu" || mode == "ucx"
                      ? "[Network only]"
                      : "[End-to-end]")
              << " Data Size: " << total_size << " B, "
              << "Time: " << total_time << " us, " << MBps << " MiB/s, " << gbps
              << " Gbps" << std::endl;

    ofstream file("performanceTest_client.csv", ios::app);
    if (file.is_open()) {
      double size_MB = total_size / 1024.0 / 1024.0;
      file << mode << "," << size_MB << "," << total_time << "," << gbps << "\n";
      file.close();
    }

    gpu_mem_op->freeBuffer(device_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  comm->disConnect(server_ip, static_cast<uint16_t>(port), conn_type);

  close_control_connection();
  delete comm;
  comm = nullptr;

  std::cout << "Client finished all transfers\n";
  return 0;
}
