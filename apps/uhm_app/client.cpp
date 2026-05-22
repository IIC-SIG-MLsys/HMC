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

int ctrl_sock = -1;

Communicator *comm = nullptr;
std::shared_ptr<ConnBuffer> buffer;
std::shared_ptr<ConnBuffer> gpu_buffer;

Memory *gpu_mem_op = nullptr;
Memory *cpu_mem_op = nullptr;

struct Context {
  void *cpu_data_ptr;
  void *gpu_data_ptr;
  size_t size;
  std::mutex *log_mutex;
};

long long total_time = 0;
std::mutex log_mutex;
bool g_last_transfer_ok = true;

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

  const bool same_host = (server_ip == client_ip);
  if (same_host) {
    // P2P mode: write GPU data to ConnBuffer, then putP2p to server's ConnBuffer
    auto start = steady_clock_t::now();

    buffer->writeFromGpu(ctx.gpu_data_ptr, ctx.size, 0);
    comm->putP2p(peer_rank, 0, 0, ctx.size);

    auto end = steady_clock_t::now();
    total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    send_control_message("Finished");
    return;
  }

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

static size_t pipeline_chunk_size = 4 * 1024 * 1024;  // 4MB default
static size_t pipeline_max_inflight = 64;
static size_t stage_chunk_size = 8 * 1024 * 1024;
static size_t stage_slots = 0;
static size_t stage_src_window = 32 * 1024 * 1024;

void send_channel_slice_pipeline(Context ctx) {
  total_time = 0;
  g_last_transfer_ok = false;
  if ((!ctx.gpu_data_ptr && buffer->mem_type != MemoryType::CPU) || ctx.size == 0) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[Pipeline] invalid input ctx." << std::endl;
    send_control_message("Finished");
    return;
  }

  const size_t window = std::max<size_t>(1, buffer->buffer_size);
  const size_t first_step = std::min(ctx.size, window);
  status_t stage_ret = status_t::SUCCESS;
  if (buffer->mem_type == MemoryType::CPU) {
    stage_ret = buffer->writeFromCpu(ctx.cpu_data_ptr, first_step, 0);
  } else {
    stage_ret = buffer->writeFromGpu(ctx.gpu_data_ptr, first_step, 0);
  }
  if (stage_ret != status_t::SUCCESS) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[Pipeline] staging to ConnBuffer failed, step=" << first_step
              << std::endl;
    send_control_message("Finished");
    return;
  }

  auto start = steady_clock_t::now();
  size_t remaining = ctx.size;
  while (remaining > 0) {
    const size_t step = std::min(remaining, window);
    auto status = comm->putPipeline(
        server_ip, static_cast<uint16_t>(g_port),
        0, 0,
        step,
        pipeline_chunk_size,
        pipeline_max_inflight,
        ConnType::RDMA);
    if (status != status_t::SUCCESS) {
      std::lock_guard<std::mutex> lock(*ctx.log_mutex);
      std::cerr << "[Pipeline] putPipeline failed, step=" << step
                << ", remaining=" << remaining << std::endl;
      send_control_message("Finished");
      return;
    }
    remaining -= step;
  }
  auto end = steady_clock_t::now();

  total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  g_last_transfer_ok = true;
  send_control_message("Finished");
}

void send_channel_slice_write_stage(Context ctx) {
  total_time = 0;
  g_last_transfer_ok = false;
  if (!ctx.gpu_data_ptr || ctx.size == 0) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[WriteStage] invalid input ctx." << std::endl;
    send_control_message("Finished");
    return;
  }
  if (!buffer || !buffer->ptr || buffer->buffer_size < 2) {
    std::lock_guard<std::mutex> lock(*ctx.log_mutex);
    std::cerr << "[WriteStage] invalid ConnBuffer." << std::endl;
    send_control_message("Finished");
    return;
  }

  const size_t half = buffer->buffer_size / 2;
  size_t chunk = std::max<size_t>(1, std::min(stage_chunk_size, half));
  size_t slots_by_buf = std::max<size_t>(1, buffer->buffer_size / chunk);
  size_t slots = stage_slots == 0 ? slots_by_buf : std::min(stage_slots, slots_by_buf);
  slots = std::max<size_t>(1, slots);
  std::vector<uint64_t> inflight(slots, 0);
  const size_t src_window = std::max<size_t>(1, std::min(ctx.size, stage_src_window));
  auto wait_one = [&](uint64_t wr_id, size_t slot, const char *phase) -> bool {
    if (wr_id == 0) return true;
    std::vector<uint64_t> ids{wr_id};
    auto st = comm->wait(ids); // uses waitWrIdMulti with timeout
    if (st != status_t::SUCCESS) {
      std::lock_guard<std::mutex> lock(*ctx.log_mutex);
      std::cerr << "[WriteStage] wait failed (" << phase << "), slot=" << slot
                << ", wr_id=" << wr_id << ", status=" << (int)st << std::endl;
      return false;
    }
    return true;
  };

  auto start = steady_clock_t::now();
  size_t sent = 0;
  size_t idx = 0;
  while (sent < ctx.size) {
    const size_t slot = idx % slots;
    const size_t local_off = slot * chunk;
    const size_t remote_off = slot * chunk;
    const size_t step = std::min(chunk, ctx.size - sent);

    if (inflight[slot] != 0) {
      if (!wait_one(inflight[slot], slot, "reuse_slot")) {
        send_control_message("Finished");
        return;
      }
      inflight[slot] = 0;
    }

    void *gpu_src = static_cast<char *>(ctx.gpu_data_ptr) + (sent % src_window);
    if (gpu_mem_op->copyDeviceToHost(static_cast<char *>(buffer->ptr) + local_off,
                                     gpu_src, step) != status_t::SUCCESS) {
      std::lock_guard<std::mutex> lock(*ctx.log_mutex);
      std::cerr << "[WriteStage] copyDeviceToHost failed, step=" << step
                << std::endl;
      send_control_message("Finished");
      return;
    }

    uint64_t wr_id = 0;
    if (comm->putNB(server_ip, static_cast<uint16_t>(g_port),
                    local_off, remote_off, step, &wr_id,
                    ConnType::RDMA) != status_t::SUCCESS) {
      std::lock_guard<std::mutex> lock(*ctx.log_mutex);
      std::cerr << "[WriteStage] putNB failed, step=" << step << std::endl;
      send_control_message("Finished");
      return;
    }
    inflight[slot] = wr_id;
    sent += step;
    idx++;
  }

  for (size_t i = 0; i < inflight.size(); ++i) {
    if (inflight[i] != 0) {
      if (!wait_one(inflight[i], i, "final_drain")) {
        send_control_message("Finished");
        return;
      }
    }
  }

  auto end = steady_clock_t::now();
  total_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  g_last_transfer_ok = true;
  send_control_message("Finished");
}

std::string get_mode_from_args(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--mode" && i + 1 < argc) {
      string mode = argv[i + 1];
      if (mode == "uhm" || mode == "serial" || mode == "g2h2g" ||
          mode == "rdma_cpu" || mode == "ucx" || mode == "pipeline" ||
          mode == "write" || mode == "write_cpu" || mode == "write_stage")
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

  const std::string default_server_ip =
      get_env_or_default("DEFAULT_SERVER_IP", DEFAULT_SERVER_IP);
  const std::string default_client_ip =
      get_env_or_default("DEFAULT_CLIENT_IP", DEFAULT_CLIENT_IP);
  const std::string default_tcp_ip =
      get_env_or_default("DEFAULT_TCP_IP", DEFAULT_TCP_IP);

  server_ip = get_env_or_default("SERVER_IP", default_server_ip);
  client_ip = get_env_or_default("CLIENT_IP", default_client_ip);
  tcp_server_ip = get_env_or_default("TCP_SERVER_IP", default_tcp_ip);

  self_rank = get_env_u32_or_default("SELF_RANK", 1);
  peer_rank = get_env_u32_or_default("PEER_RANK", 0);
  std::cout << "Ranks: self=" << self_rank << " peer=" << peer_rank << std::endl;

  const bool same_host = (server_ip == client_ip);
  if (same_host) device_id = static_cast<int>(self_rank);

  if (!gpu_mem_op) gpu_mem_op = new Memory(device_id);
  if (!cpu_mem_op) cpu_mem_op = new Memory(0, MemoryType::CPU);

  int num_channels = get_env_u32_or_default("NUM_CHANNELS", 1);
  std::cout << "Using " << num_channels << " QPs" << std::endl;

  // Pipeline parameters
  if (mode == "pipeline" || mode == "write" || mode == "write_cpu" ||
      mode == "write_stage") {
    pipeline_chunk_size = get_env_u32_or_default("PIPELINE_CHUNK", 4 * 1024 * 1024);
    pipeline_max_inflight = get_env_u32_or_default("PIPELINE_INFLIGHT", 64);
    std::cout << "Pipeline: chunk=" << (pipeline_chunk_size / 1024 / 1024) 
              << "MB, max_inflight=" << pipeline_max_inflight << std::endl;
  }
  if (mode == "write_stage") {
    stage_chunk_size = get_env_u32_or_default("STAGE_CHUNK", 8 * 1024 * 1024);
    stage_slots = get_env_u32_or_default("STAGE_SLOTS", 0);
    uint32_t src_mb = get_env_u32_or_default("STAGE_SRC_WINDOW_MB", 32);
    if (src_mb == 0) src_mb = 32;
    stage_src_window = static_cast<size_t>(src_mb) * 1024ULL * 1024ULL;
    std::cout << "WriteStage: stage_chunk=" << (stage_chunk_size / 1024 / 1024)
              << "MB, stage_slots="
              << (stage_slots == 0 ? std::string("auto")
                                   : std::to_string(stage_slots))
              << ", stage_src_window=" << (stage_src_window / 1024 / 1024)
              << "MB"
              << std::endl;
  }

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
    if (mode == "ucx") {
      gpu_buffer = std::make_shared<ConnBuffer>(device_id, buffer_size,
                                                MemoryType::DEFAULT);
    } else {
      gpu_buffer.reset();
    }
  } else {
    buffer = std::make_shared<ConnBuffer>(device_id, buffer_size,
                                          MemoryType::DEFAULT);
    gpu_buffer.reset();
  }

  comm = new Communicator(buffer, num_channels);

  ConnType conn_type = (mode == "ucx") ? ConnType::UCX : ConnType::RDMA;
  int port = g_port;

  if (same_host) {
    std::string uds_dir = get_env_or_default("CTRL_UDS_DIR", "/tmp");
    std::string uds_path = Communicator::udsPathFor(uds_dir, self_rank);
    comm->initCtrlServer("127.0.0.1", 0, uds_path);

    // Signal server via TCP before UDS connect
    if (!connect_control_server(tcp_server_ip, ctrl_port)) {
      std::cerr << "Failed to connect control server" << std::endl;
      return -1;
    }
    send_control_message("START");

    Communicator::CtrlLink link;
    link.transport = Communicator::CtrlTransport::UDS;
    link.uds_path = Communicator::udsPathFor(uds_dir, peer_rank);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    comm->connectCtrl(peer_rank, self_rank, link);

    comm->connectP2p(peer_rank, self_rank, device_id,
                     buffer->mem_ops ? buffer->mem_ops->getMemoryType() : MemoryType::DEFAULT);
    std::cout << "P2P connection established (same-host IPC)" << std::endl;
  } else {
    // ---- Original RDMA/UCX connect ----
    Communicator::CtrlLink ctrl_link;
    ctrl_link.transport = Communicator::CtrlTransport::TCP;
    ctrl_link.ip = server_ip;
    ctrl_link.port = static_cast<uint16_t>(ctrl_port + 1);
    std::cout << "Ctrl transport=TCP " << ctrl_link.ip << ":" << ctrl_link.port
              << std::endl;

    auto r = comm->connectTo(peer_rank, self_rank, server_ip,
                             static_cast<uint16_t>(port), ctrl_link, conn_type);
    if (r != status_t::SUCCESS) {
      std::cerr << "HMC connectTo failed" << std::endl;
      return -1;
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
  else if (mode == "pipeline" || mode == "write" || mode == "write_cpu")
    send_func = send_channel_slice_pipeline;
  else if (mode == "write_stage")
    send_func = send_channel_slice_write_stage;
  else
    send_func = send_channel_slice_uhm;

  ofstream csv_file("performanceTest_client.csv", ios::app);
  if (csv_file.tellp() == 0)
    csv_file << "Mode,Data Size (MB),Time(us),Bandwidth(Gbps)\n";
  csv_file.close();

  sleep(3);

  int min_power = static_cast<int>(get_env_u32_or_default("MIN_POWER", 5));
  int max_power = static_cast<int>(get_env_u32_or_default(
      "MAX_POWER",
      (mode == "write" || mode == "pipeline" || mode == "write_cpu" ||
       mode == "write_stage") ? 29 : 26));
  if (min_power < 1) min_power = 1;
  if (max_power < min_power) max_power = min_power;

  for (int power = min_power; power <= max_power; ++power) {
    size_t total_size = (size_t)1 << power;
    size_t src_size = total_size;
    if (mode == "write_stage") {
      src_size = std::min(total_size, stage_src_window);
    } else if (mode == "write" || mode == "pipeline" || mode == "write_cpu") {
      src_size = std::min(total_size, buffer->buffer_size);
    }
    std::vector<uint8_t> host_data(src_size, 'A');

    void *device_ptr = nullptr;
    if (mode != "write_cpu") {
      if (gpu_mem_op->allocateBuffer(&device_ptr, src_size) != status_t::SUCCESS ||
          !device_ptr) {
        std::cerr << "allocateBuffer failed, size=" << src_size << std::endl;
        break;
      }
      if (gpu_mem_op->copyHostToDevice(device_ptr, host_data.data(), src_size) !=
          status_t::SUCCESS) {
        std::cerr << "copyHostToDevice failed, size=" << src_size << std::endl;
        gpu_mem_op->freeBuffer(device_ptr);
        break;
      }
    }

    Context ctx = {.cpu_data_ptr = host_data.data(),
                   .gpu_data_ptr = device_ptr,
                   .size = total_size,
                   .log_mutex = &log_mutex};

    g_last_transfer_ok = true;
    send_func(ctx);
    if (!g_last_transfer_ok || total_time <= 0) {
      std::cerr << "[Mode: " << mode << "] transfer failed at size="
                << total_size << " B" << std::endl;
      if (device_ptr) gpu_mem_op->freeBuffer(device_ptr);
      break;
    }

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

    if (device_ptr) gpu_mem_op->freeBuffer(device_ptr);
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  comm->disConnect(server_ip, static_cast<uint16_t>(port), conn_type);

  close_control_connection();
  delete comm;
  comm = nullptr;

  std::cout << "Client finished all transfers\n";
  return 0;
}
