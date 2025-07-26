#include <glog/logging.h>
#include <hmc.h>
#include <cstdlib>

using namespace hmc;

int main(int argc, char* argv[]) {
  // 简单的UCX模式选择
  ConnType conn_type = ConnType::RDMA;
  if (argc > 1 && std::string(argv[1]) == "ucx") {
    conn_type = ConnType::UCX;
    std::cout << "Using UCX mode" << std::endl;
    // UCX环境变量设置
    setenv("UCX_TLS", "tcp", 1);
    setenv("UCX_WARN_UNUSED_ENV_VARS", "n", 1);
    setenv("UCX_NET_DEVICES", "all", 1);  // 覆盖硬编码的网络设备
  } else {
    std::cout << "Using RDMA mode" << std::endl;
  }

  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  int device_id = 0;
  size_t buffer_size = 1 * 1024 * 1024;

  // UCX模式使用CPU内存，RDMA模式使用默认内存
  MemoryType mem_type = (conn_type == ConnType::UCX) ? MemoryType::CPU : MemoryType::DEFAULT;
  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, mem_type);
  std::cout << "allocate buffer success " << buffer->ptr << std::endl;

  Communicator *comm = new Communicator(buffer);
  std::cout << "create communicator success " << std::endl;

  comm->initServer("192.168.2.241", 2025, conn_type);

  // wait a connection
  std::cout << "Waiting for client connection..." << std::endl;
  while (1) {
    if (comm->checkConn("192.168.2.252", conn_type) == status_t::SUCCESS) {
      std::cout << "Client connected successfully!" << std::endl;
      break;
    }
    sleep(1);
  }

  // UCX需要额外时间确保连接稳定
  if (conn_type == ConnType::UCX) {
    std::cout << "Waiting for UCX connection to stabilize..." << std::endl;
    sleep(3);
  }

  // 示例数据：一个简单的字符串
  char data_to_write[] = "Hello, RDMA!";
  size_t data_size = strlen(data_to_write) + 1; // 包括结尾的 '\0'
  size_t write_bias = 20;                       // 写入起始偏移量
  char read_buffer[256]; // 确保足够大以容纳读取的数据
  size_t read_bias = 20; // 读取起始偏移量
  status_t write_status, read_status;

  if (conn_type == ConnType::RDMA) {
    // RDMA模式：Server主动发送数据给Client
    std::cout << "RDMA Mode: Server sending data to client..." << std::endl;
    
    // 向 ConnBuffer 写入数据 (从 CPU 到缓冲区)
    write_status = buffer->writeFromCpu(data_to_write, data_size, write_bias);
    if (write_status == status_t::SUCCESS) {
      std::cout << "Data written to buffer successfully." << std::endl;
    } else {
      std::cerr << "Failed to write data to buffer." << std::endl;
      goto failed;
    }

    comm->writeTo("192.168.2.252", write_bias, data_size, conn_type); // 写入到对端

    // 等对端修改数据 - 增加等待时间
    sleep(8);
    std::cerr << "Start to read." << std::endl;
    comm->readFrom("192.168.2.252", write_bias, data_size, conn_type); // 从对端读取

    // 从 ConnBuffer 读取数据 (从缓冲区到 CPU)
    read_status = buffer->readToCpu(read_buffer, data_size, read_bias);
    if (read_status == status_t::SUCCESS) {
      std::cout << "Data read from buffer: " << read_buffer << std::endl;
      
      // 验证是否读取到修改后的数据
      if (strstr(read_buffer, "AHA") != nullptr) {
        std::cout << "SUCCESS: Received expected modified data from client!" << std::endl;
      }
    } else {
      std::cerr << "Failed to read data from buffer." << std::endl;
      goto failed;
    }
    std::cerr << "Read success." << std::endl;
    
  } else {
    // UCX模式：Server等待Client发送数据（类似ucx_app）
    std::cout << "UCX Mode: Server waiting for data from client..." << std::endl;
    
    // 初始化缓冲区为零
    char zeros[256] = {0};
    buffer->writeFromCpu(zeros, sizeof(zeros), 0);
    
    // 等待并检查是否收到数据
    std::cout << "Waiting for client to send data..." << std::endl;
    
    bool data_received = false;
    int max_wait_cycles = 40; // 增加等待时间到40秒
    int message_count = 0;
    
    for (int i = 0; i < max_wait_cycles; i++) {
      sleep(1);
      
      // 读取缓冲区检查是否有数据
      memset(read_buffer, 0, sizeof(read_buffer));
      read_status = buffer->readToCpu(read_buffer, 256, 0); // 读取更多数据
      
      if (read_status == status_t::SUCCESS && strlen(read_buffer) > 0) {
        std::cout << "Received data from client: '" << read_buffer << "'" << std::endl;
        data_received = true;
        message_count++;
        
        // 先检查消息内容再清空
        bool is_final_message = (strstr(read_buffer, "completed") != nullptr);
        
        // 验证接收到的数据
        if (strstr(read_buffer, "UCX") != nullptr || strstr(read_buffer, "AHA") != nullptr) {
          std::cout << "SUCCESS: UCX communication test message " << message_count << " received!" << std::endl;
        }
        
        // 清空缓冲区等待下一条消息
        memset(read_buffer, 0, sizeof(read_buffer));
        buffer->writeFromCpu(read_buffer, 256, 0);
        
        // 如果收到第二条消息或最终消息，就退出
        if (is_final_message || message_count >= 2) {
          std::cout << "All messages received, test completed!" << std::endl;
          break;
        }
      }
    }
    
    if (!data_received) {
      std::cout << "WARNING: No data received from client within timeout period" << std::endl;
    }
  }

failed:
  std::cout << "Cleaning up..." << std::endl;
  delete comm;
  buffer.reset();
  std::cout << "Server shutdown complete." << std::endl;
  return 1;
}