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

  // RDMA模式需要Client也作为Server，UCX模式不需要
  if (conn_type == ConnType::RDMA) {
    comm->initServer("192.168.2.252", 2024, conn_type);
  }

  std::cout << "Connecting to server..." << std::endl;
  comm->connectTo("192.168.2.241", 2025, conn_type);

  if (conn_type == ConnType::RDMA) {
    // RDMA模式：原有逻辑，等待Server发送数据然后修改
    std::cout << "RDMA Mode: Waiting for data from server..." << std::endl;
    
    sleep(3); // 等待对端写入数据

    char data_to_write[] = "Hello, AHA!";
    size_t data_size = strlen(data_to_write) + 1; // 包括结尾的 '\0'
    size_t write_bias = 20;                       // 写入起始偏移量
    char read_buffer[256]; // 确保足够大以容纳读取的数据
    size_t read_bias = 20; // 读取起始偏移量
    status_t write_status, read_status;

    read_status = buffer->readToCpu(read_buffer, data_size, read_bias);
    if (read_status == status_t::SUCCESS) {
      std::cout << "Data read from buffer: " << read_buffer << std::endl;
    } else {
      std::cerr << "Failed to read data from buffer." << std::endl;
      goto failed;
    }

    // 修改内存内容
    // 向 ConnBuffer 写入数据 (从 CPU 到缓冲区)
    write_status = buffer->writeFromCpu(data_to_write, data_size, write_bias);
    if (write_status == status_t::SUCCESS) {
      std::cout << "Data written to buffer successfully." << std::endl;
    } else {
      std::cerr << "Failed to write data to buffer." << std::endl;
      goto failed;
    }

    // 等待对端读取 - 增加等待时间让Server有足够时间读取
    sleep(5);

    std::cout << "Disconnecting from server..." << std::endl;
    comm->disConnect("192.168.2.241", conn_type);
    
  } else {
    // UCX模式：Client主动发送数据给Server（类似ucx_app）
    std::cout << "UCX Mode: Client sending data to server..." << std::endl;
    
    // UCX需要更长的等待时间确保连接稳定
    sleep(5);
    
    // 准备要发送的数据
    const char *data1 = "Hello via UCX!";
    const char *data2 = "UCX communication test completed!";
    size_t data_size1 = strlen(data1) + 1;
    size_t data_size2 = strlen(data2) + 1;
    
    // 发送第一条消息
    buffer->writeFromCpu((void*)data1, data_size1, 0);
    std::cout << "Sending message 1: \"" << data1 << "\"" << std::endl;
    
    status_t send_status = comm->writeTo("192.168.2.241", 0, data_size1, conn_type);
    if (send_status == status_t::SUCCESS) {
      std::cout << "Message 1 sent successfully" << std::endl;
    } else {
      std::cerr << "Failed to send message 1" << std::endl;
    }
    
    // 等待
    sleep(3); // 增加等待时间
    
    // 发送第二条消息
    buffer->writeFromCpu((void*)data2, data_size2, 0);
    std::cout << "Sending message 2: \"" << data2 << "\"" << std::endl;
    
    send_status = comm->writeTo("192.168.2.241", 0, data_size2, conn_type);
    if (send_status == status_t::SUCCESS) {
      std::cout << "Message 2 sent successfully" << std::endl;
    } else {
      std::cerr << "Failed to send message 2 (connection may be closed)" << std::endl;
    }
    
    // 等待服务器处理
    sleep(3);
    
    std::cout << "UCX communication test completed on client side" << std::endl;
    comm->disConnect("192.168.2.241", conn_type);
  }

failed:
  sleep(5); // wait server 的远程读活动
  delete comm;
  buffer.reset();
  return 1;
}