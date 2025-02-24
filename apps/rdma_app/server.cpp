#include <chrono>
#include <glog/logging.h>
#include <hddt.h>
#include <iostream>
#include <thread>

using namespace hddt;

int main() {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  int device_id = 1; // 根据实际情况设置设备ID
  size_t buffer_size = 1 * 1024 * 1024;

  // 创建连接缓冲区
  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
  std::cout << "Allocate buffer success: " << buffer->ptr << std::endl;

  // 创建通信器
  Communicator *comm = new Communicator(buffer);
  std::cout << "Create communicator success" << std::endl;

  // 初始化服务器
  comm->initServer("192.168.2.241", 2025, ConnType::RDMA);
  comm->addNewRankAddr(1, "192.168.2.241", 2024); // 添加客户端地址
  // comm->addNewRankAddr(0, "192.168.2.241", 2025);     // 服务端自身地址

  // 等待连接建立
  std::this_thread::sleep_for(std::chrono::seconds(3));

  char host_data[1024];
  size_t data_size = 1024;
  size_t read_bias = 0;
  status_t read_status;

  // 接收数据流程
  for (int i = 0; i < 2; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // 从缓冲区读取到主机内存
    read_status = buffer->readToCpu(host_data, data_size, read_bias);
    if (read_status == status_t::SUCCESS) {
      std::cout << "Server get data: " << host_data << std::endl;
    } else {
      std::cerr << "Failed to read data from buffer" << std::endl;
      goto failed;
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

failed:
  // 清理资源
  delete comm;
  buffer.reset();
  std::cout << "Communicator released" << std::endl;
  return 0;
}