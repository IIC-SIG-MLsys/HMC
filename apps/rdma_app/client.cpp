#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <thread>

using namespace hmc;

int main() {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  int device_id = 1; // 设备ID与服务端对应
  size_t buffer_size = 1 * 1024 * 1024;

  // 创建连接缓冲区
  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
  std::cout << "Allocate buffer success: " << buffer->ptr << std::endl;

  // 创建通信器
  Communicator *comm = new Communicator(buffer);
  std::cout << "Create communicator success" << std::endl;

  // 初始化客户端连接
  comm->addNewRankAddr(0, "192.168.2.241", 2025); // 添加服务端地址
  comm->addNewRankAddr(1, "192.168.2.241", 2024); // 客户端自身地址

  // 等待连接建立
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // 准备数据
  char *data1 = "Hello!";
  char *data2 = "Bye!";
  size_t data_size1 = strlen(data1) + 1;
  size_t data_size2 = strlen(data2) + 1;
  size_t write_bias = 0;

  // 第一次发送
  status_t write_status = buffer->writeFromCpu(data1, data_size1, write_bias);
  if (write_status != status_t::SUCCESS) {
    std::cerr << "Failed to write first data" << std::endl;
    goto cleanup;
  }
  comm->writeTo(0, write_bias, data_size1); // 发送到rank 0（服务端）
  std::cout << "Client sent: " << data1 << std::endl;

  // 第二次发送
  std::this_thread::sleep_for(std::chrono::seconds(2));
  write_status = buffer->writeFromCpu(data2, data_size2, write_bias);
  if (write_status != status_t::SUCCESS) {
    std::cerr << "Failed to write second data" << std::endl;
    goto cleanup;
  }
  comm->writeTo(0, write_bias, data_size2);
  std::cout << "Client sent: " << data2 << std::endl;

cleanup:
  // 等待操作完成
  std::this_thread::sleep_for(std::chrono::seconds(2));
  delete comm;
  buffer.reset();
  std::cout << "Client shutdown" << std::endl;
  return 0;
}