#include <glog/logging.h>
#include <hmc.h>

using namespace hmc;

int main() {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  int device_id = 0;
  size_t buffer_size = 1 * 1024 * 1024;

  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
  std::cout << "allocate buffer success " << buffer->ptr << std::endl;

  Communicator *comm = new Communicator(buffer);
  std::cout << "create communicator success " << std::endl;

  comm->initServer("192.168.2.236", 2024, ConnType::RDMA);
  comm->addNewRankAddr(1, "192.168.2.236", 2025);
  comm->addNewRankAddr(2, "192.168.2.251", 2025);

  int to_rank = 1;

  comm->connectTo(to_rank, ConnType::RDMA);

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

  // 等待对端读取
  sleep(0);

  comm->disConnect(1, ConnType::RDMA);

failed:
  sleep(5); // wait server 的远程读活动
  delete comm;
  buffer.reset();
  return 1;
}