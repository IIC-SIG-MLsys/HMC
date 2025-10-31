#include <chrono>
#include <cmath>
#include <fstream>
#include <glog/logging.h>
#include <hmc.h>
#include <iomanip>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <vector>

const std::string server_ip = "192.168.2.240";
const int base_port = 2026;
const int device_id = 1;
const size_t buffer_size = 2048ULL * 1024 * 16; // 32MB

using namespace hmc;
using namespace std;

bool DataVerfier(char *data, int data_size) {
  bool data_valid1 = true;
  for (size_t i = 0; i < std::min(static_cast<size_t>(100), (size_t)data_size);
       ++i) {
    if (data[i] != 'A') {
      data_valid1 = false;
      break;
    }
  }
  return data_valid1;
}

int receive() {
  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
  Communicator *comm = new Communicator(buffer);
  if (comm->initServer(server_ip, 12026, ConnType::RDMA) != status_t::SUCCESS) {
    return -1;
  }
  comm->addNewRankAddr(0, server_ip, 12026);
  comm->addNewRankAddr(1, server_ip, 2026);
  while (comm->checkConn(0, ConnType::RDMA) != status_t::SUCCESS) {
  }
  Memory *mem = new Memory(0);
  for (int power = 4; power <= 30; ++power) {
    const size_t data_size = std::pow(2, power);
    std::cout << "`12312321";
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // UHM interface
    size_t *UHMflag;
    void *UHMreceive;
    mem->allocateBuffer(&UHMreceive, data_size);
    comm->recvDataFrom(0, UHMreceive, data_size, MemoryType::AMD_GPU, UHMflag);
    std::cout << "UHM 数据完整性检查: "
              << (DataVerfier((char *)UHMreceive, data_size) ? "通过" : "失败")
              << std::endl;

    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // Serial interface
    void *Serialreceive;
    size_t *Serialflag;
    mem->allocateBuffer(&Serialreceive, buffer_size);

    size_t buffer_size = buffer_size / 2;
    size_t num_send_chunks = (data_size + buffer_size - 1) / buffer_size;

    for (int i = 0; i < num_send_chunks; i++) {
      comm->recvDataFrom(0, Serialreceive,
                         std::min(buffer_size, data_size - i * buffer_size),
                         MemoryType::AMD_GPU, Serialflag);
    }
    std::cout << "Serial 数据完整性检查: "
              << (DataVerfier((char *)Serialreceive, data_size) ? "通过"
                                                                : "失败")
              << std::endl;
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // G2H2G interface
    void *G2H2Greceive;
    size_t *G2H2Gflag;
    mem->allocateBuffer(&G2H2Greceive, data_size);
    comm->recvDataFrom(0, UHMreceive, data_size, MemoryType::AMD_GPU,
                       G2H2Gflag);
    std::cout << "G2H2G 数据完整性检查: "
              << (DataVerfier((char *)G2H2Greceive, data_size) ? "通过"
                                                               : "失败")
              << std::endl;

    mem->freeBuffer(UHMreceive);
    mem->freeBuffer(G2H2Greceive);
    mem->freeBuffer(Serialreceive);
  }
  return 0;
}

int main() {
  receive();
  return 0;
}