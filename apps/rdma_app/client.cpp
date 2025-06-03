#include <chrono>
#include <glog/logging.h>
#include <iostream>
#include <thread>
#include <cmath>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <iomanip>
#include <hmc.h>

using namespace hmc;
using namespace std;
using namespace std::chrono;

const int base_port = 2025;
const int device_id = 1;
const size_t buffer_size = 2048ULL * 1024 * 16;

void UHMperformanceTestSendLogic(const std::string &server_ip) {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
  Communicator* comm = new Communicator(buffer);

  Memory* mem = new Memory(0);

  comm->initServer(server_ip, 2026, ConnType::RDMA);
  comm->addNewRankAddr(0, server_ip, 12026);
  comm->addNewRankAddr(1, server_ip, 2026);
  comm->connectTo(0, ConnType::RDMA);

  for (int power = 4; power <= 30; ++power) {
    const size_t data_size = std::pow(2, power);
    std::vector<uint8_t> host_data(data_size, 'A');
    mem->allocateBuffer(&buffer->ptr, buffer_size);
    mem->copyHostToDevice(buffer->ptr, host_data.data(), data_size);
  

    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // UHM interface
    auto sendDataTo_start_time = high_resolution_clock::now();

    comm->sendDataTo(0, buffer->ptr, buffer_size, MemoryType::AMD_GPU);

    auto sendDataTo_end_time = high_resolution_clock::now();

    auto sendDataTo_write_time = duration_cast<microseconds>(sendDataTo_end_time - sendDataTo_start_time);
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // Serial interface
    usleep(10000);
    void *send1;
    mem->allocateBuffer(&send1, data_size);
    mem->copyHostToDevice(send1, host_data.data(), data_size);
    
    size_t buffer_size = buffer_size / 2;
    size_t num_send_chunks =
      (data_size + buffer_size - 1) / buffer_size;

    long long total_time = 0;
    char* send_buffer = (char*)send1;

    for(int i = 0; i < num_send_chunks; i++){
      std::cout << "i = " << i << std::endl;
      auto Serial_write_start_time = high_resolution_clock::now();
      comm->sendDataTo(0, send_buffer + i * buffer_size, std::min(buffer_size, data_size - i * buffer_size), MemoryType::AMD_GPU);
      auto Serial_write_end_time = high_resolution_clock::now();
      usleep(10000);
      total_time += duration_cast<microseconds>(
            Serial_write_end_time - Serial_write_start_time).count();
    }
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/

    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // G2H2G interface
    auto G2H2G_sendDataTo_start_time = high_resolution_clock::now();
    void *sendG2H2G;
    mem->allocateBuffer(&sendG2H2G, data_size);
    mem->copyHostToDevice(sendG2H2G, host_data.data(), data_size);

    comm->sendDataTo(0, sendG2H2G, buffer_size, MemoryType::AMD_GPU);

    auto G2H2G_end_time = high_resolution_clock::now();

    auto G2H2G_write_time = duration_cast<microseconds>(G2H2G_end_time - G2H2G_sendDataTo_start_time);
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    // Ideal RDMA interface
    auto Ideal_sendDataTo_start_time = high_resolution_clock::now();

    comm->writeTo(0, 0, buffer_size);

    auto Ideal_sendDataTo_end_time = high_resolution_clock::now();

    auto Ideal_write_time = duration_cast<microseconds>(Ideal_sendDataTo_end_time - Ideal_sendDataTo_start_time); 
    /*————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————*/
    string filename = "performanceTest.csv";
      ofstream header_file(filename, ios::app);
      if (header_file.tellp() == 0) {
          header_file << "Data Size,Ideal,Serial,UHM,G2H2G\n";
      }
      header_file.close();
      ofstream file(filename, ios::app); // Append mode
      if (file.is_open()) {
        file << power  << "," << Ideal_write_time.count() << "," << total_time <<"," << G2H2G_write_time.count() <<"\n";
        file.close();
        cout << "Data written to " << filename << endl;
      } else {
          cerr << "Failed to open file " << filename << endl;
      }
   }
}

void performanceTest(std::vector<std::string> ips) {
  for(std::string ip : ips) {
      string filename = "performanceTest.csv";
      ofstream header_file(filename, ios::app);
      if (header_file.tellp() == 0) {
          header_file << "Data Size,Ideal,Serial,UHM,G2H2G\n";
      }
      header_file.close();
      ofstream file(filename, ios::app); // Append mode
      if (file.is_open()) {
          file<< ip << "\n";
          file.close();
          cout << "Data written to " << filename << endl;
      } else {
          cerr << "Failed to open file " << filename << endl;
      }
      UHMperformanceTestSendLogic(ip);
    } 
}

int main() {
  std::string remote_ip_hygon1 = "192.168.2.240"; // hygon1
  std::string remote_ip_hygon2 = "192.168.2.253"; // hygon2
  std::string remote_ip_nv2 = "192.168.2.243"; // nv2
  std::string remote_ip_nv3 = "192.168.2.247"; // nv3
  std::string remote_ip_cam2 = "192.168.2.252"; // cam2
  std::string remote_ip_moore = "192.168.2.238"; // moore

  std::vector<std::string> ips;

  ips.push_back(remote_ip_hygon1);
  // ips.push_back(remote_ip_nv2);
  // ips.push_back(remote_ip_cam2);

  performanceTest(ips);
}