#include <hddt.h>
#include <glog/logging.h>

using namespace hddt;

int main() {
  FLAGS_colorlogtostderr = true;
  FLAGS_alsologtostderr = true;

  int device_id = 0;
  size_t buffer_size = 1*1024*1024;

  auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size);
  std::cout << "allocate buffer success " << buffer->ptr << std::endl;

  Communicator* comm = new Communicator(buffer);
  std::cout << "create communicator success " << std::endl;

  comm->addNewRankAddr(1, "192.168.2.240", 2025);

  comm->connectTo(1, ConnType::RDMA);
  sleep(1);
  delete comm;
  buffer.reset();
  return 1;
}