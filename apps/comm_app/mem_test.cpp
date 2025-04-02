#include <iostream>
#include <mem.h>

using namespace hmc;

int main() {
  /* GPU memory test */
  // Memory *mem_ops = new Memory(1);
  Memory *mem_ops = new Memory(1, hmc::MemoryType::DEFAULT);
  void *addr;
  mem_ops->allocateBuffer(&addr, 1024);

  uint8_t data[] = "Hello World!\n";
  mem_ops->copyHostToDevice(addr, data, sizeof(data));

  char host[1024];
  mem_ops->copyDeviceToHost(host, addr, sizeof(data));
  std::cout << "Server get Data: " << host << std::endl;

  delete mem_ops;

  return 1;
}