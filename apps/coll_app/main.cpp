#include <coll.h>
#include <iostream>

using namespace hddt;

int main(int argc, char *argv[]) {
  MpiOob *oob = new MpiOob(argc, argv);

  // 使用 std::cout 替代 logInfo 输出信息
  std::cout << "current rank: " << oob->rank
            << ", ip: " << oob->get_ip(oob->rank) << std::endl;
  for (int i = 0; i < oob->world_size; ++i) {
    std::cout << "rank " << i << ", oob: " << oob->get_ip(i) << std::endl;
  }

  delete oob; // 不要忘记释放分配的内存

  return 0;
}

// sudo mpirun -np 2 -host ip1,ip2 ./coll_app