#include <coll.h>
#include <iostream>

#include "utils/proto.h"

using namespace hddt;

int main(int argc, char *argv[]) {
  MPIOOB mpiOob; // 自动初始化 MPI

  hddt::RankInfoCollection rankInfo = mpiOob.collectRankInfo();

  if (mpiOob.getRank() == 0) {
    std::cout << "Collected Rank Info:\n";
    for (const auto &info : rankInfo.infos()) {
      std::cout << "  Rank: " << info.rank() << ", Host: " << info.hostname()
                << ", IP: " << info.ip_address()
                << ", Time: " << info.timestamp() << "\n";
    }
  }

  mpiOob.distributeTask(); // 任务分发
  return 0;                // 自动调用 MPI_Finalize()
}

// sudo mpirun -np 2 -host ip1,ip2 ./coll_app