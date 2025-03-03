/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_COLL_H
#define HDDT_COLL_H

#include <hddt.h>

#include <mpi.h>
#include <unistd.h>
#include <vector>

#define HOSTNAME_MAX 256
#define MAX_IP_SIZE 1024

namespace hddt {

class RankInfoCollection;

class MPIOOB {
public:
  MPIOOB();
  ~MPIOOB();

  int getRank() const { return rank; }
  int getWorldSize() const { return world_size; }

  hddt::RankInfoCollection collectRankInfo();
  void distributeTask();

private:
  int rank;
  int world_size;
  std::string getLocalIP();
};

//
class AllToAll {
public:
  MPIOOB *oob;
  /*When data needs to be transferred between different hosts, network
   * communication technologies such as RDMA are used for efficient data
   * transmission. For data transfers within the same host, GPU memory copy (or
   * similar fast intra-host transfer mechanisms) is employed to achieve faster
   * transfer speeds.*/
  Communicator *comm;

  AllToAll(MPIOOB *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~AllToAll() {}
};

class AllReduce {
  MPIOOB *oob; //
  Communicator *comm;

  AllReduce(MPIOOB *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~AllReduce() {}
};

class ReduceScatter {
  MPIOOB *oob; //
  Communicator *comm;

  ReduceScatter(MPIOOB *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~ReduceScatter() {}
};

} // namespace hddt

#endif
