/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_COLL_H
#define HDDT_COLL_H

#include <hddt.h>

#include <ifaddrs.h>
#include <map>
#include <mpi.h>
#include <netdb.h>


#define HOSTNAME_MAX 256
#define MAX_IP_SIZE 1024

namespace hddt {
class MpiOob {
public:
  int world_size;
  int rank;
  std::map<int, std::string> ip_tables;

  MpiOob(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->world_size);

    // get valid local ip address
    std::string local_ip = get_local_ip();
    if (local_ip.empty()) {
      throw std::runtime_error("Failed to get local IP address");
    }

    ip_tables[rank] = local_ip;

    // exchange ip
    exchange_ip();
  }
  ~MpiOob() { MPI_Finalize(); }

  void exchange_ip();
  std::string get_ip(int rank);

private:
  std::string get_local_ip();
};

//
class AllToAll {
public:
  MpiOob *oob;
  /*When data needs to be transferred between different hosts, network
   * communication technologies such as RDMA are used for efficient data
   * transmission. For data transfers within the same host, GPU memory copy (or
   * similar fast intra-host transfer mechanisms) is employed to achieve faster
   * transfer speeds.*/
  Communicator *comm;

  AllToAll(MpiOob *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~AllToAll() {}
};

class AllReduce {
  MpiOob *oob; //
  Communicator *comm;

  AllReduce(MpiOob *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~AllReduce() {}
};

class ReduceScatter {
  MpiOob *oob; //
  Communicator *comm;

  ReduceScatter(MpiOob *oob, Communicator *comm) : oob(oob), comm(comm) {}
  ~ReduceScatter() {}
};

} // namespace hddt

#endif