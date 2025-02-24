#include "../utils/log.h"
#include <coll.h>

namespace hddt {

void MpiOob::exchange_ip() {
  int max_ip_len = MAX_IP_SIZE;
  char *all_ips = new char[world_size * max_ip_len];
  char my_ip[MAX_IP_SIZE];
  strcpy(my_ip, ip_tables[rank].c_str());

  MPI_Allgather(my_ip, max_ip_len, MPI_CHAR, all_ips, max_ip_len, MPI_CHAR,
                MPI_COMM_WORLD);

  for (int i = 0; i < world_size; ++i) {
    char *ip = all_ips + i * max_ip_len;
    ip_tables[i] = std::string(ip);
  }

  delete[] all_ips;
}

std::string MpiOob::get_ip(int rank) {
  auto it = ip_tables.find(rank);
  if (it != ip_tables.end()) {
    return it->second;
  } else {
    throw std::out_of_range("Rank not found in ip_tables");
  }
}

std::string MpiOob::get_local_ip() {
  struct ifaddrs *ifaddr, *ifa;
  int family, s;
  char host[NI_MAXHOST];

  if (getifaddrs(&ifaddr) == -1) {
    logError("getifaddrs failed.");
    return "";
  }

  for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == NULL) {
      continue;
    }

    family = ifa->ifa_addr->sa_family;

    if (family == AF_INET || family == AF_INET6) {
      s = getnameinfo(ifa->ifa_addr,
                      (family == AF_INET) ? sizeof(struct sockaddr_in)
                                          : sizeof(struct sockaddr_in6),
                      host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
      if (s != 0) {
        logDebug("getnameinfo() failed: %s\n", gai_strerror(s));
        continue;
      }

      // Skip loopback addresses
      if (strcmp(host, "127.0.0.1") == 0 || strcmp(host, "::1") == 0) {
        continue;
      }

      // Return the first non-loopback address
      freeifaddrs(ifaddr);
      return std::string(host);
    }
  }

  freeifaddrs(ifaddr);
  return "";
}

/*
 * AllToAll
 */

} // namespace hddt
