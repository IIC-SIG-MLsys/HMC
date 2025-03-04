/**
 * @copyright Copyright (c) 2025, SDU SPgroup Holding Limited
 */
#include "driver_manager.h"
#include <infiniband/verbs.h>

namespace hddt {

bool support_rdma() {
  struct ibv_device **dev_list;
  int num_devices;

  // get device list
  dev_list = ibv_get_device_list(&num_devices);
  if (num_devices == 0) {
    std::cerr << "No RDMA devices found." << std::endl;
    return false;
  }

  // check if any device supports RDMA
  for (int i = 0; i < num_devices; ++i) {
    struct ibv_context *context;
    context = ibv_open_device(dev_list[i]);
    if (context != nullptr) {
      ibv_close_device(context);
      ibv_free_device_list(dev_list); // free device list
      return true;                    // found a device that supports RDMA
    }
  }

  ibv_free_device_list(dev_list); // free device list
  return false;                   // no device supports RDMA
}

} // namespace hddt
