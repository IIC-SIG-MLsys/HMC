/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_DRIVER_H
#define HDDT_DRIVER_H

#include "../utils/log.h"
#include "gpu_interface.h"
#include <hddt.h>
#include <mem.h>

namespace hddt {
/*
gpu resource manager
*/

class DriverManager {
public:
  virtual status_t init() = 0;
  virtual void free() = 0;
  virtual ~DriverManager() = default;
};

// GPU 设备信息结构
struct GPUDeviceInfo {
    int deviceId;
    uint64_t freeMemory;
    uint64_t totalMemory;
    std::string pcieBusId;
    MemoryType vendor;
};

class GPUManager : public DriverManager {
public:
  GPUManager() {}

  int getDeviceCount() const;
  GPUDeviceInfo getDeviceInfo(int device_id) const;

  status_t init() override;
  void free() override {};

private:
  std::vector<GPUDeviceInfo> devices;
#if ENABLE_NEUWARE
  std::vector<CNcontext> contexts;
#endif
};

/*
rdma resource manager
*/
class RDMAManager: public DriverManager {}; // TODO
bool support_rdma();

} // namespace hddt

#endif