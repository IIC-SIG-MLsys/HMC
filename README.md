# HMC User Guide

**Heterogeneous Memories Communication Framework**

---

## Overview

**HMC** is a communication framework for heterogeneous systems (CPU/GPU/accelerators). It provides:

* A unified **memory abstraction** (`Memory`, `ConnBuffer`)
* A unified **transport abstraction** (`Communicator`) with:
  * **RDMA** backend for device-direct / GPU-direct transfers (platform dependent)
  * **UCX** backend focused on **RMA read/write** (`writeTo/readFrom`)
* A lightweight **TCP control channel** (`CtrlSocketManager`) for synchronization and small messages

Core idea:

> All network operations read/write from a local registered buffer (`ConnBuffer`). Transports move bytes between local/remote buffers.

---

# Part A — C++ (Core Library)

## A1. Build & Installation (C++)

### Prerequisites

* C++14 or newer
* CMake ≥ 3.18
* glog

```bash
sudo apt-get install libgoogle-glog-dev
```

Optional:

* GTest

```bash
sudo apt-get install libgtest-dev
```

Platform SDKs (as needed): CUDA / ROCm / CNRT / MUSA / …

---

### Build from Source

> You can run `build.sh` to quickly build HMC.

```bash
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

Common options:

| Option                  | Meaning                                             |
| ----------------------- | --------------------------------------------------- |
| `-DBUILD_STATIC_LIB=ON` | Build `libhmc.a`                                    |
| `-DBUILD_SHARED_LIB=ON` | Build shared library (if supported by your project) |

---

## A2. C++ Core Concepts

### `Memory`

Device-agnostic allocate/copy API:

* allocate / free
* Host↔Device, Device↔Device copies

### `ConnBuffer`

Stable, registered buffer used for network transport. All operations are `(bias, size)` within this buffer:

* external → buffer: `writeFromCpu`, `writeFromGpu`
* buffer → external: `readToCpu`, `readToGpu`

### `Communicator`

Transport manager providing:

* `initServer()` / `connectTo()`
* RMA: `writeTo()` / `readFrom()` (RDMA/UCX)
* two-sided helper: `send()` / `recv()`
* RDMA-oriented UHM: `sendDataTo()` / `recvDataFrom()`

### `CtrlSocketManager`

TCP control path for sync / small messages (int / POD structs).

---

## A3. C++ Examples

### Example 1 — Memory

```cpp
#include <hmc.h>
#include <vector>
using namespace hmc;

int main() {
  Memory gpu_mem(0, MemoryType::NVIDIA_GPU);
  void* gpu_ptr = nullptr;

  gpu_mem.allocateBuffer(&gpu_ptr, 1 << 20);

  std::vector<char> host(1 << 20, 'A');
  gpu_mem.copyHostToDevice(gpu_ptr, host.data(), host.size());

  gpu_mem.freeBuffer(gpu_ptr);
  return 0;
}
```

---

### Example 2 — One-Sided Transfer (Buffered RMA Pattern)

#### Server

```cpp
#include <hmc.h>
#include <memory>
using namespace hmc;

int main() {
  auto buf = std::make_shared<ConnBuffer>(0, 128 * 1024 * 1024, MemoryType::CPU);
  Communicator comm(buf);

  comm.initServer("192.168.2.244", 2025, ConnType::UCX);

  // Wait for application-level signal, then read from buf->ptr
  // (e.g. CtrlSocketManager or your own signaling)
  return 0;
}
```

#### Client

```cpp
#include <hmc.h>
#include <vector>
using namespace hmc;

int main() {
  std::string server_ip = "192.168.2.244";

  auto buf = std::make_shared<ConnBuffer>(0, 128 * 1024 * 1024, MemoryType::CPU);
  Communicator comm(buf);

  comm.connectTo(server_ip, 2025, ConnType::UCX);

  std::vector<char> payload(4 * 1024 * 1024, 'A');
  buf->writeFromCpu(payload.data(), payload.size(), 0);

  comm.writeTo(server_ip, 0, payload.size(), ConnType::UCX);
  return 0;
}
```

---

### Example 3 — Control Signaling (TCP)

```cpp
#include <hmc.h>
using namespace hmc;

int main() {
  auto &ctrl = CtrlSocketManager::instance();

  // Server: ctrl.startServer("192.168.2.244");
  // Client: ctrl.getCtrlSockFd("192.168.2.244");

  ctrl.sendCtrlInt("192.168.2.244", 1); // "data ready"
  return 0;
}
```

---

## A4. C++ API Reference (Core)

### `Memory`

* `allocateBuffer`, `allocatePeerableBuffer`, `freeBuffer`
* `copyHostToDevice`, `copyDeviceToHost`, `copyDeviceToDevice`

### `ConnBuffer`

* `writeFromCpu`, `readToCpu`, `writeFromGpu`, `readToGpu`

### `Communicator`

* Server/client: `initServer`, `closeServer`, `connectTo`, `disConnect`, `checkConn`
* Transfers: `writeTo`, `readFrom`, `send`, `recv`
* RDMA UHM: `sendDataTo`, `recvDataFrom`

### `CtrlSocketManager`

* `startServer`, `stopServer`, `getCtrlSockFd`
* `sendCtrlInt`, `recvCtrlInt`
* `sendCtrlStruct`, `recvCtrlStruct`
* `closeConnection`, `closeAll`

---

# Part B — Python (SDK / PyBind Layer)

## B1. Build & Installation (Python)

> You can run `build_with_python.sh` to quickly build and install HMC.

HMC provides Python bindings via **pybind11**. Build the C++ core first, then build the wheel.

### 1) Initialize submodules

```bash
git submodule update --init --recursive
```

### 2) Build with Python module enabled

From repo root:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MOD=ON
make -j
```

### 3) Build wheel & install

From repo root:

```bash
python -m build
pip install dist/hmc-*.whl
```

Verify:

```bash
python -c "import hmc; print(hmc)"
```

Notes:

* If your build uses CUDA/ROCm, ensure the same toolchain is visible when building the wheel.

---

## B2. Python Concepts

Python wraps the C++ core and typically exposes:

* `Session` (recommended high-level API)
* `IOBuffer` (thin wrapper around `ConnBuffer`, rarely needed directly)
* Enums: `Status`, `MemoryType`, `ConnType`

Supported Python payload types (typical wrapper behavior):

* `bytes`, `bytearray`, `memoryview`
* NumPy arrays (CPU)
* PyTorch tensors (CPU, and CUDA for RDMA path)

Key rule stays the same:

> Python calls also stage into HMC’s internal buffer, then perform `writeTo/readFrom` under the hood.

---

## B3. Python Quick Start (UCX, CPU payloads)

### Server

```python
import hmc

ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(
    device_id=0,
    buffer_size=128 * 1024 * 1024,
    mem_type=hmc.MemoryType.CPU,
    num_chs=1,
)

sess.init_server(ip, port, conn=hmc.ConnType.UCX)
print("UCX server ready")

# Example: read 1MB from client into a bytearray
client_ip = "192.168.2.248"
dst = bytearray(1024 * 1024)
sess.get_into(client_ip, dst, conn=hmc.ConnType.UCX, bias=0)
print(dst[:16])
```

### Client

```python
import hmc

server_ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(
    device_id=0,
    buffer_size=128 * 1024 * 1024,
    mem_type=hmc.MemoryType.CPU,
    num_chs=1,
)

sess.connect(server_ip, port, conn=hmc.ConnType.UCX)

payload = b"hello hmc"
sess.put(server_ip, payload, conn=hmc.ConnType.UCX, bias=0)
```

---

## B4. NumPy (CPU)

### Send NumPy

```python
import hmc, numpy as np

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.connect("192.168.2.244", 2025, conn=hmc.ConnType.UCX)

x = np.arange(1024, dtype=np.int32)
sess.put("192.168.2.244", x, conn=hmc.ConnType.UCX)
```

### Receive into NumPy

```python
y = np.empty((1024,), dtype=np.int32)
sess.get_into("192.168.2.244", y, conn=hmc.ConnType.UCX)  # size inferred from y.nbytes
```

Notes:

* Source arrays are expected contiguous; wrapper may copy if needed.
* Dest arrays must be writable and contiguous.

---

## B5. PyTorch (CPU)

```python
import hmc, torch

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.connect("192.168.2.244", 2025, conn=hmc.ConnType.UCX)

t = torch.arange(4096, dtype=torch.int32)  # CPU tensor
sess.put("192.168.2.244", t, conn=hmc.ConnType.UCX)

out = torch.empty_like(t)
sess.get_into("192.168.2.244", out, conn=hmc.ConnType.UCX)
```

---

## B6. PyTorch CUDA (RDMA path)

UCX GPU-direct depends on UCX transport configuration; the common default is:

* **CUDA tensors → RDMA backend**

### Send CUDA tensor (RDMA)

```python
import hmc, torch

sess = hmc.create_session(mem_type=hmc.MemoryType.DEFAULT)
sess.connect("192.168.2.244", 2025, conn=hmc.ConnType.RDMA)

t = torch.randn(1024 * 1024, device="cuda")
sess.put_torch_cuda("192.168.2.244", t, conn=hmc.ConnType.RDMA)
```

### Receive into CUDA tensor (RDMA)

```python
recv = torch.empty_like(t)
sess.get_torch_cuda("192.168.2.244", recv, conn=hmc.ConnType.RDMA)
```

---

## B7. Python API Reference (Core)

### `create_session(...) -> Session`

Creates internal `IOBuffer + Communicator`.

Common parameters:

* `device_id`
* `buffer_size`
* `mem_type`
* `num_chs`

### `Session.connect(ip, port, conn)`

Client-side connect.

### `Session.init_server(ip, port, conn)`

Server-side listen.

### `Session.put(ip, data, bias=0, conn=ConnType.UCX, size=None)`

Copy `data` into local buffer and `writeTo()` remote.

### `Session.get(ip, size, bias=0, conn=ConnType.UCX) -> bytes`

`readFrom()` remote into local buffer and return bytes.

### `Session.get_into(ip, dest, bias=0, conn=ConnType.UCX, size=None)`

`readFrom()` then copy into `dest`.

### `Session.disconnect(ip, conn)`

Disconnect peer.

### `Session.close_server()`

Stop listener.

---

## Error Handling & Troubleshooting

Common issues:

* **buffer overflow**: `bias + size > buffer_size`
* **missing connection**: wrong peer `ip` key
* **backend mismatch**: `ConnType.UCX` vs `ConnType.RDMA`
* **device limitations**: GPU-direct depends on platform/driver/transport configuration

Best practice:

* Choose `buffer_size` ≥ max payload + max bias.
* Start with **CPU buffers for UCX** and validate correctness first.
* Add control signaling if your application needs strict consumption ordering.

---

© 2025 SDU spgroup Holding Limited. All rights reserved.
