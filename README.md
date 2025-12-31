## What is HMC?

<p align="center">
  <img src="docs/hmc.png" width="300" />
</p>

HMC(Heterogeneous Memory Communication) is a communication framework for heterogeneous systems (CPU / GPU / accelerators). It provides:

* **Unified memory abstraction**: `Memory`
  A device-agnostic interface for allocating/freeing buffers and copying data across devices.
* **Unified registered IO buffer**: `ConnBuffer`
  A stable, registered buffer used as the staging area for all network transfers.
* **Unified transport**: `Communicator`
  A single interface for one-sided data movement between peers’ buffers:
  * `ConnType.RDMA` (platform/device dependent)
  * `ConnType.UCX` (commonly used for CPU and some GPU-direct configurations)
* **Control plane** for synchronization & small messages
  A rank-based control channel over **TCP** and/or **UDS** (same-host).

### Supported devices
HMC supports CPU memory and multiple accelerator backends, including:
* **NVIDIA GPUs** (CUDA)
* **AMD GPUs** (ROCm)
* **Hygon platforms / GPUs** (platform-dependent; enabled when the corresponding backend is available in your build)
* **Cambricon MLUs** (CNRT / Neuware)
* **Moore Threads GPUs** (MUSA)

> Availability depends on how HMC is built (e.g., enabling CUDA/ROCm/CNRT/MUSA) and on the runtime/driver environment on your machine.
> We develop and test on Nvidia ConnectX NICs, so optimal performance and compatibility are expected in environments with similar hardware and driver configurations.

### Core model

> Data movement always happens between two peers’ registered buffers (`ConnBuffer`) using `(offset, size)`.
> Your application decides offsets and uses a control message (tag) to coordinate.


# Part 1 — Python (Recommended)

## 1. Installation (Python)

HMC Python bindings are built via **pybind11** on top of the C++ core.

### 1.1 Prerequisites

* Python 3.8+
* C++14+
* CMake ≥ 3.18
* UCX (required for ConnType.UCX)
  * Users must install UCX themselves and place it under: /usr/local/ucx
  * For NVIDIA / AMD GPU scenarios, it’s recommended to install a UCX build with GPU-Direct / GDR support enabled (and matched to your CUDA/ROCm toolchain)

> Tip: Make sure the UCX runtime libraries are visible at runtime (e.g., /usr/local/ucx/lib is on the dynamic linker search path via LD_LIBRARY_PATH or system linker configuration).

### 1.2 Build & Install from Source

```bash
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC
git submodule update --init --recursive
```

Build with Python module enabled:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MOD=ON
make -j
```

Build wheel & install:

```bash
cd ..
python -m build
pip install dist/hmc-*.whl
```

Verify:

```bash
python -c "import hmc; print('hmc imported:', hmc.__file__)"
```

### 1.3 (Optional) GPU/Accelerator Backends

If you enable CUDA/ROCm/CNRT/MUSA backends, make sure the same toolchain is visible during build.

---

## 2. Key Concepts

* `Session` is the recommended high-level API.
* All transfers are **buffer-staged**:

  1. Copy payload into local `ConnBuffer` (via `sess.buf.put(...)`)
  2. Send/receive bytes using `sess.put_remote(...)` / `sess.get_remote(...)`
  3. Copy bytes out of local `ConnBuffer` to your destination (via `sess.buf.get_into(...)`)
* Offsets are called `bias/offset` in docs and code.

## 3. Example: CPU-to-CPU Transfer (UCX)

### 3.1 Server

```python
import hmc

ip = "192.168.2.244"
rank = 0

sess = hmc.create_session(
    device_id=0,
    buffer_size=128 * 1024 * 1024,
    mem_type=hmc.MemoryType.CPU,
    num_chs=1,
    local_ip=ip,
)

sess.init_server(
    bind_ip=ip,
    ucx_port=2025,
    rdma_port=2026,
    ctrl_tcp_port=5001,
    self_id=rank,
    ctrl_uds_dir="/tmp",
)

print("UCX server ready")
```

### 3.2 Client

```python
import hmc

server_ip = "192.168.2.244"
my_ip = "192.168.2.248"

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU, local_ip=my_ip)

sess.connect(
    peer_id=0,
    self_id=1,
    peer_ip=server_ip,
    data_port=2025,
    ctrl_tcp_port=5001,
    conn=hmc.ConnType.UCX,
)

payload = b"hello hmc"

# stage into local ConnBuffer
n = sess.buf.put(payload, offset=0)

# put local [0, n) -> remote [0, n)
sess.put_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.UCX)
print("sent", n, "bytes")
```

### 3.3 Coordinating “data ready” (recommended)

You typically add a simple tag handshake so the server knows when to read:

```python
# sender side
sess.ctrl_send(peer=0, tag=1)
```

```python
# receiver side
tag = sess.ctrl_recv(peer=1)
assert tag == 1
# then read data from server-side ConnBuffer using sess.buf.get_into(...)
```

> Your application should define a protocol: which offset holds which message, and what tags mean.


## 4. NumPy & PyTorch (CPU)

### 4.1 NumPy Send/Recv

```python
import hmc, numpy as np

x = np.arange(1024, dtype=np.int32)

n = sess.buf.put(x, offset=0)
sess.put_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.UCX)

y = np.empty_like(x)
sess.get_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=y.nbytes, conn=hmc.ConnType.UCX)
sess.buf.get_into(y, nbytes=y.nbytes, offset=0)
```

Notes:

* Source arrays should be C-contiguous; wrapper may copy if needed.
* Destination arrays must be writable and contiguous.

### 4.2 PyTorch CPU Tensor

```python
import torch

t = torch.arange(4096, dtype=torch.int32)
n = sess.buf.put(t, offset=0)

sess.put_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.UCX)

out = torch.empty_like(t)
sess.get_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=out.numel() * out.element_size(), conn=hmc.ConnType.UCX)
sess.buf.get_into(out, nbytes=out.numel() * out.element_size(), offset=0)
```

## 5. GPU Tensors (Advanced)

HMC can stage CUDA tensors via `writeFromGpu/readToGpu` by using the tensor pointer (`data_ptr()` internally).

```python
import torch

t = torch.randn(1024 * 1024, device="cuda")
n = sess.buf.put(t, offset=0, device="cuda")  # GPU -> ConnBuffer

sess.put_remote(server_ip, 2026, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.RDMA)

recv = torch.empty_like(t)
sess.get_remote(server_ip, 2026, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.RDMA)
sess.buf.get_into(recv, nbytes=n, offset=0, device="cuda")
```

Caveats:

* GPU-direct depends on platform/driver/NIC/UCX/RDMA configuration.
* Start with CPU path first to validate protocol and correctness.

---

## 6. Troubleshooting (Python)

### Common errors

* **Buffer overflow**: `offset + nbytes > buffer_size`
  * Increase `buffer_size` or adjust offsets
* **Not connected**: `connect()` not called or wrong `data_port/conn`
* **Ctrl mismatch (UDS vs TCP)**
  * Use `CTRL_TRANSPORT=tcp` or `CTRL_TRANSPORT=uds` to force a choice
* **UDS path confusion**
  * UDS uses a **full socket file path**, not a directory

### Best practices
* Define a clear protocol (offset layout + tag meanings).
* Avoid overwriting remote offsets without an ack handshake.
* Use `put_nb/get_nb + wait` when you need overlap or concurrent transfers.

---

# Part 2 — C++ Core Library (Advanced / Integration)

## 7. Installation (C++)

### 7.1 Prerequisites

* C++14+
* CMake ≥ 3.18

Optional:

```bash
sudo apt-get install -y libgtest-dev
```

### 7.2 Build

```bash
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## 8. Usage (C++)

### 8.1 Core Components

* `Memory`: unified allocate/copy across devices
* `ConnBuffer`: stable registered buffer; all transfers use `(offset, size)`
* `CtrlSocketManager`: ctrl plane, rank-based, TCP/UDS, HELLO binding
* `Communicator`: data plane + ctrl integration: `put/get/putNB/getNB/wait`, `connectTo/initServer`

### 8.2 Minimal Data-Plane Example (UCX put)

```cpp
#include <hmc.h>
#include <memory>
#include <vector>
#include <string>

using namespace hmc;

int main() {
  auto buf = std::make_shared<ConnBuffer>(0, 128ull * 1024 * 1024, MemoryType::CPU);
  Communicator comm(buf);

  std::string server_ip = "192.168.2.244";
  uint16_t data_port = 2025;
  uint16_t ctrl_tcp_port = 5001;

  Communicator::CtrlLink link;
  link.transport = Communicator::CtrlTransport::TCP;
  link.ip = server_ip;
  link.port = ctrl_tcp_port;

  comm.connectTo(/*peer_id=*/0, /*self_id=*/1, server_ip, data_port, link, ConnType::UCX);

  std::vector<char> payload(1024, 'A');
  buf->writeFromCpu(payload.data(), payload.size(), 0);

  comm.put(server_ip, data_port, /*local_off=*/0, /*remote_off=*/0, payload.size(), ConnType::UCX);
  return 0;
}
```

## 9. Appendix: API Cheatsheet

### Python

* `create_session(...) -> Session`
* `Session.init_server(...)`
* `Session.connect(...)`
* `Session.put_remote(...) / get_remote(...)`
* `Session.put_nb(...) / get_nb(...) / wait(...)`
* `Session.ctrl_send(...) / ctrl_recv(...)`
* `sess.buf.put(...) / sess.buf.get_into(...) / buffer_copy_within(...)`

### C++

* `ConnBuffer::{writeFromCpu/readToCpu/writeFromGpu/readToGpu/copyWithin}`
* `Communicator::{initServer/connectTo/put/get/putNB/getNB/wait/ctrlSend/ctrlRecv}`
* `CtrlSocketManager::{start/connectTcp/connectUds/sendU64/recvU64/...}`

---

© 2025 SDU spgroup Holding Limited. All rights reserved.
