## 什么是 HMC？

<p align="center">
  <img src="docs/hmc.png" width="300" />
</p>

HMC 是一个面向异构系统（CPU / GPU / 各类加速器）的通信框架。它提供：

* **统一的内存抽象**：`Memory`
  一个与设备无关的接口，用于申请/释放缓冲区，并在不同设备之间拷贝数据。
* **统一的注册 IO 缓冲区**：`ConnBuffer`
  一个稳定、已注册的缓冲区，作为所有网络传输的数据“暂存区”（staging area）。
* **统一的传输抽象**：`Communicator`
  以同一套接口实现对端缓冲区之间的单边数据读写：
  * `ConnType.RDMA`（依赖平台/设备能力）
  * `ConnType.UCX`（常用于 CPU，以及部分 GPU-direct 配置）
* **控制面（Control plane）用于同步与小消息**
  一个基于 rank 的控制信道，支持 **TCP** 和/或 **UDS**（同机）。

### 支持的设备

HMC 支持 CPU 内存以及多种加速器后端，包括：

* **NVIDIA GPU**（CUDA）
* **AMD GPU**（ROCm）
* **海光平台 / GPU**（依赖具体平台；当你的构建产物中启用了对应后端时可用）
* **寒武纪 MLU**（CNRT / Neuware）
* **摩尔线程 GPU**（MUSA）

> 实际可用性取决于 HMC 的构建方式（例如是否启用 CUDA/ROCm/CNRT/MUSA）以及你本机的运行时/驱动环境。
> 我们当前在英伟达Connect-x系列网卡上开发和支持。

### 核心模型

> 数据移动始终发生在两端已注册缓冲区（`ConnBuffer`）之间，使用 `(offset, size)` 来定位。
> 偏移（offset）由你的应用自行规划，并通过控制消息（tag）来做协同与同步。

# 第 1 部分 — Python（推荐）

## 1. 安装（Python）

HMC 的 Python 绑定基于 C++ 核心，通过 **pybind11** 构建。

### 1.1 前置条件

* Python 3.8+
* C++14+
* CMake ≥ 3.18
* UCX（必需，用于 ConnType.UCX）
  * 请用户自行安装 UCX，并将其安装到：/usr/local/ucx
  * 对于 NVIDIA / AMD GPU 场景，建议安装启用 GPU-Direct（GDR）支持的 UCX 版本（需与你的 CUDA/ROCm 环境匹配）

> 提示：HMC 的 UCX 后端依赖你的系统 UCX 安装路径与运行时库可见性；请确保 /usr/local/ucx/lib 在运行时动态库搜索路径中（例如通过 LD_LIBRARY_PATH 或系统 linker 配置）。

### 1.2 从源码构建并安装

```bash
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC
git submodule update --init --recursive
```

启用 Python 模块进行构建：

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MOD=ON
make -j
```

构建 wheel 并安装：

```bash
cd ..
python -m build
pip install dist/hmc-*.whl
```

验证安装：

```bash
python -c "import hmc; print('hmc imported:', hmc.__file__)"
```

### 1.3（可选）GPU/加速器后端

如果你启用了 CUDA/ROCm/CNRT/MUSA 等后端，请确保在构建 wheel 时同一套工具链与环境变量对构建过程可见。


## 2. 关键概念

* `Session` 是推荐使用的高层 API。

* 所有传输都采用 **缓冲区暂存（buffer-staged）** 模式：

  1. 将 payload 拷贝到本地 `ConnBuffer`（通过 `sess.buf.put(...)`）
  2. 使用 `sess.put_remote(...)` / `sess.get_remote(...)` 发送/接收字节
  3. 再将数据从本地 `ConnBuffer` 拷贝到目标对象（通过 `sess.buf.get_into(...)`）

* 文档和代码里，偏移通常称为 `bias/offset`。


## 3. 示例：CPU 到 CPU 传输（UCX）

### 3.1 服务端

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

### 3.2 客户端

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

# 暂存到本地 ConnBuffer
n = sess.buf.put(payload, offset=0)

# put：本地 [0, n) -> 远端 [0, n)
sess.put_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.UCX)
print("sent", n, "bytes")
```

### 3.3 协调“数据就绪”（推荐）

通常会加一个简单的 tag 握手，让服务端知道什么时候去读：

```python
# 发送方
sess.ctrl_send(peer=0, tag=1)
```

```python
# 接收方
tag = sess.ctrl_recv(peer=1)
assert tag == 1
# 然后在服务端通过 sess.buf.get_into(...) 从 ConnBuffer 读出数据
```


## 4. NumPy 与 PyTorch（CPU）

### 4.1 NumPy 发送/接收

```python
import hmc, numpy as np

x = np.arange(1024, dtype=np.int32)

n = sess.buf.put(x, offset=0)
sess.put_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.UCX)

y = np.empty_like(x)
sess.get_remote(server_ip, 2025, local_off=0, remote_off=0, nbytes=y.nbytes, conn=hmc.ConnType.UCX)
sess.buf.get_into(y, nbytes=y.nbytes, offset=0)
```

说明：

* 源数组建议为 C-contiguous；必要时包装层可能会进行拷贝。
* 目标数组必须可写且 contiguous。

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


## 5. GPU Tensor（高级）

HMC 可以通过使用 tensor 指针（内部 `data_ptr()`）来调用 `writeFromGpu/readToGpu`，从而暂存 CUDA tensor。

```python
import torch

t = torch.randn(1024 * 1024, device="cuda")
n = sess.buf.put(t, offset=0, device="cuda")  # GPU -> ConnBuffer

sess.put_remote(server_ip, 2026, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.RDMA)

recv = torch.empty_like(t)
sess.get_remote(server_ip, 2026, local_off=0, remote_off=0, nbytes=n, conn=hmc.ConnType.RDMA)
sess.buf.get_into(recv, nbytes=n, offset=0, device="cuda")
```

注意事项：

* GPU-direct 依赖平台/驱动/NIC/UCX/RDMA 配置。
* 建议先从 CPU 路径开始，验证协议与正确性后再上 GPU 路径。


## 6. 故障排查（Python）

### 常见错误

* **缓冲区溢出**：`offset + nbytes > buffer_size`

  * 增大 `buffer_size` 或调整 offset
* **未建立连接**：未调用 `connect()` 或 `data_port/conn` 配置错误
* **控制面不匹配（UDS vs TCP）**

  * 使用 `CTRL_TRANSPORT=tcp` 或 `CTRL_TRANSPORT=uds` 强制选择
* **UDS 路径理解错误**

  * UDS 需要的是**完整 socket 文件路径**，不是目录

### 最佳实践

* 定义清晰的协议（offset 布局 + tag 含义）。
* 未收到 ack/同步前，避免覆盖远端 offset。
* 在需要计算通信重叠或并发时，使用 `put_nb/get_nb + wait`。

---

# 第 2 部分 — C++ 核心库（高级 / 集成）

## 7. 安装（C++）

### 7.1 前置条件

* C++14+
* CMake ≥ 3.18

可选：

```bash
sudo apt-get install -y libgtest-dev
```

### 7.2 构建

```bash
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```


## 8. 使用（C++）

### 8.1 核心组件

* `Memory`：跨设备统一申请/拷贝
* `ConnBuffer`：稳定注册缓冲区；所有传输使用 `(offset, size)`
* `CtrlSocketManager`：控制面，基于 rank，TCP/UDS，HELLO 绑定
* `Communicator`：数据面 + 控制面集成：`put/get/putNB/getNB/wait`、`connectTo/initServer`

### 8.2 最小数据面示例（UCX put）

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


## 9. 附录：API 速查表

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
