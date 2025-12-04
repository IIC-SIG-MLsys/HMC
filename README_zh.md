# HMC 用户指南

**异构内存通信框架（Heterogeneous Memories Communication Framework）**

---

## 概览

**HMC** 面向异构系统（CPU / GPU / 各类加速器）的高性能通信框架，提供：

* 统一的**内存抽象**：`Memory`、`ConnBuffer`
* 统一的**传输抽象**：`Communicator`，支持：
  * **RDMA** 后端：面向设备直连 / GPU-direct（取决于平台与驱动能力）
  * **UCX** 后端：当前设计以 **RMA read/write** 为主（对应 `writeTo/readFrom`）
* 轻量级 **TCP 控制通道**：`CtrlSocketManager`，用于同步/信令/小消息

核心思想一句话：

> 所有网络收发都围绕本地的注册缓冲区（`ConnBuffer`）进行；传输层只负责把“本地 buffer 的某段字节”搬到“远端 buffer 的某段字节”。

---

# Part A — C++（核心库）

## A1. 编译与安装（C++）

### 依赖

* C++14 或更新
* CMake ≥ 3.18
* glog（日志）

```bash
sudo apt-get install libgoogle-glog-dev
```

可选：

* GTest（单元测试）

```bash
sudo apt-get install libgtest-dev
```

设备侧 SDK（按需）：CUDA / ROCm / CNRT / MUSA / …

---

### 从源码编译

> 你可以运行 `build.sh` 来快速构建HMC。

```bash
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

常用 CMake 选项：

| 选项                      | 含义               |
| ----------------------- | ---------------- |
| `-DBUILD_STATIC_LIB=ON` | 构建静态库 `libhmc.a` |
| `-DBUILD_SHARED_LIB=ON` | 构建动态库（若项目支持）     |

---

## A2. C++ 核心概念

### `Memory`：统一的设备内存管理

`Memory` 封装 CUDA/ROCm/MLU/MUSA 等平台运行时，提供：

* allocate / free（在指定设备上分配/释放）
* Host↔Device、Device↔Device 的 copy

---

### `ConnBuffer`：传输用注册缓冲区

`ConnBuffer` 拥有稳定指针 `ptr` 与固定大小 `buffer_size`。
所有传输都用 `(bias, size)` 表达 buffer 内的一个切片范围：

* **外部 → buffer**

  * `writeFromCpu(src, size, bias)`
  * `writeFromGpu(src, size, bias)`
* **buffer → 外部**

  * `readToCpu(dst, size, bias)`
  * `readToGpu(dst, size, bias)`

安全规则：

> 任意操作必须满足：`bias + size <= buffer_size`，否则返回 `status_t::ERROR`。

---

### `Communicator`：RDMA/UCX 统一传输入口

`Communicator` 为某个 `ConnBuffer` 管理连接与传输：

* 建连/监听：

  * `initServer(ip, port, ConnType)`
  * `connectTo(ip, port, ConnType)`
* **一侧语义（RMA）**：

  * `writeTo(ip, bias, size, ConnType)`：本地 buffer → 远端 buffer
  * `readFrom(ip, bias, size, ConnType)`：远端 buffer → 本地 buffer
* **双侧语义（同步辅助）**：

  * `send()` / `recv()`：内部配合控制信号实现阻塞语义（更偏 RDMA 使用方式）
* **UHM 高层接口**：

  * `sendDataTo()` / `recvDataFrom()`：主要面向 RDMA 后端（UCX 当前不推荐走这条路径）

---

### `CtrlSocketManager`：TCP 控制/同步通道

用于：

* 同步双方时序（例如“数据已写完/可以读取”）
* 传小消息（int / POD struct）
* 构建简单协议（如 “Finished”）

常用接口：

* `startServer(bindIp)` / `stopServer()`
* `getCtrlSockFd(ip)`（客户端连接）
* `sendCtrlInt` / `recvCtrlInt`
* `sendCtrlStruct` / `recvCtrlStruct`

---

## A3. C++ 使用示例

### 示例 1：内存分配与拷贝

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

### 示例 2：一侧传输（推荐模式：buffer + writeTo/readFrom）

#### 服务端

```cpp
#include <hmc.h>
#include <memory>
using namespace hmc;

int main() {
  auto buf = std::make_shared<ConnBuffer>(0, 128 * 1024 * 1024, MemoryType::CPU);
  Communicator comm(buf);

  comm.initServer("192.168.2.244", 2025, ConnType::UCX);

  // 等待应用层信号（例如 CtrlSocketManager），然后从 buf->ptr 读取数据
  return 0;
}
```

#### 客户端

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

说明：

* `writeTo()` 只搬运 buffer 字节，不提供“消息已就绪/已消费”的强同步语义；
* 如果需要严格时序，请加控制信令（见下一个示例）。

---

### 示例 3：TCP 控制信令（同步）

```cpp
#include <hmc.h>
using namespace hmc;

int main() {
  auto &ctrl = CtrlSocketManager::instance();

  // 服务端：ctrl.startServer("192.168.2.244");
  // 客户端：ctrl.getCtrlSockFd("192.168.2.244");

  ctrl.sendCtrlInt("192.168.2.244", 1); // “数据就绪”
  return 0;
}
```

---

## A4. C++ API 速查（核心）

### `Memory`

* `allocateBuffer` / `allocatePeerableBuffer` / `freeBuffer`
* `copyHostToDevice` / `copyDeviceToHost` / `copyDeviceToDevice`

### `ConnBuffer`

* `writeFromCpu` / `readToCpu`
* `writeFromGpu` / `readToGpu`

### `Communicator`

* 建连：`initServer` / `closeServer` / `connectTo` / `disConnect` / `checkConn`
* 传输：`writeTo` / `readFrom` / `send` / `recv`
* UHM（偏 RDMA）：`sendDataTo` / `recvDataFrom`

### `CtrlSocketManager`

* `startServer` / `stopServer` / `getCtrlSockFd`
* `sendCtrlInt` / `recvCtrlInt`
* `sendCtrlStruct` / `recvCtrlStruct`
* `closeConnection` / `closeAll`

---

# Part B — Python（SDK / PyBind 封装层）

## B1. 编译与安装（Python）

> 你可以运行 `build_with_python.sh` 来快速构建并安装 hmc。

HMC 通过 **pybind11** 提供 Python 绑定。总体流程：

1. 初始化子模块
2. CMake 开启 Python 模块编译
3. 构建 wheel 并安装

### 1）初始化 submodules

```bash
git submodule update --init --recursive
```

### 2）启用 Python 模块编译

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_MOD=ON
make -j
```

### 3）构建 wheel 并安装

回到仓库根目录执行：

```bash
python -m build
pip install dist/hmc-*.whl
```

验证：

```bash
python -c "import hmc; print(hmc)"
```

注意：

* 如果涉及 CUDA/ROCm 等，请确保编译 wheel 时对应工具链、驱动环境可见且一致。

---

## B2. Python 侧概念

Python 层通常提供：

* `Session`：高层推荐接口（封装 IOBuffer + Communicator）
* `IOBuffer`：`ConnBuffer` 的 Python 包装（一般无需直接用）
* 枚举：`Status` / `MemoryType` / `ConnType`

常见可直接传输的数据类型（Python 侧）：

* `bytes` / `bytearray` / `memoryview`
* NumPy 数组（CPU）
* PyTorch Tensor（CPU，CUDA 通常走 RDMA）

本质仍然遵循同一个模型：

> `Session.put/get` 会把 Python 对象搬到内部 `ConnBuffer`，再调用底层 `writeTo/readFrom`。

---

## B3. Python 快速开始（UCX：CPU payload）

### 服务端

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

# 从客户端读取 1MB 到 bytearray
client_ip = "192.168.2.248"
dst = bytearray(1024 * 1024)
sess.get_into(client_ip, dst, conn=hmc.ConnType.UCX, bias=0)
print(dst[:16])
```

### 客户端

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

## B4. NumPy（CPU）

### 发送 NumPy 数组

```python
import hmc, numpy as np

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.connect("192.168.2.244", 2025, conn=hmc.ConnType.UCX)

x = np.arange(1024, dtype=np.int32)
sess.put("192.168.2.244", x, conn=hmc.ConnType.UCX)
```

### 接收到 NumPy 数组

```python
import numpy as np

y = np.empty((1024,), dtype=np.int32)
sess.get_into("192.168.2.244", y, conn=hmc.ConnType.UCX)  # size 从 y.nbytes 推断
```

备注：

* 源数组需尽量 contiguous；封装层可能为你做拷贝。
* 目的数组必须可写且 contiguous。

---

## B5. PyTorch（CPU）

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

## B6. PyTorch CUDA（通常走 RDMA）

UCX 是否支持 GPU-direct 取决于 UCX transport/系统配置；常见策略是：

* **CUDA tensor → RDMA 后端**

### 发送 CUDA tensor（RDMA）

```python
import hmc, torch

sess = hmc.create_session(mem_type=hmc.MemoryType.DEFAULT)
sess.connect("192.168.2.244", 2025, conn=hmc.ConnType.RDMA)

t = torch.randn(1024 * 1024, device="cuda")
sess.put_torch_cuda("192.168.2.244", t, conn=hmc.ConnType.RDMA)
```

### 接收到 CUDA tensor（RDMA）

```python
recv = torch.empty_like(t)
sess.get_torch_cuda("192.168.2.244", recv, conn=hmc.ConnType.RDMA)
```

---

## B7. Python API 速查（核心）

### `create_session(...) -> Session`

创建 `IOBuffer + Communicator` 组合对象。

常用参数：

* `device_id`
* `buffer_size`
* `mem_type`
* `num_chs`

### `Session.connect(ip, port, conn)`

客户端连接。

### `Session.init_server(ip, port, conn)`

服务端监听。

### `Session.put(ip, data, bias=0, conn=ConnType.UCX, size=None)`

把 `data` 拷贝进本地 buffer，再 `writeTo()` 推送到远端。

### `Session.get(ip, size, bias=0, conn=ConnType.UCX) -> bytes`

从远端 `readFrom()` 到本地 buffer，再返回 bytes。

### `Session.get_into(ip, dest, bias=0, conn=ConnType.UCX, size=None)`

从远端读到 buffer，再拷贝到 `dest`（size 可从 dest 推断）。

### `Session.disconnect(ip, conn)`

断开连接。

### `Session.close_server()`

停止监听。

---

## 错误处理与排查要点

常见错误：

* **buffer 越界**：`bias + size > buffer_size`
* **连接 key 不匹配**：传入的 `ip` 必须与后端存储 endpoint 的 key 一致
* **后端类型不匹配**：UCX/RDMA 使用错
* **GPU-direct 限制**：取决于平台/驱动/UCX transport 配置

建议：

* 先用 CPU buffer + UCX 跑通
* 再启用 RDMA/GPU-direct（并明确验证环境）
* 若你需要严格收发时序，请使用控制信道或自己定义协议

---

© 2025 SDU spgroup Holding Limited. All rights reserved.
