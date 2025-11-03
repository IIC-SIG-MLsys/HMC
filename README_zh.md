# HMC 用户使用手册

**异构内存通信框架（Heterogeneous Memories Communication）**
© 2025 SDU spgroup Holding Limited

---

## 🧩 概述

**HMC (Heterogeneous Memories Communication)** 是一个为异构计算环境设计的统一通信与内存管理框架。
支持多种计算设备，包括 **CPU、GPU、MLU、NPU、Ascend、Moore GPU** 等。

它提供：

* ✅ 面向多种设备类型的统一 **内存管理抽象层**
* ✅ 高性能 **RDMA / UCX 通信机制**
* ✅ 内置 **控制信道（基于 TCP）** 用于同步与信号交互
* ✅ 支持 **GPU 直连传输（UHM）** 与多模式性能测试

---

## ⚙️ 编译与安装

### 环境依赖

#### 系统依赖

* **C++14** 或更高版本的编译器
* **CMake ≥ 3.18**
* **Glog**（日志库）

  ```bash
  sudo apt-get install libgoogle-glog-dev
  ```
* **GTest**（可选，用于单元测试）

  ```bash
  sudo apt-get install libgtest-dev
  ```

#### 硬件依赖

根据不同设备安装对应驱动和 SDK：

| 平台               | 依赖库         |
| ----------------- | ------------- |
| NVIDIA GPU        | CUDA Toolkit  |
| AMD/Hygon GPU     | ROCm          |
| Cambricon MLU     | CNRT / MLU-OP |
| Moore GPU         | MUSA Runtime  |

---

### 源码构建

```bash
# 克隆项目
git clone https://github.com/IIC-SIG-MLsys/HMC.git
cd HMC

# 创建构建目录
mkdir build && cd build

# 生成 Makefile
cmake ..

# 编译
make -j
```

#### 可选 CMake 选项

| 参数                      | 说明                        |
| ----------------------- | ------------------------- |
| `-DBUILD_STATIC_LIB=ON` | 构建静态库（libhmc.a）           |
| `-DBUILD_PYTHON_MOD=ON` | 构建 Python 模块（通过 PyBind11） |

---

### 构建 Python 包（可选）

HMC 提供 Python 接口，基于 **PyBind11** 封装。

```bash
# 初始化子模块
git submodule update --init --recursive

# 重新构建并启用 Python 模块
cmake .. -DBUILD_PYTHON_MOD=ON
make -j

# 生成 wheel 包
python -m build

# 安装
pip install dist/hmc-*.whl
```

---

## 🚀 快速上手

### 示例 1 — 基本内存操作

```cpp
#include <hmc.h>
using namespace hmc;

int main() {
    // 创建 GPU 内存对象
    Memory gpu_mem(0, MemoryType::NVIDIA_GPU);
    void* gpu_ptr = nullptr;

    // 分配 1MB GPU 内存
    gpu_mem.allocateBuffer(&gpu_ptr, 1024 * 1024);

    // 从 CPU 向 GPU 拷贝数据
    std::vector<char> host_data(1024 * 1024, 'A');
    gpu_mem.copyHostToDevice(gpu_ptr, host_data.data(), host_data.size());

    // 释放内存
    gpu_mem.freeBuffer(gpu_ptr);
}
```

---

### 示例 2 — RDMA 通信

```cpp
#include <hmc.h>
using namespace hmc;

auto buffer = std::make_shared<ConnBuffer>(0, 64 * 1024 * 1024);
Communicator comm(buffer);

std::string server_ip = "192.168.2.100";

// 客户端连接
comm.connectTo(server_ip, 2025, ConnType::RDMA);
comm.writeTo(server_ip, 0, 4096);
comm.disConnect(server_ip, ConnType::RDMA);

// 服务端监听
comm.initServer(server_ip, 2025, ConnType::RDMA);
comm.closeServer();
```

---

### 示例 3 — 控制通道

```cpp
#include <hmc.h>
using namespace hmc;

CtrlSocketManager& ctrl = CtrlSocketManager::instance();

// 服务端启动
ctrl.startServer("0.0.0.0", 5555);

// 客户端连接
int sock_fd = ctrl.getCtrlSockFd("192.168.2.100", 5555);
ctrl.sendCtrlInt("192.168.2.100", 42);

// 接收控制消息
int value;
ctrl.recvCtrlInt("192.168.2.100", value);
printf("Received control value: %d\n", value);

// 关闭连接
ctrl.closeConnection("192.168.2.100");
```

---

## 🧠 接口说明

---

### 🧱 Memory 类 — 内存管理

统一的内存分配与拷贝接口，支持多种加速器。

| 方法                                                             | 功能描述      |
| -------------------------------------------------------------- | --------- |
| `allocateBuffer(void** addr, size_t size)`                     | 分配指定大小的内存 |
| `freeBuffer(void* addr)`                                       | 释放内存      |
| `copyHostToDevice(void* dest, const void* src, size_t size)`   | 从主机拷贝到设备  |
| `copyDeviceToHost(void* dest, const void* src, size_t size)`   | 从设备拷贝到主机  |
| `copyDeviceToDevice(void* dest, const void* src, size_t size)` | 同设备拷贝     |

支持的内存类型：

```cpp
enum class MemoryType {
  DEFAULT,
  CPU,
  NVIDIA_GPU,
  AMD_GPU,
  CAMBRICON_MLU,
  MOORE_GPU
};
```

---

### 🪣 ConnBuffer 类 — 通信缓冲区

| 方法                                                  | 说明            |
| --------------------------------------------------- | ------------- |
| `writeFromCpu(void* src, size_t size, size_t bias)` | 将 CPU 数据写入缓冲区 |
| `readToCpu(void* dest, size_t size, size_t bias)`   | 从缓冲区读取到 CPU   |
| `writeFromGpu(void* src, size_t size, size_t bias)` | 将 GPU 数据写入缓冲区 |
| `readToGpu(void* dest, size_t size, size_t bias)`   | 从缓冲区读取到 GPU   |

---

### 🌐 Communicator 类 — 通信管理器

| 方法                                                  | 功能          |
| --------------------------------------------------- | ----------- |
| `initServer(ip, port, type)`                        | 启动服务端       |
| `connectTo(ip, port, type)`                         | 连接远端        |
| `writeTo(ip, offset, size, type)`                   | 执行 RDMA 写操作 |
| `readFrom(ip, offset, size, type)`                  | 执行 RDMA 读操作 |
| `sendDataTo(ip, buf, size, buf_type, type)`         | 发送大数据块      |
| `recvDataFrom(ip, buf, size, buf_type, flag, type)` | 接收大数据块      |
| `closeServer()`                                     | 关闭服务        |
| `disConnect(ip, type)`                              | 断开连接        |

---

### 🛰️ CtrlSocketManager 类 — 控制信道

用于发送同步信号或元数据，底层基于 TCP。

| 方法                           | 功能          |
| ---------------------------- | ----------- |
| `startServer(bind_ip, port)` | 启动 TCP 控制服务 |
| `getCtrlSockFd(ip, port)`    | 建立客户端连接     |
| `sendCtrlInt(ip, value)`     | 发送整型控制消息    |
| `recvCtrlInt(ip, &value)`    | 接收整型控制消息    |
| `sendCtrlStruct(ip, obj)`    | 发送结构体       |
| `recvCtrlStruct(ip, obj)`    | 接收结构体       |
| `closeConnection(ip)`        | 关闭单连接       |
| `closeAll()`                 | 关闭所有连接      |

---

### 🔖 状态定义

所有接口返回值均为 `status_t`：

| 枚举值              | 含义      |
| ---------------- | ------- |
| `SUCCESS`        | 操作成功    |
| `ERROR`          | 操作失败    |
| `UNSUPPORT`      | 当前平台不支持 |
| `INVALID_CONFIG` | 配置错误    |
| `TIMEOUT`        | 超时      |
| `NOT_FOUND`      | 未找到目标   |

---

## 🧪 性能测试示例

```bash
# RDMA CPU 模式
./build/apps/uhm_app/uhm_server --mode uhm
./build/apps/uhm_app/uhm_client --mode uhm
```

支持模式说明：

| 模式         | 功能描述            |
| ---------- | --------------- |
| `uhm`      | GPU 直连传输（UHM）   |
| `rdma_cpu` | 纯 CPU RDMA 模式   |
| `g2h2g`    | GPU→Host→GPU 模式 |
| `serial`   | 顺序分段传输          |

---

## 📚 总结

HMC 为异构计算平台提供了统一、模块化的编程接口，
简化了开发者在多设备环境下的内存与通信管理。

* ✅ 简洁一致的 C++ 接口
* ✅ 支持 GPU/NPU/MLU 的高性能 RDMA 通信
* ✅ 可选 Python 封装，方便快速原型开发
* ✅ 兼容主流异构硬件平台

---

```
© 2025 SDU spgroup Holding Limited  
保留所有权利。
```