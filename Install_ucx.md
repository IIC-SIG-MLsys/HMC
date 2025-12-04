````markdown
# UCX 安装教程（CPU / CUDA / ROCm）— 干净版

目标：把 UCX 安装到默认路径 **`/usr/local/ucx`**，让 CMake/运行时都能稳定找到：
- `libucp`
- `libuct`
- `libucs`

---

## 1) 安装系统依赖

### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential autoconf automake libtool pkg-config \
  git cmake \
  libnuma-dev \
  libibverbs-dev librdmacm-dev rdma-core \
  libssl-dev
````

### CentOS / RHEL / Rocky / Alma

```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
  autoconf automake libtool pkgconfig \
  git cmake \
  numactl-devel \
  libibverbs-devel librdmacm-devel rdma-core \
  openssl-devel
```

### SUSE / openSUSE

```bash
sudo zypper install -y \
  gcc gcc-c++ make autoconf automake libtool pkg-config \
  git cmake \
  libnuma-devel \
  rdma-core-devel libibverbs-devel librdmacm-devel \
  libopenssl-devel
```

---

## 2) 获取 UCX 源码

```bash
git clone https://github.com/openucx/ucx.git
cd ucx
./autogen.sh
```

---

## 3) 安装 UCX（CPU 版）

```bash
mkdir -p build && cd build

../contrib/configure-release \
  --prefix=/usr/local/ucx

make -j
sudo make install
```

---

## 4) 安装 UCX（CUDA 版）

前提：已安装 CUDA Toolkit（通常在 `/usr/local/cuda`）。

```bash
cd /path/to/ucx
mkdir -p build-cuda && cd build-cuda

../contrib/configure-release \
  --prefix=/usr/local/ucx \
  --with-cuda=/usr/local/cuda

make -j
sudo make install
```

---

## 5) 安装 UCX（ROCm 版）

前提：已安装 ROCm（通常在 `/opt/rocm`）。

```bash
cd /path/to/ucx
mkdir -p build-rocm && cd build-rocm

../contrib/configure-release \
  --prefix=/usr/local/ucx \
  --with-rocm=/opt/rocm

make -j
sudo make install
```

---

## 6) 兼容你的默认路径（/usr/local/ucx-rocm → /usr/local/ucx）

如果你已经装到了 `/usr/local/ucx-rocm`，但工程要求 `/usr/local/ucx`：

```bash
sudo ln -sfn /usr/local/ucx-rocm /usr/local/ucx
```

---

## 7) 安装完成后的必做检查

### 7.1 确认库文件存在

```bash
ls -l /usr/local/ucx/lib/libucp* /usr/local/ucx/lib/libuct* /usr/local/ucx/lib/libucs* 2>/dev/null
```

### 7.2 配置运行时动态库搜索路径（推荐永久）

```bash
echo "/usr/local/ucx/lib" | sudo tee /etc/ld.so.conf.d/ucx.conf
sudo ldconfig
```

（或临时）

```bash
export LD_LIBRARY_PATH=/usr/local/ucx/lib:$LD_LIBRARY_PATH
```

---

## 8) CMake 工程使用建议

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/usr/local/ucx
cmake --build build -j
```

---

## 9) 常用调试命令

```bash
/usr/local/ucx/bin/ucx_info -v
/usr/local/ucx/bin/ucx_info -d
ldd /usr/local/ucx/lib/libucp.so | head -n 50
```

```
::contentReference[oaicite:0]{index=0}
```
