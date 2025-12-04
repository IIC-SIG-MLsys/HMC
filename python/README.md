# HMC Python SDK — User Guide (Core Usage)

This document explains the **core** Python-facing APIs of the HMC package and how to use them for data transfer with **UCX** or **RDMA**.

---

## Overview

HMC exposes a low-level C++ core via `pybind11`, then provides a small Python wrapper layer for a simpler developer experience.

You will typically use:

* **`Session`**: a high-level communication object (recommended)
* **`IOBuffer`**: a wrapper around the shared `ConnBuffer` (rarely needed directly)
* **Enums**: `Status`, `MemoryType`, `ConnType`

Supported payload types (Python-level):

* `bytes`, `bytearray`, `memoryview`
* **NumPy arrays** (CPU)
* **PyTorch tensors** (CPU and CUDA)

---

## Quick Start

### Install / Import

```python
import hmc
```

---

## Core Concepts

### 1) `ConnBuffer` (Shared Buffer)

All network operations read/write from a local buffer (`ConnBuffer`).
The wrapper class **`IOBuffer`** provides clearer “directional” naming:

* `buffer_put_*` : **external → buffer**
* `buffer_get_*` : **buffer → external**

### 2) `Communicator` (Transport)

The C++ communicator manages connections and issues transfers:

* `connectTo(ip, port, connType)` — client connects to server
* `initServer(ip, port, connType)` — start server listener
* `writeTo(ip, bias, size, connType)` — write local buffer to remote
* `readFrom(ip, bias, size, connType)` — read remote to local buffer

### 3) `Session` (Recommended)

`Session` combines a local buffer + communicator and provides **one-call** sending/receiving:

* `put(ip, data, conn=ConnType.UCX, bias=0, size=None)`
* `get(ip, size, conn=ConnType.UCX, bias=0) -> bytes`
* `get_into(ip, dest, conn=ConnType.UCX, bias=0, size=None)`

---

## Enums (Python-side)

Use these for better IDE hints and readability:

```python
hmc.Status.SUCCESS
hmc.MemoryType.CPU
hmc.ConnType.UCX
```

(Exact enum names depend on your `__init__.py`, but the idea is: Python enums wrap the C++ enum values.)

---

## Create a Session

```python
sess = hmc.create_session(
    device_id=0,
    buffer_size=128 * 1024 * 1024,
    mem_type=hmc.MemoryType.CPU,
    num_chs=1,
)
```

* **UCX mode** usually uses **CPU buffers** (`MemoryType.CPU`).
* `buffer_size` must be large enough for your maximum message size.

---

## UCX Mode (Most Common): CPU Payloads + UCX RMA

### Server

```python
import hmc

ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.init_server(ip, port, conn=hmc.ConnType.UCX)

print("UCX server ready")

# Example: receive 1MB into a bytearray
client_ip = "192.168.2.248"
dst = bytearray(1024 * 1024)
sess.get_into(client_ip, dst, conn=hmc.ConnType.UCX)
print(dst[:16])
```

### Client (send bytes)

```python
import hmc

server_ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.connect(server_ip, port, conn=hmc.ConnType.UCX)

payload = b"hello hmc"
sess.put(server_ip, payload, conn=hmc.ConnType.UCX)
```

---

## NumPy (CPU)

### Send a NumPy array

```python
import hmc
import numpy as np

server_ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.connect(server_ip, port, conn=hmc.ConnType.UCX)

x = np.arange(1024, dtype=np.int32)
sess.put(server_ip, x, conn=hmc.ConnType.UCX)
```

### Receive into a NumPy array

```python
import numpy as np

y = np.empty((1024,), dtype=np.int32)
sess.get_into(server_ip, y, conn=hmc.ConnType.UCX)   # size inferred from y.nbytes
```

Notes:

* The wrapper ensures the source is C-contiguous (copies if needed).
* Destination arrays must be writable and contiguous.

---

## PyTorch (CPU)

### Send a CPU tensor

```python
import hmc
import torch

server_ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(mem_type=hmc.MemoryType.CPU)
sess.connect(server_ip, port, conn=hmc.ConnType.UCX)

t = torch.arange(4096, dtype=torch.int32)  # CPU
sess.put(server_ip, t, conn=hmc.ConnType.UCX)
```

### Receive into a CPU tensor

```python
out = torch.empty_like(t)
sess.get_into(server_ip, out, conn=hmc.ConnType.UCX)
```

Notes:

* CPU tensors are made contiguous if needed.

---

## PyTorch CUDA (GPU Pointer Path)

HMC supports reading/writing GPU memory via device pointers (CUDA tensors).
Whether **UCX** supports GPU-direct on your platform depends on UCX/transport configuration, so the wrapper commonly defaults these calls to **RDMA**.

### Client: send a CUDA tensor (RDMA)

```python
import hmc
import torch

server_ip = "192.168.2.244"
port = 2025

sess = hmc.create_session(mem_type=hmc.MemoryType.DEFAULT)
sess.connect(server_ip, port, conn=hmc.ConnType.RDMA)

t = torch.randn(1024 * 1024, device="cuda")
sess.put_torch_cuda(server_ip, t, conn=hmc.ConnType.RDMA)
```

### Client: receive into a CUDA tensor (RDMA)

```python
recv = torch.empty_like(t)
sess.get_torch_cuda(server_ip, recv, conn=hmc.ConnType.RDMA)
```

---

## Common Parameters

### `bias` (offset)

Many calls take `bias`, which is the byte offset inside the shared buffer.

Example: write data into buffer starting at offset 4096:

```python
sess.put(server_ip, payload, bias=4096, conn=hmc.ConnType.UCX)
```

### `size`

`Session.put()` and `Session.get_into()` can infer size from the object:

* bytes-like: `len(data)`
* numpy: `array.nbytes`
* torch: `tensor.numel() * tensor.element_size()`

You can override with `size=` if needed.

---

## Error Handling

The wrapper raises Python exceptions on non-success statuses (recommended), so failures are visible immediately.

Typical failures:

* buffer overflow (`size + bias > buffer_size`)
* incompatible destination buffer (readonly / non-contiguous)
* connection missing or wrong peer key (`ip` must match what the backend uses)

---

## Minimal API Reference (Core)

### `create_session(...) -> Session`

Creates `IOBuffer + Communicator`.

### `Session.connect(ip, port, conn)`

Client-side connect.

### `Session.init_server(ip, port, conn)`

Server-side listen.

### `Session.put(ip, data, bias=0, conn=ConnType.UCX, size=None)`

Copy `data` into local buffer and write to remote.

### `Session.get(size, ip, bias=0, conn=ConnType.UCX) -> bytes`

Read remote into local buffer and return bytes.

### `Session.get_into(ip, dest, bias=0, conn=ConnType.UCX, size=None)`

Read remote into local buffer and copy into `dest`.

### `Session.disconnect(ip, conn)`

Disconnect a peer.

### `Session.close_server()`

Stop server listener.

---

## Best Practices

* Prefer **`Session`** over calling `Communicator` directly.
* Use **CPU buffers** for UCX unless you explicitly enable and validate GPU-direct.
* For NumPy / Torch CPU:
  * use contiguous arrays/tensors for best performance
* Ensure `buffer_size` ≥ max payload size (plus any offset/bias used).
