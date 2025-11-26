import sys
import time
import ctypes
import hmc


BUF_CAPACITY = 1024
PORT = 23456
IP = "192.168.2.243"  # 用你机器的 IP


def make_capsule_from_ptr(ptr: int, name: bytes = b"hmc_buffer"):
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return PyCapsule_New(ctypes.c_void_p(ptr), name, None)


def run_server():
    print("[server] creating ConnBuffer ...")
    conn_buf = hmc.ConnBuffer(
        0,                      # device_id
        BUF_CAPACITY,           # buffer_size：缓冲区容量
        hmc.MemoryType.CPU
    )

    comm = hmc.Communicator(conn_buf, num_chs=1)

    print(f"[server] initServer({IP}, {PORT}) ...")
    ret = comm.initServer(IP, PORT, hmc.ConnType.RDMA)
    print("[server] initServer ret:", ret)

    print("[server] waiting for client to connect ...")
    time.sleep(4)

    # 假设我们约定 real data size
    msg_max_size = 64  # 本例中要接收的数据不会超过 64 字节

    # 把 ConnBuffer 的指针封成 capsule，传给 recvDataFrom
    recv_capsule = make_capsule_from_ptr(conn_buf.ptr)

    print("[server] calling recvDataFrom ...")
    ret = comm.recvDataFrom(
        IP,                         # 对端 IP
        recv_capsule,               # ← 这里传 capsule，而不是裸 ptr
        BUF_CAPACITY,               # buffer 容量上限
        hmc.MemoryType.CPU,         # buffer 类型
        msg_max_size,               # flag = 实际要接收的数据大小
                                   # （如果你后面知道具体长度，也可以换成 data_size）
        hmc.ConnType.RDMA
    )
    print("[server] recvDataFrom ret:", ret)

    # 从 ConnBuffer 拷数据回 Python
    host_recv = bytearray(msg_max_size)
    conn_buf.readToCpu(host_recv, msg_max_size, 0)

    print("[server] received raw bytes:", host_recv)
    print("[server] as string:",
          host_recv.split(b"\x00", 1)[0].decode("utf-8", errors="ignore"))

    comm.closeServer()
    print("[server] done.")


def run_client():
    print("[client] creating ConnBuffer ...")
    conn_buf = hmc.ConnBuffer(
        0,
        BUF_CAPACITY,
        hmc.MemoryType.CPU
    )

    comm = hmc.Communicator(conn_buf, num_chs=1)

    print(f"[client] connectTo({IP}, {PORT}) ...")
    ret = comm.connectTo(IP, PORT, hmc.ConnType.RDMA)
    print("[client] connectTo ret:", ret)

    msg = "Hello hmc.sendDataTo / recvDataFrom!"
    data_bytes = msg.encode("utf-8")
    data_size = len(data_bytes)
    print("[client] msg len =", data_size)

    if data_size > BUF_CAPACITY:
        raise RuntimeError("message too long for BUF_CAPACITY")

    # 准备 host 侧 buffer
    host_send = bytearray(data_size)
    host_send[:] = data_bytes

    # 写入 ConnBuffer
    conn_buf.writeFromCpu(host_send, data_size, 0)

    # 同样，把 ConnBuffer 的指针封成 capsule
    send_capsule = make_capsule_from_ptr(conn_buf.ptr)

    print("[client] calling sendDataTo ...")
    ret = comm.sendDataTo(
        IP,                         # 对端 IP（server）
        send_capsule,               # ← 注意这里也传 capsule
        data_size,                  # 实际发送的大小
        hmc.MemoryType.CPU,
        hmc.ConnType.RDMA
    )
    print("[client] sendDataTo ret:", ret)

    print("[client] done.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("server", "client"):
        print("用法: python mhytest.py [server|client]")
        sys.exit(1)

    role = sys.argv[1]
    if role == "server":
        run_server()
    else:
        time.sleep(1)
        run_client()
