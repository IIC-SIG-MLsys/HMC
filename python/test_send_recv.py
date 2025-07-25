import hmc
import time
 
def run_receiver():
    # 创建内存和通信缓冲区
    memory = hmc.Memory(0, hmc.MemoryType.CPU)
    status, buf = memory.allocateBuffer(1024)
    assert status == hmc.status_t.SUCCESS

    # 创建 ConnBuffer 和 Communicator
    conn_buf = hmc.ConnBuffer(0, 1024, hmc.MemoryType.CPU)
    comm = hmc.Communicator(conn_buf)

    # 启动服务端
    comm.initServer("192.168.2.243", 12025)
    print("Receiver: server started, waiting for connection...")
    
    time.sleep(5)
    # # 等待客户端连接
    # while comm.checkConn("192.168.2.243") != hmc.status_t.SUCCESS:
    #     print(comm.checkConn("192.168.2.243"))
    #     time.sleep(2)

    # 接收数据
    comm.recv("192.168.2.243", 0, 1024)
    print("Receiver: data received.")

    # 读取数据到 Python
    arr = bytearray(1024)
    conn_buf.readToCpu(arr, 1024)
    print("Receiver: first 10 bytes:", list(arr[:10]))

    comm.closeServer()

def run_sender():
    # 创建内存和通信缓冲区
    memory = hmc.Memory(0, hmc.MemoryType.CPU)
    status, buf = memory.allocateBuffer(1024)
    assert status == hmc.status_t.SUCCESS

    # 填充数据
    mv = memoryview(buf)
    for i in range(1024):
        mv[i] = i % 256

    # 创建 ConnBuffer 和 Communicator
    conn_buf = hmc.ConnBuffer(0, 1024, hmc.MemoryType.CPU)
    comm = hmc.Communicator(conn_buf)

    # 连接到服务端
    time.sleep(1)  # 确保 receiver 已启动
    comm.connectTo("192.168.2.243", 12025)
    print("Sender: connected to server.")

    # 写数据到 ConnBuffer
    conn_buf.writeFromCpu(mv, 1024)

    # 发送数据
    comm.send("192.168.2.243", 0, 1024)
    print("Sender: data sent.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("send", "recv"):
        print("用法: python test_send_recv.py [send|recv]")
        exit(1)
    if sys.argv[1] == "recv":
        run_receiver()
    else:
        run_sender()