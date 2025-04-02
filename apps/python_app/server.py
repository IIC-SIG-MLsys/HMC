import time
import hmc

def main():
    # 配置参数
    device_id = 0  # 服务端通常使用默认设备
    buffer_size = 1 * 1024 * 1024  # 1MB缓冲区
    server_rank = 0
    server_ip = "192.168.2.241"
    server_port = 2025

    # 初始化ConnBuffer
    buffer = hmc.ConnBuffer(
        device_id=device_id,
        buffer_size=buffer_size,
        mem_type=hmc.MemoryType.CPU
    )
    print(f"Server buffer allocated. Size: {buffer_size} bytes")

    # 创建通信器
    comm = hmc.Communicator(buffer)
    print("Communicator initialized")

    try:
        # 注册服务端地址信息
        comm.addNewRankAddr(server_rank, server_ip, server_port)
        print(f"Registered server address at {server_ip}:{server_port}")

        # 初始化服务端
        comm.initServer(server_ip, server_port, hmc.ConnType.RDMA)
        print("RDMA server started, waiting for connections...")

        # 主循环
        for i in range(5):
            # 等待客户端数据到达（假设客户端使用相同的buffer_size）
            print("\nWaiting for client data...")

            # 接收数据到缓冲区
            status = comm.readFrom(
                node_rank=server_rank,  # 从任意客户端读取
                ptr_bias=0,
                size=buffer_size,
                connType=hmc.ConnType.RDMA
            )

            if status != hmc.status_t.SUCCESS:
                print(f"Data receive failed with status: {status}")
                continue

            # 从缓冲区读取到CPU内存
            verify_buffer = bytearray(buffer_size)
            read_status = buffer.readToCpu(verify_buffer, buffer_size, bias=0)

            if read_status != hmc.status_t.SUCCESS:
                print(f"Buffer read failed with status: {read_status}")
                continue

            # 数据验证
            valid_data = b"Hello!" + b'\x00' * 1020
            if len(verify_buffer) >= len(valid_data):
                # 提取可打印头部
                header = verify_buffer[:6].decode('ascii', errors='replace')

                # 查找有效数据结束位置
                end = verify_buffer.find(b'\x00', 6)
                end = end if end != -1 else 10

                printable = bytes(b if 32 <= b <= 126 else 46 for b in verify_buffer[:end])

                print(f"Received {len(verify_buffer)} bytes")
                print(f"Header: {header}")
                print(f"Preview: {printable.decode('ascii')}...")

                # 校验数据有效性
                if verify_buffer.startswith(b"Hello!") or verify_buffer.startswith(b"Bye!"):
                    print("Data verification: Valid pattern detected")
                else:
                    print("Data verification: Unknown data format")
            else:
                print("Received incomplete data")

            time.sleep(5)  # 处理间隔

    except KeyboardInterrupt:
        print("\nServer shutdown requested")
    finally:
        print("Releasing resources...")
        # Python会自动释放资源，无需显式del

if __name__ == "__main__":
    main()