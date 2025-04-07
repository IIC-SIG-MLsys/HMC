import time
import hmc

def main():
    # 配置参数
    device_id = 1
    buffer_size = 1 * 1024 * 1024  # 1MB缓冲区
    server_rank = 0
    server_ip = "192.168.2.241"
    server_port = 2025

    # 初始化ConnBuffer(使用正确顺序的参数)
    buffer = hmc.ConnBuffer(
        device_id=device_id,
        buffer_size=buffer_size,
        mem_type=hmc.MemoryType.CPU
    )
    print(f"Buffer allocated. Size: {buffer_size} bytes")

    # 创建通信器（直接传递buffer实例）
    comm = hmc.Communicator(buffer)
    print("Communicator created")

    try:
        # 初始化客户端连接
        comm.addNewRankAddr(server_rank, server_ip, server_port)
        print(f"Registered server address {server_rank} {server_ip}:{server_port}")

        # 建立连接
        comm.connectTo(server_rank, hmc.ConnType.RDMA)
        print("RDMA connection established")
        time.sleep(1)  # 等待连接初始化

        # 准备发送数据（使用bytes类型自动适配buffer协议）
        data_payloads = [
            b"Hello!" + b'\x00' * 1020,  # 填充1KB数据
            b"Bye!" + b'\x00' * 1020     # 填充1KB数据
        ]

        for idx, data in enumerate(data_payloads):
            # 写入数据到缓冲区（修正参数名为bias）
            data_size = len(data)
            status = buffer.writeFromCpu(data, data_size, bias=0)  # 参数名修正

            if status != hmc.status_t.SUCCESS:
                print(f"Data {idx+1} write failed with status: {status}")
                continue

            # 发送到服务端
            comm.writeTo(
                node_rank=server_rank,
                ptr_bias=0,  # 使用缓冲区起始位置
                size=data_size,
                connType=hmc.ConnType.RDMA
            )
            print(f"Sent {data_size} bytes (Payload {idx+1})")
            time.sleep(3)  # 等待时间

    finally:
        # 自动释放资源（Python GC处理）
        print("Cleaning up resources...")

if __name__ == "__main__":
    main()