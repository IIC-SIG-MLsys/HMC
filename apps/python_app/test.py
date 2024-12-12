import time
import hddt

def main():
    # 设置客户端和服务端的 IP 地址
    client_ip = "192.168.2.251"
    server_ip = "0.0.0.0"

    # 创建内存操作对象
    mem_ops = hddt.Memory(1)

    # 创建通信实例，设置为客户端，使用默认的通信类型
    comm = hddt.CreateCommunicator(mem_ops, hddt.CommunicatorType.DEFAULT, is_client=True, client_ip=client_ip)

    # 启动通信
    sret = comm.Start()
    if sret != hddt.status_t.SUCCESS:
        print("Failed to start communication!")
        return

    # 第一次数据
    data = b"HelloHelloHelloHelloHelloHelloHelloHello!"

    # 分配发送缓冲区并将数据从主机复制到设备
    print("allocate input buffer")
    send = mem_ops.allocate_buffer(1024)
    mem_ops.copy_host_to_device(send, data)

    # 从设备复制数据回主机
    host_data = bytearray(len(data))
    mem_ops.copy_device_to_host(host_data, send)
    print(f"Client Write Data: {host_data.decode()}")

    # 等待两秒钟，确保客户端和服务端初始化完成
    time.sleep(2)

    # 发送数据到服务端
    comm.Send(send, len(data))

    # 第二次数据
    data2 = b"ByeByeByeByeByeByeByeByeByeBye!"
    mem_ops.copy_host_to_device(send, data2)

    # 从设备复制数据回主机
    host_data2 = bytearray(len(data2))
    mem_ops.copy_device_to_host(host_data2, send)
    print(f"Client Write Data: {host_data2.decode()}")

    # 发送第二段数据
    comm.Send(send, len(data2))

    # 等待两秒钟，确保数据发送完毕
    time.sleep(2)

    # 关闭通信连接
    comm.Close()
    time.sleep(2)

    # 清理资源
    del comm

if __name__ == "__main__":
    main()
