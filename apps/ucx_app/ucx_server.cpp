#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <thread>
#include <string.h>
#include <signal.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <vector>
#include <netdb.h>
#include <unistd.h>
#include <iomanip>
#include <fstream>
#include <cstdlib>

using namespace hmc;
using namespace std;

// 默认配置
const std::string DEFAULT_SERVER_IP = "192.168.2.251";
const std::string DEFAULT_CLIENT_IP = "192.168.2.241";
const uint16_t DEFAULT_PORT = 2024;

std::string server_ip;
std::string client_ip;
size_t buffer_size = 1 * 1024 * 1024;
const int device_id = 0;

// 用于干净关闭的信号处理
volatile bool running = true;

void signal_handler(int sig) {
    running = false;
}

// 使用函数封装环境变量读取逻辑
std::string get_env_or_default(const char* var_name, const std::string& default_val) {
    const char* val = getenv(var_name);
    return (val != nullptr) ? std::string(val) : default_val;
}

// 存储网络接口信息的结构
struct NetworkInterface {
    std::string name;
    std::string ip;
};

// 获取所有可用网络接口及其IP的函数
std::vector<NetworkInterface> getNetworkInterfaces() {
    std::vector<NetworkInterface> interfaces;
    struct ifaddrs *ifaddr, *ifa;
    int family, s;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        return interfaces;
    }

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;

        if (family == AF_INET) {
            s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                           host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
            if (s != 0) {
                printf("getnameinfo() failed: %s\n", gai_strerror(s));
                continue;
            }

            NetworkInterface interface;
            interface.name = ifa->ifa_name;
            interface.ip = host;
            interfaces.push_back(interface);
        }
    }

    freeifaddrs(ifaddr);
    return interfaces;
}

// 写入端口发现文件
void writePortDiscoveryFile(uint16_t port) {
    std::ofstream port_file("/tmp/ucx_server_port.txt");
    if (port_file.is_open()) {
        port_file << port;
        port_file.close();
        std::cout << "服务器端口 " << port << " 已写入文件 /tmp/ucx_server_port.txt" << std::endl;
    } else {
        std::cerr << "无法写入端口发现文件" << std::endl;
    }
}

// 辅助函数：以十六进制格式打印数据
void printDataHex(const char* data, size_t length, size_t max_bytes = 64) {
    std::cout << "数据的十六进制表示 (最多 " << max_bytes << " 字节): " << std::endl;
    
    size_t print_length = (length > max_bytes) ? max_bytes : length;
    
    std::cout << "    ";
    for (size_t i = 0; i < print_length; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(static_cast<unsigned char>(data[i])) << " ";
        
        if ((i + 1) % 16 == 0 && i < print_length - 1) {
            std::cout << std::endl << "    ";
        }
    }
    
    if (length > max_bytes) {
        std::cout << "... (数据太长，已截断)";
    }
    
    std::cout << std::dec << std::endl;
}

int main(int argc, char* argv[]) {
    FLAGS_colorlogtostderr = true;
    FLAGS_alsologtostderr = true;

    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 读取环境变量
    server_ip = get_env_or_default("SERVER_IP", DEFAULT_SERVER_IP);
    client_ip = get_env_or_default("CLIENT_IP", DEFAULT_CLIENT_IP);

    std::cout << "UCX服务器配置:" << std::endl;
    std::cout << "  服务器IP: " << server_ip << std::endl;
    std::cout << "  客户端IP: " << client_ip << std::endl;

    // 设置UCX环境变量
    setenv("UCX_TLS", "tcp", 1);
    setenv("UCX_SOCKADDR_TLS_PRIORITY", "tcp", 1);
    setenv("UCX_TCP_CM_REUSEADDR", "y", 1);
    setenv("UCX_WARN_UNUSED_ENV_VARS", "n", 1);
    setenv("UCX_RNDV_THRESH", "8192", 1);

    // 获取网络接口信息
    auto interfaces = getNetworkInterfaces();
    std::cout << "\n可用的网络接口:" << std::endl;
    for (const auto& iface : interfaces) {
        std::cout << "  " << iface.name << ": " << iface.ip << std::endl;
    }

    // 创建连接缓冲区
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::CPU);
    std::cout << "\n分配缓冲区成功: " << buffer->ptr << std::endl;
    std::cout << "缓冲区大小: " << buffer_size << " 字节" << std::endl;

    // 初始化缓冲区内容为零
    char *zeros = new char[buffer_size]();
    buffer->writeFromCpu(zeros, buffer_size, 0);
    delete[] zeros;
    std::cout << "缓冲区已初始化为零" << std::endl;

    // 创建通信器
    Communicator *comm = new Communicator(buffer);
    std::cout << "创建通信器成功" << std::endl;

    // 自动选择合适的绑定地址
    std::string bind_ip = "0.0.0.0";  // 绑定到所有接口
    
    // 检查SERVER_IP是否在本机网络接口上
    bool server_ip_available = false;
    for (const auto& iface : interfaces) {
        if (iface.ip == server_ip) {
            server_ip_available = true;
            bind_ip = server_ip;  // 如果SERVER_IP在本机，直接使用
            break;
        }
    }
    
    if (!server_ip_available) {
        std::cout << "注意: SERVER_IP " << server_ip << " 不在本机网络接口上" << std::endl;
        std::cout << "      服务器将绑定到所有接口 (0.0.0.0)" << std::endl;
    }

    // 使用UCX初始化服务器
    status_t status = comm->initServer(bind_ip, DEFAULT_PORT, ConnType::UCX);
    if (status != status_t::SUCCESS) {
        std::cerr << "初始化服务器失败" << std::endl;
        delete comm;
        buffer.reset();
        return 1;
    }

    // 写入端口发现文件
    writePortDiscoveryFile(DEFAULT_PORT);

    std::cout << "\n服务器正在监听 (绑定地址: " << bind_ip << ":" << DEFAULT_PORT << ")" << std::endl;
    if (bind_ip == "0.0.0.0") {
        std::cout << "客户端可通过以下任一IP连接:" << std::endl;
        for (const auto& iface : interfaces) {
            if (iface.ip != "127.0.0.1") {  // 跳过回环地址
                std::cout << "  " << iface.ip << ":" << DEFAULT_PORT << std::endl;
            }
        }
    }
    std::cout << "端口已写入: /tmp/ucx_server_port.txt" << std::endl;
    std::cout << "按 Ctrl+C 退出" << std::endl;

    // 主服务器循环
    char host_data[1024];
    int msg_count = 0;
    const int expected_messages = 2;

    std::cout << "\n================ 等待数据接收 ================\n" << std::endl;

    // 记录上次成功读取到的数据的哈希
    size_t last_data_hash = 0;

    while (running) {
        // 直接从缓冲区读取
        memset(host_data, 0, sizeof(host_data));
        buffer->readToCpu(host_data, sizeof(host_data), 0);

        // 计算简单哈希以检测内容变化
        size_t current_hash = 0;
        bool has_data = false;

        for (size_t i = 0; i < sizeof(host_data); i++) {
            if (host_data[i] != '\0') {
                has_data = true;
                current_hash = current_hash * 31 + static_cast<unsigned char>(host_data[i]);
            }
        }

        // 检查是否有新数据
        if (has_data && current_hash != last_data_hash) {
            size_t data_len = strlen(host_data);

            std::cout << "\n---------------------------------------------" << std::endl;
            std::cout << "【内存互传确认】成功接收客户端数据!" << std::endl;
            std::cout << "数据长度: " << data_len << " 字节" << std::endl;
            std::cout << "ASCII内容: \"" << host_data << "\"" << std::endl;

            // 打印数据的十六进制表示
            printDataHex(host_data, data_len);

            // 打印缓冲区地址
            std::cout << "数据接收地址: " << static_cast<void*>(buffer->ptr) << std::endl;
            std::cout << "---------------------------------------------\n" << std::endl;

            last_data_hash = current_hash;
            msg_count++;

            // 处理后清空缓冲区
            memset(host_data, 0, sizeof(host_data));
            buffer->writeFromCpu(host_data, sizeof(host_data), 0);

            // 接收指定数量消息后退出
            if (msg_count >= expected_messages) {
                std::cout << "\n===============================================" << std::endl;
                std::cout << "服务器已成功接收到 " << msg_count << " 条消息" << std::endl;
                std::cout << "UCX通信功能验证完成!" << std::endl;
                std::cout << "===============================================\n" << std::endl;
                std::cout << "等待2秒后退出..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                running = false;
            }
        }

        // 短暂休眠以避免忙等待
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 清理资源
    std::cout << "\n服务器正在关闭..." << std::endl;
    delete comm;
    buffer.reset();
    std::cout << "服务器关闭完成" << std::endl;
    
    return 0;
}