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
#include <iomanip> // For std::setw and std::setfill
#include <fstream> // For std::ofstream

// 默认服务器配置
#define DEFAULT_SERVER_PORT 2024

using namespace hmc;

// 用于干净关闭的信号处理
volatile bool running = true;

void signal_handler(int sig) {
    running = false;
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

    // 遍历接口链表
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;

        // 只过滤IPv4地址
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

// Helper function to write the port to discovery file
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
    
    // 限制输出长度
    size_t print_length = (length > max_bytes) ? max_bytes : length;
    
    std::cout << "    ";
    for (size_t i = 0; i < print_length; i++) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') 
                  << static_cast<int>(static_cast<unsigned char>(data[i])) << " ";
        
        // 每16个字节换行
        if ((i + 1) % 16 == 0 && i < print_length - 1) {
            std::cout << std::endl << "    ";
        }
    }
    
    if (length > max_bytes) {
        std::cout << "... (数据太长，已截断)";
    }
    
    std::cout << std::dec << std::endl;  // 恢复十进制输出
}

// 打印使用信息的函数
void print_usage(char* program_name) {
    std::cout << "使用方法: " << program_name << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -i, --interface <n>   绑定到特定网络接口" << std::endl;
    std::cout << "  -a, --address <ip>    绑定到特定IP地址" << std::endl;
    std::cout << "  -p, --port <port>     绑定到特定端口 (默认: " << DEFAULT_SERVER_PORT << ")" << std::endl;
    std::cout << "  -h, --help            显示此帮助信息" << std::endl;
    std::cout << "  -d, --debug           启用更详细的调试信息" << std::endl;
}

// 获取首选服务器IP地址
std::string getPreferredServerIP(const std::vector<NetworkInterface>& interfaces) {
    // 优先级: 192.168.2.x > 10.x.x.x > 其他
    
    // 首先检查192.168.2.x网段
    for (const auto& iface : interfaces) {
        if (iface.ip.find("192.168.2.") == 0) {
            return iface.ip;
        }
    }
    
    // 其次检查10.x.x.x网段
    for (const auto& iface : interfaces) {
        if (iface.ip.find("10.") == 0) {
            return iface.ip;
        }
    }
    
    // 最后返回第一个非回环地址
    for (const auto& iface : interfaces) {
        if (iface.ip != "127.0.0.1") {
            return iface.ip;
        }
    }
    
    // 默认回退到回环地址
    return "127.0.0.1";
}

int main(int argc, char* argv[]) {
    FLAGS_colorlogtostderr = true;
    FLAGS_alsologtostderr = true;
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 设置UCX环境变量
    setenv("UCX_TLS", "tcp", 1);
    setenv("UCX_SOCKADDR_TLS_PRIORITY", "tcp", 1);
    setenv("UCX_TCP_CM_REUSEADDR", "y", 1);
    setenv("UCX_WARN_UNUSED_ENV_VARS", "n", 1);  // 抑制未使用环境变量的警告
    setenv("UCX_RNDV_THRESH", "8192", 1);        // 设置rendezvous阈值
    
    // 调试模式标志
    bool debug_mode = false;
    
    int device_id = 0; // 内存分配的设备ID
    size_t buffer_size = 1 * 1024 * 1024;
    
    // 默认绑定参数
    std::string bind_ip = "0.0.0.0"; // 默认监听所有接口
    uint16_t bind_port = DEFAULT_SERVER_PORT; // 使用RDMA相同的固定端口
    std::string interface_name = "";
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i" || arg == "--interface") {
            if (i + 1 < argc) {
                interface_name = argv[++i];
            } else {
                std::cerr << "--interface 需要一个名称" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-a" || arg == "--address") {
            if (i + 1 < argc) {
                bind_ip = argv[++i];
            } else {
                std::cerr << "--address 需要一个IP地址" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                bind_port = std::stoi(argv[++i]);
            } else {
                std::cerr << "--port 需要一个端口号" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-d" || arg == "--debug") {
            debug_mode = true;
            std::cout << "调试模式已启用" << std::endl;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "未知参数: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // 如果提供了接口名称，查找其IP地址
    if (!interface_name.empty()) {
        auto interfaces = getNetworkInterfaces();
        bool found = false;
        
        for (const auto& iface : interfaces) {
            if (iface.name == interface_name) {
                bind_ip = iface.ip;
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cerr << "接口 " << interface_name << " 未找到或没有IPv4地址" << std::endl;
            return 1;
        }
    }

    // 获取网络接口信息
    auto interfaces = getNetworkInterfaces();
    std::cout << "可用的网络接口:" << std::endl;
    for (const auto& iface : interfaces) {
        std::cout << "  " << iface.name << ": " << iface.ip << std::endl;
    }
    
    // 获取首选的IP地址用于客户端连接
    std::string preferred_ip = getPreferredServerIP(interfaces);

    // 创建连接缓冲区 - 使用CPU内存类型以确保兼容性
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::CPU);
    std::cout << "分配缓冲区成功: " << buffer->ptr << std::endl;
    std::cout << "缓冲区大小: " << buffer_size << " 字节" << std::endl;
    
    // 初始化缓冲区内容为零
    char *zeros = new char[buffer_size]();  // 创建全零的缓冲区
    buffer->writeFromCpu(zeros, buffer_size, 0);
    delete[] zeros;
    std::cout << "缓冲区已初始化为零" << std::endl;

    // 创建通信器
    Communicator *comm = new Communicator(buffer);
    std::cout << "创建通信器成功" << std::endl;

    // 使用UCX初始化服务器
    status_t status = comm->initServer(bind_ip, bind_port, ConnType::UCX);
    if (status != status_t::SUCCESS) {
        std::cerr << "初始化服务器失败" << std::endl;
        delete comm;
        buffer.reset();
        return 1;
    }
    
    // 总是写入端口发现文件，无论绑定地址是什么
    writePortDiscoveryFile(bind_port);
    
    // 注册服务器地址和预期的客户端地址 - 使用首选IP
    comm->addNewRankAddr(0, preferred_ip, bind_port);  // 服务器自身地址（rank 0）
    
    // 为客户端预添加一些可能的地址，增加连接成功率
    uint32_t client_rank = 1;  // 客户端rank
    
    // 添加可能的客户端地址 - 尝试常见的IP
    for (const auto& iface : interfaces) {
        if (iface.name != "lo") { // 跳过回环接口
            comm->addNewRankAddr(client_rank, iface.ip, DEFAULT_SERVER_PORT - 1); // 使用2023端口
        }
    }
    
    // 添加其他可能的客户端地址
    comm->addNewRankAddr(client_rank, "127.0.0.1", DEFAULT_SERVER_PORT - 1);
    
    std::cout << "服务器正在监听 (地址: " << bind_ip << ":" << bind_port << ")" << std::endl;
    std::cout << "端口已写入: /tmp/ucx_server_port.txt" << std::endl;
    std::cout << "按 Ctrl+C 退出" << std::endl;

    // 打印客户端的连接命令
    if (bind_ip == "0.0.0.0") {
        std::cout << "客户端连接命令:" << std::endl;
        std::cout << "  ./ucx_client                  # 自动发现服务器" << std::endl;
        std::cout << "或者明确指定服务器:" << std::endl;
        std::cout << "  ./ucx_client --server " << preferred_ip << " --port " << bind_port << std::endl;
    } else {
        std::cout << "客户端连接命令:" << std::endl;
        std::cout << "  ./ucx_client --server " << bind_ip << " --port " << bind_port << std::endl;
    }

    // 主服务器循环
    char host_data[1024];
    int msg_count = 0;
    
    std::cout << "\n================ 等待数据接收 ================\n" << std::endl;
    
    // 记录上次成功读取到的数据的哈希
    size_t last_data_hash = 0;
    
    // 循环检查缓冲区
    int check_counter = 0;
    
    while (running) {
        // 直接从缓冲区读取，无需依赖连接状态
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
        
        // 检查是否有新数据（非空且哈希不同）
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
            
            // 接收2条消息后退出
            if (msg_count >= 2) {
                std::cout << "\n===============================================" << std::endl;
                std::cout << "服务器已成功接收到 " << msg_count << " 条消息" << std::endl;
                std::cout << "内存互传功能验证完成!" << std::endl;
                std::cout << "===============================================\n" << std::endl;
                std::cout << "等待2秒后退出..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(2));
                running = false;
            }
        } else if (debug_mode && (++check_counter % 20 == 0)) {
            // 在调试模式下，定期输出缓冲区状态
            std::cout << "等待数据中..." << std::endl;
            if (has_data) {
                std::cout << "检测到数据，但与上次相同" << std::endl;
            }
        }
        
        // 短暂休眠以避免忙等待
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // 清理资源
    std::cout << "服务器正在关闭..." << std::endl;
    delete comm;
    buffer.reset();
    std::cout << "服务器关闭完成" << std::endl;
    return 0;
}