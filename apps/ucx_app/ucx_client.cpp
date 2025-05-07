#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <thread>
#include <string.h>
#include <signal.h>
#include <fstream>
#include <iomanip> // For std::setw and std::setfill
#include <netdb.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

// 默认服务器配置
#define DEFAULT_SERVER_IP "192.168.2.241" 
#define DEFAULT_SERVER_PORT 2024
#define CONNECTION_TIMEOUT_MS 500

using namespace hmc;

// 用于干净关闭的信号处理
volatile bool running = true;

void signal_handler(int sig) {
    running = false;
}

// 尝试连接到指定IP的服务器
bool tryConnectToServer(Communicator* comm, const std::string& test_ip, uint16_t port) {
    std::cout << "尝试连接到: " << test_ip << ":" << port << std::endl;
    
    // 测试TCP连接是否可用 - 快速检查
    struct sockaddr_in addr;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return false;
    
    // 设置超时
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = CONNECTION_TIMEOUT_MS * 1000; // 转换为微秒
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
    
    // 尝试TCP连接
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, test_ip.c_str(), &addr.sin_addr) <= 0) {
        close(sockfd);
        return false;
    }
    
    // 连接失败直接返回
    if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        close(sockfd);
        return false;
    }
    
    // TCP连接成功，可能有服务器
    close(sockfd);
    
    // 添加新的服务器地址映射并尝试UCX连接
    comm->addNewRankAddr(0, test_ip, port);
    comm->addNewRankAddr(1, "0.0.0.0", 2024);  // 客户端地址
    
    // 尝试实际的UCX连接
    status_t conn_status = comm->connectTo(0, ConnType::UCX);
    if (conn_status == status_t::SUCCESS) {
        std::cout << "UCX连接成功: " << test_ip << ":" << port << std::endl;
        return true;
    }
    
    // 连接失败，删除映射
    comm->delRankAddr(0);
    return false;
}

// 尝试自动发现服务器
bool discoverServer(Communicator* comm, uint16_t port = DEFAULT_SERVER_PORT) {
    std::cout << "正在自动发现服务器..." << std::endl;
    
    // 首先尝试默认服务器
    if (tryConnectToServer(comm, DEFAULT_SERVER_IP, port)) {
        return true;
    }
    
    // 尝试192.168.2.x网段
    for (int i = 240; i <= 252; i++) {
        std::string test_ip = "192.168.2." + std::to_string(i);
        if (test_ip == DEFAULT_SERVER_IP) continue; // 跳过已测试的IP
        
        if (tryConnectToServer(comm, test_ip, port)) {
            return true;
        }
    }
    
    // 尝试本地回环
    if (tryConnectToServer(comm, "127.0.0.1", port)) {
        return true;
    }
    
    // 尝试其他常见网段
    for (int i = 230; i <= 250; i++) {
        std::string test_ip = "10.102.0." + std::to_string(i);
        if (tryConnectToServer(comm, test_ip, port)) {
            return true;
        }
    }
    
    return false;
}

void print_usage(char* program_name) {
    std::cout << "使用方法: " << program_name << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -s, --server <ip>   服务器IP地址 (默认: 自动发现)" << std::endl;
    std::cout << "  -p, --port <port>   服务器端口 (默认: " << DEFAULT_SERVER_PORT << ")" << std::endl;
    std::cout << "  -h, --help          显示此帮助信息" << std::endl;
    std::cout << "  -v, --verbose       启用详细日志输出" << std::endl;
}

// 辅助函数：以十六进制格式打印数据
void printHex(const char* data, size_t length, size_t max_bytes = 64) {
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
    setenv("UCX_WARN_UNUSED_ENV_VARS", "n", 1); // 抑制未使用环境变量的警告
    setenv("UCX_RNDV_THRESH", "8192", 1);      // 设置rendezvous阈值

    // 默认连接参数
    std::string server_ip = "";  // 空字符串表示自动发现
    uint16_t server_port = DEFAULT_SERVER_PORT;
    bool verbose = false;
    bool auto_discovery = true;  // 默认启用自动发现
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-s" || arg == "--server") {
            if (i + 1 < argc) {
                server_ip = argv[++i];
                auto_discovery = false;  // 禁用自动发现
            } else {
                std::cerr << "--server 需要一个IP地址" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                server_port = std::stoi(argv[++i]);
            } else {
                std::cerr << "--port 需要一个端口号" << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "未知参数: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    int device_id = 0; // 内存分配的设备ID
    size_t buffer_size = 1 * 1024 * 1024;

    // 创建连接缓冲区 - 使用CPU内存类型以确保兼容性
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::CPU);
    std::cout << "分配缓冲区成功: " << buffer->ptr << std::endl;
    std::cout << "缓冲区大小: " << buffer_size << " 字节" << std::endl;

    // 创建通信器
    Communicator *comm = new Communicator(buffer);
    std::cout << "创建通信器成功" << std::endl;
    
    bool connected = false;
    
    if (auto_discovery) {
        // 自动发现服务器
        if (discoverServer(comm, server_port)) {
            connected = true;
        } else {
            std::cerr << "自动发现服务器失败" << std::endl;
            std::cerr << "请使用 --server 参数指定服务器IP" << std::endl;
            delete comm;
            buffer.reset();
            return 1;
        }
    } else {
        // 使用指定的服务器IP和端口
        std::cout << "尝试连接到指定服务器: " << server_ip << ":" << server_port << std::endl;
        
        comm->addNewRankAddr(0, server_ip, server_port);
        comm->addNewRankAddr(1, "0.0.0.0", 2024);
        
        int max_retries = 5;
        for (int retry = 0; retry < max_retries; retry++) {
            std::cout << "连接尝试 " << (retry + 1) << "/" << max_retries << std::endl;
            
            status_t conn_status = comm->connectTo(0, ConnType::UCX);
            if (conn_status == status_t::SUCCESS) {
                std::cout << "成功连接到服务器 " << server_ip << ":" << server_port << std::endl;
                connected = true;
                break;
            }
            
            std::cout << "连接尝试失败，2秒后重试..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
    }
    
    if (!connected) {
        std::cerr << "连接服务器失败" << std::endl;
        delete comm;
        buffer.reset();
        return 1;
    }

    // 等待连接稳定
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // 准备测试数据
    const char *data1 = "Hello via UCX!";
    const char *data2 = "Bye via UCX!";
    size_t data_size1 = strlen(data1) + 1;
    size_t data_size2 = strlen(data2) + 1;
    
    // 通信循环
    int msg_count = 0;
    int max_send_attempts = 5;
    
    while (running && msg_count < 2) {
        if (msg_count == 0) {
            // 第一条消息
            buffer->writeFromCpu((void*)data1, data_size1, 0);
            std::cout << "数据复制到缓冲区: \"" << data1 << "\"" << std::endl;
            
            if (verbose) {
                printHex(data1, data_size1);
            }
            
            bool send_success = false;
            for (int attempt = 0; attempt < max_send_attempts; attempt++) {
                std::cout << "尝试发送消息1 (尝试 " << (attempt + 1) << "/" << max_send_attempts << ")" << std::endl;
                status_t send_status = comm->writeTo(0, 0, data_size1, ConnType::UCX);
                if (send_status == status_t::SUCCESS) {
                    std::cout << "【内存互传确认】客户端成功发送: \"" << data1 << "\"" << std::endl;
                    send_success = true;
                    msg_count++;
                    break;
                } else {
                    std::cerr << "发送第一条数据失败，1秒后重试" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            
            if (!send_success) {
                std::cerr << "在所有尝试后发送第一条消息失败" << std::endl;
                break;
            }
            
            // 等待一段时间让服务器处理第一条消息
            std::this_thread::sleep_for(std::chrono::seconds(2));
        } else if (msg_count == 1) {
            // 第二条消息
            buffer->writeFromCpu((void*)data2, data_size2, 0);
            std::cout << "数据复制到缓冲区: \"" << data2 << "\"" << std::endl;
            
            if (verbose) {
                printHex(data2, data_size2);
            }
            
            bool send_success = false;
            for (int attempt = 0; attempt < max_send_attempts; attempt++) {
                std::cout << "尝试发送消息2 (尝试 " << (attempt + 1) << "/" << max_send_attempts << ")" << std::endl;
                status_t send_status = comm->writeTo(0, 0, data_size2, ConnType::UCX);
                if (send_status == status_t::SUCCESS) {
                    std::cout << "【内存互传确认】客户端成功发送: \"" << data2 << "\"" << std::endl;
                    send_success = true;
                    msg_count++;
                    break;
                } else {
                    std::cerr << "发送第二条数据失败，1秒后重试" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
            }
            
            if (!send_success) {
                std::cerr << "在所有尝试后发送第二条消息失败" << std::endl;
                break;
            }
        }
        
        // 短暂休眠
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    // 等待服务器处理最后的消息
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 优雅关闭
    std::cout << "客户端断开连接..." << std::endl;
    comm->disConnect(0, ConnType::UCX);
    delete comm;
    buffer.reset();
    std::cout << "客户端关闭完成" << std::endl;
    return 0;
}