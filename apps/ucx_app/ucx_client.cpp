#include <chrono>
#include <glog/logging.h>
#include <hmc.h>
#include <iostream>
#include <thread>
#include <string.h>
#include <signal.h>
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

int main(int argc, char* argv[]) {
    FLAGS_colorlogtostderr = true;
    FLAGS_alsologtostderr = true;

    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // 读取环境变量
    server_ip = get_env_or_default("SERVER_IP", DEFAULT_SERVER_IP);
    client_ip = get_env_or_default("CLIENT_IP", DEFAULT_CLIENT_IP);

    std::cout << "UCX客户端配置:" << std::endl;
    std::cout << "  服务器IP: " << server_ip << std::endl;
    std::cout << "  客户端IP: " << client_ip << std::endl;

    // 设置UCX环境变量
    setenv("UCX_TLS", "tcp", 1);
    setenv("UCX_SOCKADDR_TLS_PRIORITY", "tcp", 1);
    setenv("UCX_TCP_CM_REUSEADDR", "y", 1);
    setenv("UCX_WARN_UNUSED_ENV_VARS", "n", 1);
    setenv("UCX_RNDV_THRESH", "8192", 1);

    // 创建连接缓冲区
    auto buffer = std::make_shared<ConnBuffer>(device_id, buffer_size, MemoryType::CPU);
    std::cout << "分配缓冲区成功: " << buffer->ptr << std::endl;
    std::cout << "缓冲区大小: " << buffer_size << " 字节" << std::endl;

    // 创建通信器
    Communicator *comm = new Communicator(buffer);
    std::cout << "创建通信器成功" << std::endl;

    // 连接到服务器
    std::cout << "尝试连接到服务器: " << server_ip << ":" << DEFAULT_PORT << std::endl;
    
    int max_retries = 5;
    bool connected = false;
    
    for (int retry = 0; retry < max_retries; retry++) {
        std::cout << "连接尝试 " << (retry + 1) << "/" << max_retries << std::endl;
        
        status_t conn_status = comm->connectTo(server_ip, DEFAULT_PORT, ConnType::UCX);
        if (conn_status == status_t::SUCCESS) {
            std::cout << "成功连接到服务器 " << server_ip << ":" << DEFAULT_PORT << std::endl;
            connected = true;
            break;
        }
        
        std::cout << "连接尝试失败，2秒后重试..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
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
    const char *data2 = "UCX communication test completed!";
    size_t data_size1 = strlen(data1) + 1;
    size_t data_size2 = strlen(data2) + 1;

    // 发送第一条消息
    buffer->writeFromCpu((void*)data1, data_size1, 0);
    std::cout << "\n数据复制到缓冲区: \"" << data1 << "\"" << std::endl;
    
    for (int attempt = 0; attempt < 3; attempt++) {
        std::cout << "尝试发送消息1 (尝试 " << (attempt + 1) << "/3)" << std::endl;
        
        status_t send_status = comm->writeTo(server_ip, 0, data_size1, ConnType::UCX);
        if (send_status == status_t::SUCCESS) {
            std::cout << "【内存互传确认】客户端成功发送: \"" << data1 << "\"" << std::endl;
            break;
        } else {
            std::cerr << "发送数据失败，1秒后重试" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // 等待服务器处理
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 发送第二条消息
    buffer->writeFromCpu((void*)data2, data_size2, 0);
    std::cout << "\n数据复制到缓冲区: \"" << data2 << "\"" << std::endl;
    
    for (int attempt = 0; attempt < 3; attempt++) {
        std::cout << "尝试发送消息2 (尝试 " << (attempt + 1) << "/3)" << std::endl;
        
        status_t send_status = comm->writeTo(server_ip, 0, data_size2, ConnType::UCX);
        if (send_status == status_t::SUCCESS) {
            std::cout << "【内存互传确认】客户端成功发送: \"" << data2 << "\"" << std::endl;
            break;
        } else {
            std::cerr << "发送数据失败，1秒后重试" << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // 等待服务器处理最后的消息
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 优雅关闭
    std::cout << "\n客户端断开连接..." << std::endl;
    comm->disConnect(server_ip, ConnType::UCX);
    delete comm;
    buffer.reset();
    std::cout << "客户端关闭完成" << std::endl;
    
    return 0;
}