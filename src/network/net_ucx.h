/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HMC_NET_UCX_H
#define HMC_NET_UCX_H

#include "net.h"

#ifdef ENABLE_UCX
#include <ucp/api/ucp.h>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

namespace hmc {

// UCX操作类型
enum class UcxOpType {
    NONE,
    SEND,
    RECV,
    PUT,
    GET,
    FLUSH
};

// 定义用于数据传输的标签
constexpr ucp_tag_t UCX_TAG_DATA = 0x123456789UL;
constexpr ucp_tag_t UCX_TAG_RKEY = 0x234567890UL;
constexpr ucp_tag_t UCX_TAG_CONNECTION_TEST = 0x987654321UL;

// 内存信息结构体
struct ucx_mem_info {
    uint64_t addr;     // 内存地址
    size_t   length;   // 内存长度
    char     rkey_buffer[256]; // RKEY缓冲区
    size_t   rkey_size;        // RKEY大小
};

// UCX请求结构
struct ucx_request {
    std::mutex mutex;
    std::condition_variable cv;
    bool completed;
    ucs_status_t status;
    UcxOpType op_type;
    
    ucx_request() : completed(false), status(UCS_OK), op_type(UcxOpType::NONE) {}
    
    void complete(ucs_status_t s) {
        std::lock_guard<std::mutex> lock(mutex);
        completed = true;
        status = s;
        cv.notify_all();
    }
    
    bool wait(int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex);
        if (completed) return true;
        
        return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                       [this]() { return completed; });
    }
};

// UCX端点类
class UCXEndpoint : public Endpoint {
public:
    UCXEndpoint(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep, 
                void* local_buf, size_t buf_size);
    ~UCXEndpoint();

    status_t writeData(size_t data_bias, size_t size) override;
    status_t readData(size_t data_bias, size_t size) override;
    status_t recvData(size_t data_bias, size_t size) override;

    // Non-blocking interfaces - 真正的非阻塞实现
    status_t writeDataNB(size_t data_bias, size_t size) override;
    status_t readDataNB(size_t data_bias, size_t size) override;
    status_t recvDataNB(size_t data_bias, size_t size) override;
    status_t pollCompletion(int num_completions_to_process) override;

    // UHM interface - UCX不支持，返回错误
    status_t uhm_send(void *input_buffer, const size_t send_flags, 
                     MemoryType mem_type) override {
        return status_t::UNSUPPORT;
    }
    status_t uhm_recv(void *output_buffer, const size_t buffer_size,
                     size_t *recv_flags, MemoryType mem_type) override {
        return status_t::UNSUPPORT;
    }

    status_t closeEndpoint() override;

    // 内存注册函数
    status_t registerMemory();
    // 内存取消注册函数
    status_t unregisterMemory();
    // 交换内存信息
    status_t exchangeMemoryInfo();

    // 获取远程内存地址
    uint64_t getRemoteAddress(size_t bias = 0) const;

    // 获取本地内存地址
    void* getLocalAddress(size_t bias = 0) const;
    
    // 获取worker - 新增方法
    ucp_worker_h getWorker() const;

private:
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_ep_h ep;
    void* buffer;                // 本地内存缓冲区
    size_t buffer_size;          // 内存大小
    std::atomic<bool> is_connected;

    // 内存注册相关
    ucp_mem_h memh;              // 内存句柄
    ucx_mem_info local_mem_info; // 本地内存信息
    ucx_mem_info remote_mem_info; // 远程内存信息
    ucp_rkey_h remote_rkey;      // 远程内存密钥
    bool mem_registered;         // 内存是否已注册
    bool rkey_exchanged;         // 是否已交换密钥

    // 异步请求管理
    std::vector<ucs_status_ptr_t> pending_requests;  // 挂起的异步请求列表
    std::mutex requests_mutex;                       // 保护请求列表的互斥锁

    // 等待请求完成（此函数保留超时，用于内存信息交换等特殊场景）
    bool waitRequest(ucs_status_ptr_t req, int timeout_ms = 1000);
    // 等待所有挂起请求完成
    status_t waitAllPendingRequests();
    // 等待Worker进度
    void progressWorker(int timeout_ms = 1000);
};

// UCX服务器类 - 修改：增加buffer成员
class UCXServer : public Server {
public:
    // 修改：构造函数同时接收conn_manager和buffer
    UCXServer(std::shared_ptr<ConnManager> conn_manager, 
              std::shared_ptr<ConnBuffer> buffer);
    ~UCXServer();

    status_t listen(std::string ip, uint16_t port) override;
    status_t stopListen() override;

    // 获取UCX资源(用于回调函数)
    ucp_worker_h getWorker() { return worker; }
    ucp_context_h getContext() { return context; }
    ucp_listener_h getListener() { return listener; }
    
    // 新增：获取buffer的方法
    std::shared_ptr<ConnBuffer> getBuffer() { return buffer; }

    // 使conn_manager在ucx_conn_handler中可访问
    friend void ucx_conn_handler(ucp_conn_request_h conn_request, void *arg);

private:
    ucp_context_h context;
    ucp_worker_h worker;
    ucp_listener_h listener;
    std::atomic<bool> running;
    std::thread worker_thread;
    
    // 新增：直接持有buffer，与Client保持一致
    std::shared_ptr<ConnBuffer> buffer;

    void progressThread();
    void configureUCX(ucp_config_t* config, const std::string& bind_ip);
};

// UCX客户端类
class UCXClient : public Client {
public:
    UCXClient(std::shared_ptr<ConnBuffer> buffer);
    ~UCXClient();

    std::unique_ptr<Endpoint> connect(std::string ip, uint16_t port) override;

private:
    std::shared_ptr<ConnBuffer> buffer;
    void configureUCX(ucp_config_t* config, const std::string& target_ip);
};

// 回调函数
void ucx_request_init(void *request);
void ucx_request_cleanup(void *request);
void ucx_send_cb(void *request, ucs_status_t status);
void ucx_recv_cb(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
void ucx_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status);
void ucx_conn_handler(ucp_conn_request_h conn_request, void *arg);

// 端口检查辅助函数
bool isPortAvailable(uint16_t port);

// 网络接口选择辅助函数
std::string getLocalIPForTarget(const std::string& target_ip);
std::string getInterfaceForIP(const std::string& ip);

} // namespace hmc

#endif // ENABLE_UCX
#endif // HMC_NET_UCX_H