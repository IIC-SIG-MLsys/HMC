/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_ucx.h"

#ifdef ENABLE_UCX
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <ifaddrs.h>
#include <netdb.h>

namespace hmc {

// UCX回调函数实现
void ucx_request_init(void *request) {
    auto *ctx = static_cast<ucx_request*>(request);
    new(ctx) ucx_request();
}

void ucx_request_cleanup(void *request) {
    auto *ctx = static_cast<ucx_request*>(request);
    ctx->~ucx_request();
}

void ucx_send_cb(void *request, ucs_status_t status) {
    auto *ctx = static_cast<ucx_request*>(request);
    ctx->complete(status);
}

void ucx_recv_cb(void *request, ucs_status_t status, ucp_tag_recv_info_t *info) {
    auto *ctx = static_cast<ucx_request*>(request);
    ctx->complete(status);
}

void ucx_error_handler(void *arg, ucp_ep_h ep, ucs_status_t status) {
    // 减少错误日志，避免崩溃
    if (status != UCS_ERR_CONNECTION_RESET && status != UCS_ERR_CANCELED) {
        logDebug("UCX error: %s", ucs_status_string(status));
    }
}

void ucx_conn_handler(ucp_conn_request_h conn_request, void *arg) {
    auto *server = static_cast<UCXServer*>(arg);
    
    logInfo("===============================================");
    logInfo("UCX SERVER: NEW CONNECTION REQUEST RECEIVED!");
    logInfo("===============================================");
    
    if (!server) {
        logError("UCX connection handler called with null server");
        ucp_listener_reject(server->getListener(), conn_request);
        return;
    }
    
    // 获取客户端地址信息
    ucp_conn_request_attr_t attr;
    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    ucs_status_t query_status = ucp_conn_request_query(conn_request, &attr);
    
    std::string client_ip = "unknown_client";
    if (query_status == UCS_OK && attr.client_address.ss_family == AF_INET) {
        struct sockaddr_in* addr_in = (struct sockaddr_in*)&attr.client_address;
        char ip_str[INET_ADDRSTRLEN];
        if (inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, INET_ADDRSTRLEN)) {
            client_ip = std::string(ip_str);
            logInfo("UCX SERVER: Client connecting from IP: %s", client_ip.c_str());
        }
    } else {
        logInfo("UCX SERVER: Failed to query client address, status: %s", 
                  ucs_status_string(query_status));
    }
    
    // 创建端点参数
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_request;
    
    logInfo("UCX SERVER: Creating endpoint for client %s", client_ip.c_str());
    
    // 创建端点
    ucp_ep_h client_ep = nullptr;
    ucs_status_t status = ucp_ep_create(server->getWorker(), &ep_params, &client_ep);
    
    if (status != UCS_OK) {
        logError("UCX SERVER: Failed to create endpoint: %s", 
                ucs_status_string(status));
        ucp_listener_reject(server->getListener(), conn_request);
        return;
    }
    
    logInfo("UCX SERVER: Endpoint created successfully for client %s", client_ip.c_str());
    
    // 获取buffer
    auto buffer = server->getBuffer();
    if (!buffer) {
        logError("UCX SERVER: Failed to get buffer from server");
        ucp_ep_destroy(client_ep);
        return;
    }
    
    logInfo("UCX SERVER: Using server buffer: %p, size: %zu", 
           buffer->ptr, buffer->buffer_size);
    
    // 创建端点对象
    auto endpoint = std::make_unique<UCXEndpoint>(
        server->getContext(), server->getWorker(), client_ep, 
        buffer->ptr, buffer->buffer_size);
    
    endpoint->role = EndpointType::Server;
    
    logInfo("UCX SERVER: UCXEndpoint object created for client %s", client_ip.c_str());
    
    // 注册内存
    if (endpoint->registerMemory() != status_t::SUCCESS) {
        logError("UCX SERVER: Failed to register memory for client %s", client_ip.c_str());
        return;
    }
    
    logInfo("UCX SERVER: Memory registered for client %s", client_ip.c_str());
    
    // 保存端点指针
    UCXEndpoint* ep_ptr = endpoint.get();
    
    // 将端点添加到连接管理器
    auto conn_manager = server->conn_manager;
    conn_manager->_addEndpoint(client_ip, std::move(endpoint));
    
    logInfo("UCX SERVER: Added new endpoint for client %s", client_ip.c_str());
    
    // 启动后台任务处理连接测试和内存交换
    std::thread connection_handler([ep_ptr, client_ip, server]() {
        logInfo("UCX SERVER: Starting connection handler thread for client %s", client_ip.c_str());
        
        // 1. 等待并处理连接测试消息
        char test_buffer[16];
        ucp_tag_t tag_mask = ~0ULL;
        
        logInfo("UCX SERVER: Waiting for connection test message from %s", client_ip.c_str());
        
        ucs_status_ptr_t recv_req = ucp_tag_recv_nb(server->getWorker(), test_buffer, sizeof(test_buffer),
                                                 ucp_dt_make_contig(1),
                                                 UCX_TAG_CONNECTION_TEST, tag_mask,
                                                 ucx_recv_cb);
        
        if (UCS_PTR_IS_ERR(recv_req)) {
            logError("UCX SERVER: Failed to post test message receive: %s", 
                    ucs_status_string(UCS_PTR_STATUS(recv_req)));
        } else {
            // 等待接收完成
            bool test_received = false;
            if (UCS_PTR_IS_PTR(recv_req)) {
                auto *ctx = static_cast<ucx_request*>(recv_req);
                if (ctx->wait(10000)) {  // 10秒超时
                    test_received = true;
                    logInfo("UCX SERVER: Received connection test message from %s: %s", 
                           client_ip.c_str(), test_buffer);
                } else {
                    logError("UCX SERVER: Connection test message timeout from %s", client_ip.c_str());
                }
                ucp_request_free(recv_req);
            } else {
                // 立即完成
                test_received = true;
                logInfo("UCX SERVER: Received connection test message from %s (immediate): %s", 
                       client_ip.c_str(), test_buffer);
            }
            
            if (!test_received) {
                logError("UCX SERVER: Failed to receive connection test from %s", client_ip.c_str());
                return;
            }
        }
        
        // 2. 等待连接稳定
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        
        // 3. 进行内存信息交换
        logInfo("UCX SERVER: Starting memory info exchange with %s", client_ip.c_str());
        
        int max_retries = 3;
        for (int retry = 0; retry < max_retries; retry++) {
            logInfo("UCX SERVER: Memory exchange attempt %d/%d with client %s", 
                   retry + 1, max_retries, client_ip.c_str());
                   
            if (ep_ptr->exchangeMemoryInfo() == status_t::SUCCESS) {
                logInfo("UCX SERVER: Successfully exchanged memory info with client %s", client_ip.c_str());
                logInfo("===============================================");
                logInfo("UCX SERVER: CONNECTION SETUP COMPLETED FOR %s", client_ip.c_str());
                logInfo("===============================================");
                return;
            }
            
            if (retry < max_retries - 1) {
                logError("UCX SERVER: Memory info exchange failed with %s, retrying in 2 seconds...", client_ip.c_str());
                std::this_thread::sleep_for(std::chrono::seconds(2));
            }
        }
        
        logError("UCX SERVER: Failed to exchange memory info with client %s after %d retries", 
                client_ip.c_str(), max_retries);
    });
    
    // 分离线程
    connection_handler.detach();
    
    logInfo("UCX SERVER: Connection handler started for %s", client_ip.c_str());
}

// 检查端口是否可用
bool isPortAvailable(uint16_t port) {
    if (port == 0) return true;
    
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) return false;
    
    int opt = 1;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    
    int result = bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
    close(sockfd);
    
    return result == 0;
}

UCXServer::UCXServer(std::shared_ptr<ConnManager> conn_manager, 
                     std::shared_ptr<ConnBuffer> buffer) 
    : Server(conn_manager), context(nullptr), worker(nullptr), 
      listener(nullptr), running(false), buffer(buffer) {
    logInfo("UCXServer created with buffer %p, size %zu", buffer->ptr, buffer->buffer_size);
}

UCXServer::~UCXServer() {
    logInfo("UCXServer: Starting cleanup...");
    
    // 停止进度线程
    running.store(false);
    if (worker_thread.joinable()) {
        worker_thread.join();
        logInfo("UCXServer: Progress thread stopped");
    }
    
    // 清理资源
    if (listener) {
        ucp_listener_destroy(listener);
        logInfo("UCXServer: Listener destroyed");
    }
    
    if (worker) {
        ucp_worker_destroy(worker);
        logInfo("UCXServer: Worker destroyed");
    }
    
    if (context) {
        ucp_cleanup(context);
        logInfo("UCXServer: Context cleaned up");
    }
    
    logInfo("UCXServer destroyed");
}

void UCXServer::configureUCX(ucp_config_t* config, const std::string& bind_ip) {
    logInfo("UCXServer: Configuring UCX for bind IP %s", bind_ip.c_str());
    
    // 基础TCP配置
    ucp_config_modify(config, "TLS", "tcp");
    ucp_config_modify(config, "WARN_UNUSED_ENV_VARS", "n");
    ucp_config_modify(config, "TCP_CM_REUSEADDR", "y");
    
    // 网络接口配置 - 修复：让UCX自动选择
    const char* env_interfaces = getenv("UCX_NET_DEVICES");
    if (env_interfaces && strlen(env_interfaces) > 0) {
        ucp_config_modify(config, "NET_DEVICES", env_interfaces);
        logInfo("UCXServer: Using network interfaces from environment: %s", env_interfaces);
    } else {
        // 不设置NET_DEVICES，让UCX自动选择最佳接口
        logInfo("UCXServer: Using UCX automatic interface selection");
    }
    
    // 添加调试和稳定性配置
    ucp_config_modify(config, "LOG_LEVEL", "info");
    ucp_config_modify(config, "SOCKADDR_TLS_PRIORITY", "tcp");
    ucp_config_modify(config, "TCP_KEEPIDLE", "600");
    ucp_config_modify(config, "TCP_KEEPINTVL", "60");
    ucp_config_modify(config, "TCP_KEEPCNT", "5");
    
    logInfo("UCXServer: UCX configuration completed for bind IP %s", bind_ip.c_str());
}

status_t UCXServer::listen(std::string ip, uint16_t port) {
    logInfo("===============================================");
    logInfo("UCXServer: STARTING TO LISTEN ON %s:%d", ip.c_str(), port);
    logInfo("===============================================");
    
    // 初始化UCP
    ucp_config_t *config = nullptr;
    ucp_params_t ucp_params;
    
    if (ucp_config_read(NULL, NULL, &config) != UCS_OK) {
        logError("UCXServer: Failed to read UCP config");
        return status_t::ERROR;
    }
    
    // 配置UCX
    configureUCX(config, ip);
    
    // 初始化UCP上下文
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | 
                           UCP_PARAM_FIELD_REQUEST_SIZE | 
                           UCP_PARAM_FIELD_REQUEST_INIT | 
                           UCP_PARAM_FIELD_REQUEST_CLEANUP;
    ucp_params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA | UCP_FEATURE_WAKEUP;
    ucp_params.request_size = sizeof(ucx_request);
    ucp_params.request_init = ucx_request_init;
    ucp_params.request_cleanup = ucx_request_cleanup;
    
    if (ucp_init(&ucp_params, config, &context) != UCS_OK) {
        logError("UCXServer: Failed to initialize UCP");
        ucp_config_release(config);
        return status_t::ERROR;
    }
    
    ucp_config_release(config);
    logInfo("UCXServer: UCP context initialized");
    
    // 创建worker
    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE | UCP_WORKER_PARAM_FIELD_FLAGS;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    worker_params.flags = UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK;
    
    if (ucp_worker_create(context, &worker_params, &worker) != UCS_OK) {
        logError("UCXServer: Failed to create worker");
        ucp_cleanup(context);
        return status_t::ERROR;
    }
    
    logInfo("UCXServer: Worker created");
    
    // 设置监听地址
    struct sockaddr_in listen_addr;
    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_port = htons(port);
    
    if (ip == "0.0.0.0") {
        listen_addr.sin_addr.s_addr = INADDR_ANY;
        logInfo("UCXServer: Binding to all interfaces (0.0.0.0)");
    } else if (inet_pton(AF_INET, ip.c_str(), &listen_addr.sin_addr) <= 0) {
        logError("UCXServer: Invalid IP address: %s", ip.c_str());
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return status_t::ERROR;
    } else {
        logInfo("UCXServer: Binding to specific interface: %s", ip.c_str());
    }
    
    // 创建监听参数
    ucp_listener_params_t listener_params;
    memset(&listener_params, 0, sizeof(listener_params));
    listener_params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | 
                               UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listener_params.sockaddr.addr = (struct sockaddr*)&listen_addr;
    listener_params.sockaddr.addrlen = sizeof(listen_addr);
    listener_params.conn_handler.cb = ucx_conn_handler;
    listener_params.conn_handler.arg = this;
    
    logInfo("UCXServer: Creating listener with connection handler...");
    
    // 尝试监听
    ucs_status_t status = ucp_listener_create(worker, &listener_params, &listener);
    if (status != UCS_OK) {
        logError("UCXServer: Failed to create listener: %s", 
                ucs_status_string(status));
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return status_t::ERROR;
    }
    
    logInfo("UCXServer: Listener created successfully");
    
    logInfo("===============================================");
    logInfo("UCXServer: READY TO ACCEPT CONNECTIONS ON %s:%d", ip.c_str(), port);
    logInfo("===============================================");
    
    // 启动进度线程
    running.store(true);
    worker_thread = std::thread(&UCXServer::progressThread, this);
    
    return status_t::SUCCESS;
}

status_t UCXServer::stopListen() {
    logInfo("UCXServer: Stopping listener");
    
    // 停止运行标志
    running.store(false);
    
    // 等待进度线程结束
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
    
    // 销毁监听器
    if (listener) {
        ucp_listener_destroy(listener);
        listener = nullptr;
    }
    
    logInfo("UCXServer: Listener stopped successfully");
    return status_t::SUCCESS;
}

void UCXServer::progressThread() {
    logInfo("UCXServer: Progress thread started, running worker progress loop...");
    
    int progress_counter = 0;
    
    // 进度线程主循环
    while (running.load()) {
        // 推进UCX工作进度
        ucp_worker_progress(worker);
        progress_counter++;
        
        // 每30秒输出一次活跃状态（减少日志频率）
        if (progress_counter % 30000 == 0) {
            logInfo("UCXServer: Progress thread active, processed %d cycles", progress_counter);
        }
        
        // 适当休眠，减少CPU占用
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    logInfo("UCXServer: Progress thread stopped after %d cycles", progress_counter);
}

} // namespace hmc

#endif // ENABLE_UCX