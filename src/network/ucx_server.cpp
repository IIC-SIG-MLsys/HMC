/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
 #include "net_ucx.h"
 #include <arpa/inet.h>
 #include <netinet/in.h>
 #include <sys/socket.h>
 #include <unistd.h>
 
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
     logError("UCX error occurred: %s", ucs_status_string(status));
 }
 
 void ucx_conn_handler(ucp_conn_request_h conn_request, void *arg) {
    auto *server = static_cast<UCXServer*>(arg);
    
    if (!server) {
        logError("UCX connection handler called with null server");
        return;
    }
    
    logInfo("UCX connection handler: New connection request received");
    
    // 创建端点参数
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST | 
                          UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.conn_request = conn_request;
    ep_params.err_handler.cb = ucx_error_handler;
    ep_params.err_handler.arg = nullptr;
    
    // 创建端点
    ucp_ep_h client_ep = nullptr;
    ucs_status_t status = ucp_ep_create(server->getWorker(), &ep_params, &client_ep);
    
    if (status != UCS_OK) {
        logError("UCX connection handler: Failed to create endpoint: %s", 
                ucs_status_string(status));
        return;
    }
    
    logInfo("UCX connection handler: Endpoint created successfully");
    
    // 获取通信器(ConnManager)
    auto conn_manager = server->conn_manager;
    auto buffer = conn_manager->getBuffer();
    
    logInfo("UCX connection handler: Server buffer: %p, size: %zu", 
           buffer->ptr, buffer->buffer_size);
    
    // 创建端点对象
    auto endpoint = std::make_unique<UCXEndpoint>(
        server->getContext(), server->getWorker(), client_ep, 
        buffer->ptr, buffer->buffer_size);
    
    endpoint->role = EndpointType::Server;
    
    // 确保内存正确注册
    if (endpoint->registerMemory() != status_t::SUCCESS) {
        logError("UCX connection handler: Failed to register memory");
        return;
    }
    
    // 尝试获取客户端IP
    std::string client_ip = "client";
    
    // 保存端点的指针，以便在添加到连接管理器后仍能访问
    UCXEndpoint* ep_ptr = endpoint.get();
    
    // 将端点添加到连接管理器
    conn_manager->_addEndpoint(client_ip, std::move(endpoint));
    
    logInfo("UCX connection handler: Added new endpoint for client %s", client_ip.c_str());
    
    // 在新线程中启动内存信息交换，避免阻塞连接处理线程
    std::thread exchange_thread([ep_ptr]() {
        // 短暂等待确保连接稳定
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        logInfo("UCX server: Starting memory info exchange");
        
        // 主动交换内存信息
        if (ep_ptr->exchangeMemoryInfo() != status_t::SUCCESS) {
            logError("UCX server: Failed to exchange memory info with client");
        } else {
            logInfo("UCX server: Successfully exchanged memory info with client");
        }
    });
    
    // 分离线程，让它在后台运行
    exchange_thread.detach();
}
 
 // 检查端口是否可用
 bool isPortAvailable(uint16_t port) {
     if (port == 0) return true;  // 动态端口总是可用的
     
     int sockfd = socket(AF_INET, SOCK_STREAM, 0);
     if (sockfd < 0) return false;
     
     struct sockaddr_in addr;
     memset(&addr, 0, sizeof(addr));
     addr.sin_family = AF_INET;
     addr.sin_addr.s_addr = htonl(INADDR_ANY);
     addr.sin_port = htons(port);
     
     int result = bind(sockfd, (struct sockaddr*)&addr, sizeof(addr));
     close(sockfd);
     
     return result == 0;
 }
 
 UCXServer::UCXServer(std::shared_ptr<ConnManager> conn_manager) 
     : Server(conn_manager), context(nullptr), worker(nullptr), 
       listener(nullptr), running(false) {
     logInfo("UCXServer created");
 }
 
 UCXServer::~UCXServer() {
     // 停止进度线程
     running.store(false);
     if (worker_thread.joinable()) {
         worker_thread.join();
     }
     
     // 清理资源
     if (listener) {
         ucp_listener_destroy(listener);
     }
     
     if (worker) {
         ucp_worker_destroy(worker);
     }
     
     if (context) {
         ucp_cleanup(context);
     }
     
     logInfo("UCXServer destroyed");
 }

 // 新增的stopListen方法实现
 status_t UCXServer::stopListen() {
     logInfo("UCXServer: Stopping listener");
     
     // 停止运行标志，让进度线程退出
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
     
     // 不销毁worker和context，留给析构函数处理
     // 这样可以确保已连接的端点不会受到影响
     
     logInfo("UCXServer: Listener stopped successfully");
     return status_t::SUCCESS;
 }
 
 status_t UCXServer::listen(std::string ip, uint16_t port) {
     logInfo("UCXServer: Starting to listen on %s:%d", ip.c_str(), port);
     
     // 初始化UCP
     ucp_config_t *config = nullptr;
     ucp_params_t ucp_params;
     
     if (ucp_config_read(NULL, NULL, &config) != UCS_OK) {
         logError("UCXServer: Failed to read UCP config");
         return status_t::ERROR;
     }
     
     // 简化配置
     ucp_config_modify(config, "TLS", "tcp");
     ucp_config_modify(config, "SOCKADDR_TLS_PRIORITY", "tcp");
     ucp_config_modify(config, "TCP_CM_REUSEADDR", "y");
     
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
     
     // 设置监听地址
     struct sockaddr_in listen_addr;
     memset(&listen_addr, 0, sizeof(listen_addr));
     listen_addr.sin_family = AF_INET;
     listen_addr.sin_port = htons(port);
     
     if (ip == "0.0.0.0") {
         listen_addr.sin_addr.s_addr = INADDR_ANY;
     } else if (inet_pton(AF_INET, ip.c_str(), &listen_addr.sin_addr) <= 0) {
         logError("UCXServer: Invalid IP address: %s", ip.c_str());
         ucp_worker_destroy(worker);
         ucp_cleanup(context);
         return status_t::ERROR;
     }
     
     // 检查端口是否可用
     if (!isPortAvailable(port)) {
         logInfo("UCXServer: Port %d is busy, trying with another port", port);
         listen_addr.sin_port = 0;  // 使用随机端口
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
     
     // 尝试监听
     ucs_status_t status = ucp_listener_create(worker, &listener_params, &listener);
     if (status != UCS_OK) {
         logError("UCXServer: Failed to create listener: %s", 
                 ucs_status_string(status));
         ucp_worker_destroy(worker);
         ucp_cleanup(context);
         return status_t::ERROR;
     }
     
     // 获取实际使用的端口 
     uint16_t actual_port = port;  // 默认使用请求的端口
     
     // 如果使用了动态端口，尝试获取实际端口
     if (port == 0) {
         // 由于无法可靠地获取实际端口，我们只能使用一个预定义的端口
         // 或者询问用户提供一个端口
         actual_port = 12345;  // 使用预设的端口
         logInfo("UCXServer: Using dynamic port allocation, assume port %d", actual_port);
     }
     
     // 写入端口发现文件，便于客户端连接
     std::ofstream port_file("/tmp/ucx_server_port.txt");
     if (port_file.is_open()) {
         port_file << actual_port;
         port_file.close();
         logInfo("UCXServer: Port info written to /tmp/ucx_server_port.txt");
     } else {
         logInfo("UCXServer: Could not write port info to file");
     }
     
     // 启动进度线程
     running.store(true);
     worker_thread = std::thread(&UCXServer::progressThread, this);
     
     return status_t::SUCCESS;
 }
 
 void UCXServer::progressThread() {
     logInfo("UCXServer: Progress thread started");
     
     // 简化的进度线程 - 直接使用工作线程轮询而不是epoll
     while (running.load()) {
         // 推进UCX工作进度
         ucp_worker_progress(worker);
         
         // 短暂休眠，减少CPU占用
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
     }
     
     logInfo("UCXServer: Progress thread stopped");
 }
 
 } // namespace hmc