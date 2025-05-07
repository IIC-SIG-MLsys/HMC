/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
 #include "net_ucx.h"
 #include <arpa/inet.h>
 #include <netinet/in.h>
 #include <sys/socket.h>
 
 namespace hmc {
 
 UCXClient::UCXClient(std::shared_ptr<ConnBuffer> buffer) : buffer(buffer) {
     logInfo("UCXClient created");
 }
 
 UCXClient::~UCXClient() {
     // 由Endpoint析构函数负责清理UCX资源
 }
 
 std::unique_ptr<Endpoint> UCXClient::connect(std::string ip, uint16_t port) {
    logInfo("UCXClient: Connecting to %s:%d", ip.c_str(), port);
    
    // 创建UCP上下文
    ucp_context_h context = nullptr;
    ucp_worker_h worker = nullptr;
    ucp_ep_h ep = nullptr;
    
    // 初始化UCP
    ucp_config_t *config = nullptr;
    ucp_params_t ucp_params;
    
    if (ucp_config_read(NULL, NULL, &config) != UCS_OK) {
        logError("UCXClient: Failed to read UCP config");
        return nullptr;
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
        logError("UCXClient: Failed to initialize UCP");
        ucp_config_release(config);
        return nullptr;
    }
    
    ucp_config_release(config);
    
    // 创建worker
    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE | UCP_WORKER_PARAM_FIELD_FLAGS;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    worker_params.flags = UCP_WORKER_FLAG_IGNORE_REQUEST_LEAK;
    
    if (ucp_worker_create(context, &worker_params, &worker) != UCS_OK) {
        logError("UCXClient: Failed to create worker");
        ucp_cleanup(context);
        return nullptr;
    }
    
    // 连接到服务器
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) <= 0) {
        logError("UCXClient: Invalid IP address: %s", ip.c_str());
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    // 创建端点
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | 
                          UCP_EP_PARAM_FIELD_SOCK_ADDR |
                          UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = (struct sockaddr*)&server_addr;
    ep_params.sockaddr.addrlen = sizeof(server_addr);
    ep_params.err_handler.cb = ucx_error_handler;
    ep_params.err_handler.arg = nullptr;
    
    ucs_status_t status = ucp_ep_create(worker, &ep_params, &ep);
    if (status != UCS_OK) {
        logError("UCXClient: Failed to create endpoint: %s", ucs_status_string(status));
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    // 确保连接建立 - 增加重试次数和超时时间
    int max_retries = 10;
    bool connected = false;
    
    for (int retry = 0; retry < max_retries; retry++) {
        // 处理进度并等待
        for (int i = 0; i < 20; i++) {
            ucp_worker_progress(worker);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // 发送测试消息确认连接
        char test_msg[] = "connection_test";
        logInfo("UCXClient: Sending connection test message (attempt %d/%d)", 
               retry + 1, max_retries);
        
        ucs_status_ptr_t req = ucp_tag_send_nb(ep, test_msg, sizeof(test_msg),
                                             ucp_dt_make_contig(1), UCX_TAG_CONNECTION_TEST, 
                                             ucx_send_cb);
        
        if (UCS_PTR_IS_ERR(req)) {
            if (retry < max_retries - 1) {
                logInfo("UCXClient: Connection test failed (retry %d/%d): %s", 
                        retry + 1, max_retries, ucs_status_string(UCS_PTR_STATUS(req)));
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            } else {
                logError("UCXClient: All connection tests failed: %s", 
                        ucs_status_string(UCS_PTR_STATUS(req)));
                ucp_ep_destroy(ep);
                ucp_worker_destroy(worker);
                ucp_cleanup(context);
                return nullptr;
            }
        }
        
        bool req_completed = false;
        
        if (UCS_PTR_IS_PTR(req)) {
            // 等待请求完成
            for (int i = 0; i < 100; i++) {
                ucp_worker_progress(worker);
                
                if (ucp_request_is_completed(req)) {
                    req_completed = true;
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            auto *ctx = static_cast<ucx_request*>(req);
            bool success = false;
            
            {
                std::unique_lock<std::mutex> lock(ctx->mutex);
                success = ctx->completed && (ctx->status == UCS_OK);
            }
            
            ucp_request_free(req);
            
            if (success) {
                req_completed = true;
            } else if (retry < max_retries - 1) {
                logInfo("UCXClient: Connection test timed out (retry %d/%d)", 
                        retry + 1, max_retries);
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            } else {
                logError("UCXClient: All connection tests timed out");
                ucp_ep_destroy(ep);
                ucp_worker_destroy(worker);
                ucp_cleanup(context);
                return nullptr;
            }
        } else {
            // 操作立即完成
            req_completed = true;
        }
        
        if (req_completed) {
            // 等待额外的时间，确保服务器已准备好
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            connected = true;
            break;
        }
    }
    
    if (!connected) {
        logError("UCXClient: Failed to establish connection after %d retries", max_retries);
        ucp_ep_destroy(ep);
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    logInfo("UCXClient: Successfully connected to %s:%d", ip.c_str(), port);
    
    // 创建UCX端点对象
    auto endpoint = std::make_unique<UCXEndpoint>(
        context, worker, ep, buffer->ptr, buffer->buffer_size);
    
    endpoint->role = EndpointType::Client;
    
    // 注册内存
    logInfo("UCXClient: Registering memory...");
    if (endpoint->registerMemory() != status_t::SUCCESS) {
        logError("UCXClient: Failed to register memory");
        return nullptr;
    }
    
    // 在返回成功之前，确保内存信息交换完成
    logInfo("UCXClient: Connection established and memory registered, exchanging memory info...");
    
    // 等待一段时间，以确保服务器已准备好交换内存信息
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
    // 提前交换内存信息，避免第一次通信时的延迟
    if (endpoint->exchangeMemoryInfo() != status_t::SUCCESS) {
        logError("UCXClient: Failed to exchange memory info");
        return nullptr;
    }
    
    logInfo("UCXClient: Memory info exchanged successfully");
    logInfo("UCXClient: Connection established successfully");
    
    return endpoint;
 }
 
 } // namespace hmc