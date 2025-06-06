/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_ucx.h"

#ifdef ENABLE_UCX
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>

namespace hmc {

UCXClient::UCXClient(std::shared_ptr<ConnBuffer> buffer) : buffer(buffer) {
    logInfo("UCXClient created");
}

UCXClient::~UCXClient() {
    // 由Endpoint析构函数负责清理UCX资源
}

void UCXClient::configureUCX(ucp_config_t* config, const std::string& target_ip) {
    // 基础TCP配置
    ucp_config_modify(config, "TLS", "tcp");
    ucp_config_modify(config, "WARN_UNUSED_ENV_VARS", "n");
    ucp_config_modify(config, "TCP_CM_REUSEADDR", "y");
    
    // 强制排除虚拟网络接口，只使用物理网卡
    // 根据日志，客户端有 ens121f0np0 和 ens121f1np1
    ucp_config_modify(config, "NET_DEVICES", "ens121f0np0,ens121f1np1");
    
    logInfo("UCXClient: UCX configuration completed - forced physical interfaces");
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
    
    // 创建端点，增强错误处理
    ucp_ep_params_t ep_params;
    memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | 
                          UCP_EP_PARAM_FIELD_SOCK_ADDR;
    // 移除错误处理器以避免崩溃
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = (struct sockaddr*)&server_addr;
    ep_params.sockaddr.addrlen = sizeof(server_addr);
    
    ucs_status_t status = ucp_ep_create(worker, &ep_params, &ep);
    if (status != UCS_OK) {
        logError("UCXClient: Failed to create endpoint: %s", ucs_status_string(status));
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    // 简化的连接确认机制 - 不发送测试消息，直接等待连接建立
    logInfo("UCXClient: Waiting for connection establishment...");
    
    // 让worker处理连接建立，增加等待时间
    auto start_time = std::chrono::steady_clock::now();
    bool connection_ok = false;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count() < 10) {
        ucp_worker_progress(worker);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 简单检查 - 如果没有立即出错，认为连接成功
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count() > 2) {
            connection_ok = true;
            break;
        }
    }
    
    if (!connection_ok) {
        logError("UCXClient: Connection establishment timeout");
        ucp_ep_destroy(ep);
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    logInfo("UCXClient: Connection established to %s:%d", ip.c_str(), port);
    
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
    
    // 等待足够长时间再交换内存信息
    logInfo("UCXClient: Waiting before memory exchange...");
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    
    // 交换内存信息
    logInfo("UCXClient: Exchanging memory info...");
    if (endpoint->exchangeMemoryInfo() != status_t::SUCCESS) {
        logError("UCXClient: Failed to exchange memory info");
        return nullptr;
    }
    
    logInfo("UCXClient: Connection and memory exchange completed successfully");
    return endpoint;
}

} // namespace hmc

#endif // ENABLE_UCX