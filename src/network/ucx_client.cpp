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
#include <cstdlib>

namespace hmc {

UCXClient::UCXClient(std::shared_ptr<ConnBuffer> buffer) : buffer(buffer) {
    logInfo("UCXClient created");
}

UCXClient::~UCXClient() {
    // 由Endpoint析构函数负责清理UCX资源
}

// 根据目标IP获取本地合适的网络接口
std::string UCXClient::getLocalInterfaceForTarget(const std::string& target_ip) {
    struct ifaddrs *ifaddr, *ifa;
    std::vector<std::string> physical_interfaces;
    std::string best_interface;
    
    if (getifaddrs(&ifaddr) == -1) {
        logError("UCXClient: Failed to get network interfaces");
        return "";
    }
    
    logInfo("UCXClient: Analyzing network interfaces for target %s", target_ip.c_str());
    
    // 收集所有物理网络接口
    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL || ifa->ifa_addr->sa_family != AF_INET)
            continue;
            
        std::string if_name = ifa->ifa_name;
        
        // 排除虚拟接口 (lo, docker, veth, br-, virbr等)
        if (if_name.find("lo") == 0 || 
            if_name.find("docker") == 0 ||
            if_name.find("veth") == 0 ||
            if_name.find("br-") == 0 ||
            if_name.find("virbr") == 0 ||
            if_name.find("vxlan") == 0) {
            continue;
        }
        
        struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
        char local_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr_in->sin_addr, local_ip, INET_ADDRSTRLEN);
        
        logDebug("UCXClient: Found interface %s with IP %s", if_name.c_str(), local_ip);
        
        // 优先选择与目标IP在同一网段的接口
        struct in_addr target_addr, local_addr;
        if (inet_pton(AF_INET, target_ip.c_str(), &target_addr) == 1 &&
            inet_pton(AF_INET, local_ip, &local_addr) == 1) {
            
            uint32_t target_subnet = ntohl(target_addr.s_addr) & 0xFFFFFF00;
            uint32_t local_subnet = ntohl(local_addr.s_addr) & 0xFFFFFF00;
            
            if (target_subnet == local_subnet) {
                best_interface = if_name;
                logInfo("UCXClient: Found interface %s (%s) in same subnet as target %s", 
                       if_name.c_str(), local_ip, target_ip.c_str());
                break;
            }
        }
        
        // 收集物理接口作为备选
        physical_interfaces.push_back(if_name);
    }
    
    freeifaddrs(ifaddr);
    
    // 如果没找到同网段的接口，使用第一个物理接口
    if (best_interface.empty() && !physical_interfaces.empty()) {
        best_interface = physical_interfaces[0];
        logInfo("UCXClient: Using first available physical interface: %s", best_interface.c_str());
    }
    
    return best_interface;
}

void UCXClient::configureUCX(ucp_config_t* config, const std::string& target_ip) {
    logInfo("UCXClient: Configuring UCX for target %s", target_ip.c_str());
    
    // 基础TCP配置
    ucp_config_modify(config, "TLS", "tcp");
    ucp_config_modify(config, "WARN_UNUSED_ENV_VARS", "n");
    ucp_config_modify(config, "TCP_CM_REUSEADDR", "y");
    
    // 网络接口配置 - 修复：让UCX自动选择
    const char* env_interfaces = getenv("UCX_NET_DEVICES");
    if (env_interfaces && strlen(env_interfaces) > 0) {
        ucp_config_modify(config, "NET_DEVICES", env_interfaces);
        logInfo("UCXClient: Using network interfaces from environment: %s", env_interfaces);
    } else {
        // 不设置NET_DEVICES，让UCX自动选择最佳接口
        logInfo("UCXClient: Using UCX automatic interface selection");
    }
    
    // 添加调试和稳定性配置
    ucp_config_modify(config, "LOG_LEVEL", "info");
    ucp_config_modify(config, "SOCKADDR_TLS_PRIORITY", "tcp");
    ucp_config_modify(config, "TCP_KEEPIDLE", "600");
    ucp_config_modify(config, "TCP_KEEPINTVL", "60");
    ucp_config_modify(config, "TCP_KEEPCNT", "5");
    
    // 性能优化配置
    const char* rndv_thresh = getenv("UCX_RNDV_THRESH");
    if (rndv_thresh) {
        ucp_config_modify(config, "RNDV_THRESH", rndv_thresh);
    } else {
        ucp_config_modify(config, "RNDV_THRESH", "8192");
    }
    
    const char* tcp_rx_queue_len = getenv("UCX_TCP_RX_QUEUE_LEN");
    if (tcp_rx_queue_len) {
        ucp_config_modify(config, "TCP_RX_QUEUE_LEN", tcp_rx_queue_len);
    }
    
    logInfo("UCXClient: UCX configuration completed for target %s", target_ip.c_str());
}

std::unique_ptr<Endpoint> UCXClient::connect(std::string ip, uint16_t port) {
    logInfo("UCXClient: ===== STARTING CONNECTION TO %s:%d =====", ip.c_str(), port);
    
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
    
    // 使用目标IP配置UCX
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
    logInfo("UCXClient: UCP context initialized successfully");
    
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
    
    logInfo("UCXClient: Worker created successfully");
    
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
    ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr = (struct sockaddr*)&server_addr;
    ep_params.sockaddr.addrlen = sizeof(server_addr);
    
    logInfo("UCXClient: Creating endpoint to %s:%d", ip.c_str(), port);
    ucs_status_t status = ucp_ep_create(worker, &ep_params, &ep);
    if (status != UCS_OK) {
        logError("UCXClient: Failed to create endpoint: %s", ucs_status_string(status));
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    logInfo("UCXClient: Endpoint created, establishing connection...");
    
    // 改进的连接确认机制 - 发送一个简单的连接测试消息
    auto start_time = std::chrono::steady_clock::now();
    bool connection_confirmed = false;
    
    // 尝试发送一个简单的测试消息来确认连接
    for (int attempt = 0; attempt < 10; attempt++) {
        logInfo("UCXClient: Testing connection, attempt %d/10", attempt + 1);
        
        char test_msg[16] = "HELLO_SERVER";
        ucs_status_ptr_t req = ucp_tag_send_nb(ep, test_msg, sizeof(test_msg),
                                             ucp_dt_make_contig(1), 
                                             UCX_TAG_CONNECTION_TEST,
                                             ucx_send_cb);
        
        if (UCS_PTR_IS_ERR(req)) {
            logInfo("UCXClient: Connection test failed: %s", 
                      ucs_status_string(UCS_PTR_STATUS(req)));
        } else {
            // 等待发送完成
            if (UCS_PTR_IS_PTR(req)) {
                auto *ctx = static_cast<ucx_request*>(req);
                if (ctx->wait(2000)) {  // 2秒超时
                    connection_confirmed = true;
                    ucp_request_free(req);
                    logInfo("UCXClient: Connection confirmed via test message");
                    break;
                } else {
                    logInfo("UCXClient: Connection test timeout");
                    ucp_request_cancel(worker, req);
                    ucp_request_free(req);
                }
            } else {
                // 立即完成
                connection_confirmed = true;
                logInfo("UCXClient: Connection confirmed (immediate completion)");
                break;
            }
        }
        
        // 进度推进
        for (int i = 0; i < 100; i++) {
            ucp_worker_progress(worker);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // 检查总超时
        if (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count() > 20) {
            break;
        }
    }
    
    if (!connection_confirmed) {
        logError("UCXClient: Failed to confirm connection after 20 seconds");
        ucp_ep_destroy(ep);
        ucp_worker_destroy(worker);
        ucp_cleanup(context);
        return nullptr;
    }
    
    logInfo("UCXClient: Connection established and confirmed to %s:%d", ip.c_str(), port);
    
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
    
    // 等待连接稳定
    logInfo("UCXClient: Waiting for connection to stabilize...");
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    
    // 交换内存信息
    logInfo("UCXClient: Starting memory info exchange...");
    if (endpoint->exchangeMemoryInfo() != status_t::SUCCESS) {
        logError("UCXClient: Failed to exchange memory info");
        return nullptr;
    }
    
    logInfo("UCXClient: ===== CONNECTION COMPLETED SUCCESSFULLY =====");
    return endpoint;
}

} // namespace hmc

#endif // ENABLE_UCX