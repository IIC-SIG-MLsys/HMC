/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#include "net_ucx.h"

#ifdef ENABLE_UCX

namespace hmc {

UCXEndpoint::UCXEndpoint(ucp_context_h context, ucp_worker_h worker, ucp_ep_h ep, 
                        void* local_buf, size_t buf_size)
    : context(context), worker(worker), ep(ep), buffer(local_buf), buffer_size(buf_size), 
      is_connected(true), memh(nullptr), remote_rkey(nullptr), 
      mem_registered(false), rkey_exchanged(false) {
    logInfo("UCXEndpoint created with buffer %p, size %zu", buffer, buffer_size);
    
    // 初始化内存信息结构
    memset(&local_mem_info, 0, sizeof(local_mem_info));
    memset(&remote_mem_info, 0, sizeof(remote_mem_info));
}

UCXEndpoint::~UCXEndpoint() {
    closeEndpoint();
}

ucp_worker_h UCXEndpoint::getWorker() const {
    return worker;
}

// UCX标签化接收 - 真正的recvData实现
status_t UCXEndpoint::recvData(size_t data_bias, size_t size) {
    logDebug("UCXEndpoint: Starting blocking receive of size %zu at bias %zu", size, data_bias);
    
    // 发起非阻塞操作
    status_t ret = recvDataNB(data_bias, size);
    if (ret != status_t::SUCCESS) {
        return ret;
    }
    
    // 等待所有挂起的请求完成
    ret = waitAllPendingRequests();
    if (ret == status_t::SUCCESS) {
        logInfo("UCXEndpoint: Blocking receive operation completed successfully");
    }
    
    return ret;
}

status_t UCXEndpoint::recvDataNB(size_t data_bias, size_t size) {
    if (data_bias + size > buffer_size) {
        logError("UCXEndpoint: Invalid data bias and size");
        return status_t::ERROR;
    }

    if (!is_connected.load()) {
        logError("UCXEndpoint: Endpoint is closed");
        return status_t::ERROR;
    }

    logDebug("UCXEndpoint: Starting non-blocking receive of size %zu at bias %zu", size, data_bias);

    // 获取接收地址
    void* recv_addr = getLocalAddress(data_bias);
    
    // 创建标签掩码，确保我们只接收数据消息
    ucp_tag_t tag_mask = ~0ULL;
    
    // 使用标签化接收数据（非阻塞）
    ucs_status_ptr_t req = ucp_tag_recv_nb(worker, recv_addr, size,
                                          ucp_dt_make_contig(1),
                                          UCX_TAG_DATA, tag_mask,
                                          ucx_recv_cb);
                                          
    if (UCS_PTR_IS_ERR(req)) {
        logError("UCXEndpoint: Failed to post receive: %s", 
                ucs_status_string(UCS_PTR_STATUS(req)));
        return status_t::ERROR;
    }

    // 如果操作没有立即完成，保存请求句柄供后续poll使用
    if (UCS_PTR_IS_PTR(req)) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        pending_requests.push_back(req);
        logDebug("UCXEndpoint: Added receive request to pending list, total pending: %zu", 
                pending_requests.size());
    } else {
        // UCS_OK - 操作立即完成
        logDebug("UCXEndpoint: Receive operation completed immediately");
    }

    return status_t::SUCCESS;
}

// 真正的非阻塞接口 - 只发起操作，不等待完成
status_t UCXEndpoint::writeDataNB(size_t data_bias, size_t size) {
    if (data_bias + size > buffer_size) {
        logError("UCXEndpoint: Invalid data bias and size");
        return status_t::ERROR;
    }

    if (!is_connected.load()) {
        logError("UCXEndpoint: Endpoint is closed");
        return status_t::ERROR;
    }
    
    // 确保内存已注册且已交换密钥
    if (!mem_registered || !rkey_exchanged) {
        logInfo("UCXEndpoint: Memory info not exchanged yet, doing it now");
        if (exchangeMemoryInfo() != status_t::SUCCESS) {
            logError("UCXEndpoint: Failed to exchange memory info");
            return status_t::ERROR;
        }
    }
    
    // 验证远程内存信息有效
    if (remote_mem_info.addr == 0 || remote_mem_info.length == 0 || remote_rkey == nullptr) {
        logError("UCXEndpoint: Invalid remote memory information");
        return status_t::ERROR;
    }

    logDebug("UCXEndpoint: Starting non-blocking write of size %zu at bias %zu", size, data_bias);
    
    // 获取本地和远程地址
    void* src_addr = getLocalAddress(data_bias);
    uint64_t dst_addr = getRemoteAddress(data_bias);
    
    // 使用PUT操作写入数据（非阻塞）
    ucs_status_ptr_t req = ucp_put_nb(ep, src_addr, size, dst_addr, remote_rkey, 
                                    (ucp_send_callback_t)ucx_send_cb);
                                     
    if (UCS_PTR_IS_ERR(req)) {
        logError("UCXEndpoint: Failed to put data: %s", 
                ucs_status_string(UCS_PTR_STATUS(req)));
        return status_t::ERROR;
    }
    
    // 如果操作没有立即完成，保存请求句柄供后续poll使用
    if (UCS_PTR_IS_PTR(req)) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        pending_requests.push_back(req);
        logDebug("UCXEndpoint: Added write request to pending list, total pending: %zu", 
                pending_requests.size());
    } else {
        // UCS_OK - 操作立即完成
        logDebug("UCXEndpoint: Write operation completed immediately");
    }
    
    return status_t::SUCCESS;
}

status_t UCXEndpoint::readDataNB(size_t data_bias, size_t size) {
    if (data_bias + size > buffer_size) {
        logError("UCXEndpoint: Invalid data bias and size");
        return status_t::ERROR;
    }

    if (!is_connected.load()) {
        logError("UCXEndpoint: Endpoint is closed");
        return status_t::ERROR;
    }

    // 确保内存已注册且已交换密钥
    if (!mem_registered || !rkey_exchanged) {
        if (exchangeMemoryInfo() != status_t::SUCCESS) {
            logError("UCXEndpoint: Failed to exchange memory info");
            return status_t::ERROR;
        }
    }

    logDebug("UCXEndpoint: Starting non-blocking read of size %zu at bias %zu", size, data_bias);

    // 获取本地和远程地址
    void* dst_addr = getLocalAddress(data_bias);
    uint64_t src_addr = getRemoteAddress(data_bias);

    // 使用GET操作读取数据（非阻塞）
    ucs_status_ptr_t req = ucp_get_nb(ep, dst_addr, size, src_addr, remote_rkey, 
                                    (ucp_send_callback_t)ucx_send_cb);
                                    
    if (UCS_PTR_IS_ERR(req)) {
        logError("UCXEndpoint: Failed to get data: %s", 
                ucs_status_string(UCS_PTR_STATUS(req)));
        return status_t::ERROR;
    }

    // 如果操作没有立即完成，保存请求句柄供后续poll使用
    if (UCS_PTR_IS_PTR(req)) {
        std::lock_guard<std::mutex> lock(requests_mutex);
        pending_requests.push_back(req);
        logDebug("UCXEndpoint: Added read request to pending list, total pending: %zu", 
                pending_requests.size());
    } else {
        // UCS_OK - 操作立即完成
        logDebug("UCXEndpoint: Read operation completed immediately");
    }

    return status_t::SUCCESS;
}

// 轮询完成 - 处理指定数量的完成事件
status_t UCXEndpoint::pollCompletion(int num_completions_to_process) {
    std::lock_guard<std::mutex> lock(requests_mutex);
    
    int completed = 0;
    auto it = pending_requests.begin();
    
    while (it != pending_requests.end() && completed < num_completions_to_process) {
        ucs_status_ptr_t req = *it;
        
        // 推进worker进度
        ucp_worker_progress(worker);
        
        // 检查请求是否完成
        if (ucp_request_is_completed(req)) {
            // 检查请求状态
            ucs_status_t status = ucp_request_check_status(req);
            if (status != UCS_OK) {
                logError("UCXEndpoint: Request completed with error: %s", 
                        ucs_status_string(status));
                ucp_request_free(req);
                it = pending_requests.erase(it);
                return status_t::ERROR;
            }
            
            logDebug("UCXEndpoint: Request completed successfully");
            
            // 释放请求并从列表中移除
            ucp_request_free(req);
            it = pending_requests.erase(it);
            completed++;
        } else {
            ++it;
        }
    }
    
    logDebug("UCXEndpoint: Polled %d completions, %zu requests still pending", 
            completed, pending_requests.size());
    
    return status_t::SUCCESS;
}

// 阻塞接口 - 基于非阻塞接口实现
status_t UCXEndpoint::writeData(size_t data_bias, size_t size) {
    logDebug("UCXEndpoint: Starting blocking write of size %zu at bias %zu", size, data_bias);
    
    // 发起非阻塞操作
    status_t ret = writeDataNB(data_bias, size);
    if (ret != status_t::SUCCESS) {
        return ret;
    }
    
    // 等待所有挂起的请求完成
    ret = waitAllPendingRequests();
    if (ret == status_t::SUCCESS) {
        logInfo("UCXEndpoint: Blocking write operation completed successfully");
    }
    
    return ret;
}

status_t UCXEndpoint::readData(size_t data_bias, size_t size) {
    logDebug("UCXEndpoint: Starting blocking read of size %zu at bias %zu", size, data_bias);
    
    // 发起非阻塞操作
    status_t ret = readDataNB(data_bias, size);
    if (ret != status_t::SUCCESS) {
        return ret;
    }
    
    // 等待所有挂起的请求完成
    ret = waitAllPendingRequests();
    if (ret == status_t::SUCCESS) {
        logInfo("UCXEndpoint: Blocking read operation completed successfully");
    }
    
    return ret;
}

status_t UCXEndpoint::closeEndpoint() {
    if (!is_connected.exchange(false)) {
        // 已经关闭
        return status_t::SUCCESS;
    }

    logInfo("UCXEndpoint: Closing endpoint");

    // 等待所有挂起的请求完成
    waitAllPendingRequests();
    
    // 清理挂起的请求
    {
        std::lock_guard<std::mutex> lock(requests_mutex);
        for (auto req : pending_requests) {
            if (UCS_PTR_IS_PTR(req)) {
                ucp_request_cancel(worker, req);
                ucp_request_free(req);
            }
        }
        pending_requests.clear();
    }

    // 取消注册内存
    unregisterMemory();

    // 发送关闭请求
    ucs_status_ptr_t req = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(req)) {
        logError("UCXEndpoint: Error closing endpoint: %s", 
                ucs_status_string(UCS_PTR_STATUS(req)));
        return status_t::ERROR;
    }

    if (UCS_PTR_IS_PTR(req)) {
        // 等待关闭请求完成
        auto start_time = std::chrono::steady_clock::now();
        while (!UCS_PTR_IS_ERR(req) && !ucp_request_is_completed(req)) {
            ucp_worker_progress(worker);
            
            // 超时检查
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count() > 3000) {
                logError("UCXEndpoint: Close operation timed out");
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ucp_request_free(req);
    }

    // 清理资源 - 注意：context和worker在Server/Client中管理，这里不释放
    ep = nullptr;

    logInfo("UCXEndpoint: Endpoint closed successfully");
    return status_t::SUCCESS;
}

status_t UCXEndpoint::registerMemory() {
    if (mem_registered) {
        return status_t::SUCCESS;  // 已经注册过
    }
    
    // 设置内存映射参数
    ucp_mem_map_params_t mem_params;
    memset(&mem_params, 0, sizeof(mem_params));
    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | 
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mem_params.address = buffer;
    mem_params.length = buffer_size;
    
    // 注册内存
    ucs_status_t status = ucp_mem_map(context, &mem_params, &memh);
    if (status != UCS_OK) {
        logError("UCXEndpoint: Failed to register memory: %s", ucs_status_string(status));
        return status_t::ERROR;
    }
    
    // 打包内存密钥
    void* rkey_buffer_ptr = nullptr;
    size_t rkey_buffer_size = 0;
    
    status = ucp_rkey_pack(context, memh, &rkey_buffer_ptr, &rkey_buffer_size);
    if (status != UCS_OK) {
        logError("UCXEndpoint: Failed to pack rkey: %s", ucs_status_string(status));
        ucp_mem_unmap(context, memh);
        return status_t::ERROR;
    }
    
    // 验证rkey大小不超过我们的缓冲区
    if (rkey_buffer_size > sizeof(local_mem_info.rkey_buffer)) {
        logError("UCXEndpoint: Rkey size (%zu) exceeds buffer size (%zu)",
                rkey_buffer_size, sizeof(local_mem_info.rkey_buffer));
        ucp_rkey_buffer_release(rkey_buffer_ptr);
        ucp_mem_unmap(context, memh);
        return status_t::ERROR;
    }
    
    // 复制内存信息
    local_mem_info.addr = (uint64_t)buffer;
    local_mem_info.length = buffer_size;
    local_mem_info.rkey_size = rkey_buffer_size;
    memcpy(local_mem_info.rkey_buffer, rkey_buffer_ptr, rkey_buffer_size);
    
    // 释放rkey缓冲区
    ucp_rkey_buffer_release(rkey_buffer_ptr);
    
    mem_registered = true;
    logDebug("UCXEndpoint: Memory registered successfully, addr: %p, size: %zu", 
            buffer, buffer_size);
    
    return status_t::SUCCESS;
}

status_t UCXEndpoint::unregisterMemory() {
    if (!mem_registered) {
        return status_t::SUCCESS;  // 没有注册过
    }

    // 释放远程rkey
    if (remote_rkey != nullptr) {
        ucp_rkey_destroy(remote_rkey);
        remote_rkey = nullptr;
    }

    // 取消内存映射
    if (memh != nullptr) {
        ucs_status_t status = ucp_mem_unmap(context, memh);
        if (status != UCS_OK) {
            logError("UCXEndpoint: Failed to unregister memory: %s", 
                    ucs_status_string(status));
            return status_t::ERROR;
        }
        memh = nullptr;
    }

    mem_registered = false;
    rkey_exchanged = false;

    logDebug("UCXEndpoint: Memory unregistered successfully");
    return status_t::SUCCESS;
}

status_t UCXEndpoint::exchangeMemoryInfo() {
    if (rkey_exchanged) {
        return status_t::SUCCESS;  // 已经交换过
    }
    
    // 确保内存已注册
    if (!mem_registered) {
        logInfo("UCXEndpoint: Memory not registered yet, registering...");
        if (registerMemory() != status_t::SUCCESS) {
            return status_t::ERROR;
        }
    }
    
    logInfo("UCXEndpoint: Starting memory info exchange");
    
    // 创建发送/接收内存数据的缓冲区
    char send_buffer[sizeof(ucx_mem_info)];
    char recv_buffer[sizeof(ucx_mem_info)];
    
    // 复制本地内存信息到发送缓冲区
    memcpy(send_buffer, &local_mem_info, sizeof(ucx_mem_info));
    
    // 创建标签掩码，确保我们只接收内存信息消息
    ucp_tag_t tag_mask = ~0ULL;
    
    // 1. 发送本地内存信息
    logInfo("UCXEndpoint: Sending local memory info - addr: %p, length: %zu", 
            (void*)local_mem_info.addr, local_mem_info.length);
    
    ucs_status_ptr_t send_req = ucp_tag_send_nb(ep, send_buffer, sizeof(ucx_mem_info),
                                             ucp_dt_make_contig(1), 
                                             UCX_TAG_RKEY,
                                             ucx_send_cb);
    
    if (UCS_PTR_IS_ERR(send_req)) {
        logError("UCXEndpoint: Failed to send memory info: %s", 
                ucs_status_string(UCS_PTR_STATUS(send_req)));
        return status_t::ERROR;
    }
    
    // 等待发送完成，使用改进的等待机制
    if (UCS_PTR_IS_PTR(send_req)) {
        if (!waitRequest(send_req, 10000)) {  // 10秒超时
            logError("UCXEndpoint: Send memory info request timed out");
            return status_t::ERROR;
        }
    }
    
    logInfo("UCXEndpoint: Local memory info sent");
    
    // 2. 接收远程内存信息
    logInfo("UCXEndpoint: Waiting for remote memory info");
    
    // 准备接收
    ucs_status_ptr_t recv_req = ucp_tag_recv_nb(worker, recv_buffer, sizeof(ucx_mem_info),
                                             ucp_dt_make_contig(1),
                                             UCX_TAG_RKEY, tag_mask,
                                             ucx_recv_cb);
    
    if (UCS_PTR_IS_ERR(recv_req)) {
        logError("UCXEndpoint: Failed to post receive: %s", 
                ucs_status_string(UCS_PTR_STATUS(recv_req)));
        return status_t::ERROR;
    }
    
    // 等待接收完成，使用改进的等待机制
    bool received = false;
    
    if (UCS_PTR_IS_PTR(recv_req)) {
        if (waitRequest(recv_req, 10000)) {  // 10秒超时
            received = true;
        } else {
            logError("UCXEndpoint: Receive memory info request timed out");
            ucp_request_cancel(worker, recv_req);
            ucp_request_free(recv_req);
            return status_t::ERROR;
        }
    } else {
        // 立即完成
        received = true;
    }
    
    if (!received) {
        logError("UCXEndpoint: Failed to receive remote memory info");
        return status_t::ERROR;
    }
    
    // 复制接收到的内存信息
    memcpy(&remote_mem_info, recv_buffer, sizeof(ucx_mem_info));
    
    // 验证接收到的内存信息
    if (remote_mem_info.addr == 0 || remote_mem_info.length == 0) {
        logError("UCXEndpoint: Received invalid memory info - addr: %p, length: %zu", 
                (void*)remote_mem_info.addr, remote_mem_info.length);
        return status_t::ERROR;
    }
    
    logInfo("UCXEndpoint: Received remote memory info - addr: %p, length: %zu",
            (void*)remote_mem_info.addr, remote_mem_info.length);
    
    // 3. 解包远程内存密钥
    ucs_status_t status = ucp_ep_rkey_unpack(ep, remote_mem_info.rkey_buffer, &remote_rkey);
    if (status != UCS_OK) {
        logError("UCXEndpoint: Failed to unpack rkey: %s", 
                ucs_status_string(status));
        return status_t::ERROR;
    }
    
    rkey_exchanged = true;
    logInfo("UCXEndpoint: Memory info exchange completed successfully");
    
    return status_t::SUCCESS;
}

uint64_t UCXEndpoint::getRemoteAddress(size_t bias) const {
    return remote_mem_info.addr + bias;
}

void* UCXEndpoint::getLocalAddress(size_t bias) const {
    return static_cast<char*>(buffer) + bias;
}

bool UCXEndpoint::waitRequest(ucs_status_ptr_t req, int timeout_ms) {
    if (!UCS_PTR_IS_PTR(req)) {
        // 请求已立即完成或出错
        return !UCS_PTR_IS_ERR(req);
    }

    auto *ctx = static_cast<ucx_request*>(req);
    bool success = false;

    // 使用设置的超时时间等待请求完成
    auto start = std::chrono::steady_clock::now();
    while (true) {
        // 推进worker进度
        ucp_worker_progress(worker);
        
        // 检查请求是否完成
        {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            if (ctx->completed) {
                success = (ctx->status == UCS_OK);
                break;
            }
        }
        
        // 检查是否超时
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        if (elapsed.count() >= timeout_ms) {
            logError("UCXEndpoint: Request timed out after %d ms", timeout_ms);
            // 取消请求
            ucp_request_cancel(worker, req);
            break;
        }
        
        // 短暂休眠，减少CPU占用
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // 获取状态并释放请求
    ucs_status_t final_status;
    {
        std::unique_lock<std::mutex> lock(ctx->mutex);
        final_status = ctx->status;
    }
    ucp_request_free(req);

    if (!success && final_status != UCS_OK) {
        logError("UCXEndpoint: Request failed with status: %s", 
                ucs_status_string(final_status));
    }

    return success;
}

// 等待所有挂起请求完成的辅助函数 - 移除超时限制
status_t UCXEndpoint::waitAllPendingRequests() {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            if (pending_requests.empty()) {
                logDebug("UCXEndpoint: All pending requests completed");
                return status_t::SUCCESS;  // 所有请求都完成了
            }
            logDebug("UCXEndpoint: Still have %zu pending requests", pending_requests.size());
        }
        
        // 检查连接状态，如果连接已断开则停止等待
        if (!is_connected.load()) {
            logError("UCXEndpoint: Connection is closed, stopping wait for pending requests");
            return status_t::ERROR;
        }
        
        // 尝试处理完成的请求
        status_t poll_ret = pollCompletion(10);  // 每次处理最多10个完成的请求
        if (poll_ret != status_t::SUCCESS) {
            logError("UCXEndpoint: Error occurred while polling completions");
            return poll_ret;
        }
        
        // 短暂休眠，减少CPU占用
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void UCXEndpoint::progressWorker(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (true) {
        // 推进worker进度
        ucp_worker_progress(worker);
        
        // 检查是否超时
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        if (elapsed.count() >= timeout_ms) {
            break;
        }
        
        // 短暂休眠，减少CPU占用
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace hmc

#endif // ENABLE_UCX