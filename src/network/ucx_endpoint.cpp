/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
 #include "net_ucx.h"
 
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
     
     // 注册内存
     if (registerMemory() != status_t::SUCCESS) {
         logError("UCXEndpoint: Failed to register memory");
     }
 }
 
 UCXEndpoint::~UCXEndpoint() {
     closeEndpoint();
 }
 
 status_t UCXEndpoint::writeData(size_t data_bias, size_t size) {
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
    
    // 必须验证远程内存信息有效
    if (remote_mem_info.addr == 0 || remote_mem_info.length == 0 || remote_rkey == nullptr) {
        logError("UCXEndpoint: Invalid remote memory information");
        // 尝试再次交换内存信息
        if (exchangeMemoryInfo() != status_t::SUCCESS) {
            return status_t::ERROR;
        }
        
        // 再次检查
        if (remote_mem_info.addr == 0 || remote_mem_info.length == 0 || remote_rkey == nullptr) {
            logError("UCXEndpoint: Remote memory info still invalid after re-exchange");
            return status_t::ERROR;
        }
    }

    logDebug("UCXEndpoint: Writing data of size %zu at bias %zu", size, data_bias);
    
    // 获取本地和远程地址
    void* src_addr = getLocalAddress(data_bias);
    uint64_t dst_addr = getRemoteAddress(data_bias);
    
    // 使用PUT操作写入数据 - non-blocking版本
    ucs_status_ptr_t req = ucp_put_nb(ep, src_addr, size, dst_addr, remote_rkey, 
                                    (ucp_send_callback_t)ucx_send_cb);
                                     
    if (UCS_PTR_IS_ERR(req)) {
        logError("UCXEndpoint: Failed to put data: %s", 
                ucs_status_string(UCS_PTR_STATUS(req)));
        return status_t::ERROR;
    }
    
    if (UCS_PTR_IS_PTR(req)) {
        // 等待请求完成
        if (!waitRequest(req)) {
            logError("UCXEndpoint: Request wait failed for PUT operation");
            return status_t::ERROR;
        }
    }
    
    // 刷新worker以确保操作完成
    ucs_status_ptr_t flush_req = ucp_worker_flush_nb(worker, 0, ucx_send_cb);
    if (UCS_PTR_IS_ERR(flush_req)) {
        logError("UCXEndpoint: Failed to flush worker: %s", 
                ucs_status_string(UCS_PTR_STATUS(flush_req)));
        return status_t::ERROR;
    }
    
    if (UCS_PTR_IS_PTR(flush_req)) {
        if (!waitRequest(flush_req)) {
            logError("UCXEndpoint: Request wait failed for FLUSH operation");
            return status_t::ERROR;
        }
    }
    
    logInfo("UCXEndpoint: PUT operation completed successfully");
    return status_t::SUCCESS;
 }
 
 status_t UCXEndpoint::readData(size_t data_bias, size_t size) {
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
 
 logDebug("UCXEndpoint: Reading data of size %zu at bias %zu", size, data_bias);
 
 // 获取本地和远程地址
 void* dst_addr = getLocalAddress(data_bias);
 uint64_t src_addr = getRemoteAddress(data_bias);
 
 // 使用GET操作读取数据 - non-blocking版本
 ucs_status_ptr_t req = ucp_get_nb(ep, dst_addr, size, src_addr, remote_rkey, 
                                (ucp_send_callback_t)ucx_send_cb);
                                
 if (UCS_PTR_IS_ERR(req)) {
    logError("UCXEndpoint: Failed to get data: %s", 
            ucs_status_string(UCS_PTR_STATUS(req)));
    return status_t::ERROR;
 }
 
 if (UCS_PTR_IS_PTR(req)) {
    // 等待请求完成
    if (!waitRequest(req)) {
        logError("UCXEndpoint: Request wait failed for GET operation");
        return status_t::ERROR;
    }
 }
 
 // 刷新worker以确保操作完成
 ucs_status_ptr_t flush_req = ucp_worker_flush_nb(worker, 0, ucx_send_cb);
 if (UCS_PTR_IS_ERR(flush_req)) {
    logError("UCXEndpoint: Failed to flush worker: %s", 
            ucs_status_string(UCS_PTR_STATUS(flush_req)));
    return status_t::ERROR;
 }
 
 if (UCS_PTR_IS_PTR(flush_req)) {
    if (!waitRequest(flush_req)) {
        logError("UCXEndpoint: Request wait failed for FLUSH operation");
        return status_t::ERROR;
    }
 }
 
 return status_t::SUCCESS;
 }

 // Non-blocking write implementation
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

    logDebug("UCXEndpoint: Writing data non-blocking of size %zu at bias %zu", size, data_bias);
    
    // 获取本地和远程地址
    void* src_addr = getLocalAddress(data_bias);
    uint64_t dst_addr = getRemoteAddress(data_bias);
    
    // 使用PUT操作写入数据 - non-blocking版本
    ucs_status_ptr_t req = ucp_put_nb(ep, src_addr, size, dst_addr, remote_rkey, 
                                    (ucp_send_callback_t)ucx_send_cb);
                                     
    if (UCS_PTR_IS_ERR(req)) {
        logError("UCXEndpoint: Failed to put data (NB): %s", 
                ucs_status_string(UCS_PTR_STATUS(req)));
        return status_t::ERROR;
    }
    
    // 在非阻塞模式中不等待完成
    if (UCS_PTR_IS_PTR(req)) {
        ucp_request_free(req);
    }
    
    return status_t::SUCCESS;
 }

 // Non-blocking read implementation
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
    
    logDebug("UCXEndpoint: Reading data non-blocking of size %zu at bias %zu", size, data_bias);
    
    // 获取本地和远程地址
    void* dst_addr = getLocalAddress(data_bias);
    uint64_t src_addr = getRemoteAddress(data_bias);
    
    // 使用GET操作读取数据 - non-blocking版本
    ucs_status_ptr_t req = ucp_get_nb(ep, dst_addr, size, src_addr, remote_rkey, 
                                   (ucp_send_callback_t)ucx_send_cb);
                                   
    if (UCS_PTR_IS_ERR(req)) {
       logError("UCXEndpoint: Failed to get data (NB): %s", 
               ucs_status_string(UCS_PTR_STATUS(req)));
       return status_t::ERROR;
    }
    
    // 在非阻塞模式中不等待完成
    if (UCS_PTR_IS_PTR(req)) {
       ucp_request_free(req);
    }
    
    return status_t::SUCCESS;
 }

 // Universal Host Memory send (stub implementation)
 status_t UCXEndpoint::uhm_send(void *input_buffer, const size_t send_flags, MemoryType mem_type) {
    // UCX does not have direct equivalent to RDMA UHM
    logInfo("UCXEndpoint::uhm_send: RDMA-specific feature not implemented for UCX");
    return status_t::ERROR;  // Fixed: Use ERROR instead of NOT_IMPLEMENTED
 }

 // Universal Host Memory receive (stub implementation)
 status_t UCXEndpoint::uhm_recv(void *output_buffer, const size_t buffer_size,
                      size_t *recv_flags, MemoryType mem_type) {
    // UCX does not have direct equivalent to RDMA UHM
    logInfo("UCXEndpoint::uhm_recv: RDMA-specific feature not implemented for UCX");
    return status_t::ERROR;  // Fixed: Use ERROR instead of NOT_IMPLEMENTED
 }

 // Poll for completions
 status_t UCXEndpoint::pollCompletion(int num_completions_to_process) {
    // UCX handles completions differently than RDMA
    // Simply progress the worker to process completions
    for (int i = 0; i < num_completions_to_process; i++) {
        ucp_worker_progress(worker);
    }
    return status_t::SUCCESS;
 }
 
 status_t UCXEndpoint::closeEndpoint() {
 if (!is_connected.exchange(false)) {
    // 已经关闭
    return status_t::SUCCESS;
 }
 
 logInfo("UCXEndpoint: Closing endpoint");
 
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
    while (!UCS_PTR_IS_ERR(req) && !ucp_request_is_completed(req)) {
        ucp_worker_progress(worker);
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
    ucp_tag_t tag_mask = ~0ULL;  // 匹配所有位
    
    // 客户端和服务器使用相同的流程
    
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
    
    // 等待发送完成
    if (UCS_PTR_IS_PTR(send_req)) {
        while (!UCS_PTR_IS_ERR(send_req) && !ucp_request_is_completed(send_req)) {
            ucp_worker_progress(worker);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
        if (UCS_PTR_IS_ERR(send_req)) {
            logError("UCXEndpoint: Error during send: %s", 
                    ucs_status_string(UCS_PTR_STATUS(send_req)));
            return status_t::ERROR;
        }
        
        ucp_request_free(send_req);
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
    
    // 等待接收完成，增加超时机制
    bool received = false;
    int max_wait_iterations = 500;  // 总共等待5秒
    
    for (int i = 0; i < max_wait_iterations; ++i) {
        ucp_worker_progress(worker);
        
        if (UCS_PTR_IS_PTR(recv_req) && ucp_request_is_completed(recv_req)) {
            received = true;
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    if (!received) {
        logError("UCXEndpoint: Timed out waiting for remote memory info");
        ucp_request_cancel(worker, recv_req);
        ucp_request_free(recv_req);
        return status_t::ERROR;
    }
    
    // 复制接收到的内存信息
    memcpy(&remote_mem_info, recv_buffer, sizeof(ucx_mem_info));
    
    if (UCS_PTR_IS_PTR(recv_req)) {
        ucp_request_free(recv_req);
    }
    
    // 验证接收到的内存信息
    if (remote_mem_info.addr == 0 || remote_mem_info.length == 0) {
        logError("UCXEndpoint: Received invalid memory info - addr: %p, length: %zu", 
                (void*)remote_mem_info.addr, remote_mem_info.length);
        return status_t::ERROR;
    }
    
    logInfo("UCXEndpoint: Received remote memory info - addr: %p, length: %zu",
            (void*)remote_mem_info.addr, remote_mem_info.length);
    
    // 3. 解包远程内存密钥
    status_t ret = status_t::SUCCESS;
    
    ucs_status_t status = ucp_ep_rkey_unpack(ep, remote_mem_info.rkey_buffer, &remote_rkey);
    if (status != UCS_OK) {
        logError("UCXEndpoint: Failed to unpack rkey: %s", 
                ucs_status_string(status));
        ret = status_t::ERROR;
    } else {
        rkey_exchanged = true;
        logInfo("UCXEndpoint: Memory info exchange completed successfully");
    }
    
    return ret;
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
 ucs_status_t status = ctx->status;
 ucp_request_free(req);
 
 if (!success && status != UCS_OK) {
    logError("UCXEndpoint: Request failed with status: %s", 
            ucs_status_string(status));
 }
 
 return success;
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