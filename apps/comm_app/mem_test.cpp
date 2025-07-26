#include <iostream>
#include <mem.h>

using namespace hmc;

int main() {
    /* GPU memory test */
    std::cout << "Starting memory test..." << std::endl;
    
    // 使用CPU内存类型以确保兼容性
    Memory *mem_ops = new Memory(1, hmc::MemoryType::CPU);
    void *addr;
    size_t buffer_size = 1024;
    
    std::cout << "Allocating buffer of size " << buffer_size << std::endl;
    status_t alloc_status = mem_ops->allocateBuffer(&addr, buffer_size);
    
    if (alloc_status != status_t::SUCCESS || !addr) {
        std::cerr << "Failed to allocate buffer" << std::endl;
        delete mem_ops;
        return 1;
    }
    
    std::cout << "Buffer allocated at " << addr << std::endl;

    uint8_t data[] = "Hello World!\n";
    std::cout << "Copying data to buffer: " << (char*)data << std::endl;
    
    status_t write_status = mem_ops->copyHostToDevice(addr, data, sizeof(data));
    if (write_status != status_t::SUCCESS) {
        std::cerr << "Failed to copy data to buffer" << std::endl;
        mem_ops->freeBuffer(addr);
        delete mem_ops;
        return 1;
    }

    char host[1024] = {0};
    std::cout << "Reading data from buffer..." << std::endl;
    
    status_t read_status = mem_ops->copyDeviceToHost(host, addr, sizeof(data));
    if (read_status != status_t::SUCCESS) {
        std::cerr << "Failed to read data from buffer" << std::endl;
        mem_ops->freeBuffer(addr);
        delete mem_ops;
        return 1;
    }
    
    std::cout << "Memory Test Data: " << host << std::endl;

    // 清理资源
    mem_ops->freeBuffer(addr);
    delete mem_ops;
    
    std::cout << "Memory test completed successfully" << std::endl;
    return 0;
}