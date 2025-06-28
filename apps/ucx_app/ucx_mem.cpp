#include <hmc.h>
#include <iostream>
#include <string.h>

using namespace hmc;

int main() {
    // Simple test for UCX memory operations
    std::cout << "Starting UCX memory test" << std::endl;

    // Create memory with CPU type to avoid Neuware linking issues
    Memory *mem_ops = new Memory(0, MemoryType::CPU);
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

    // Test data operations
    const char* data = "Hello World via UCX!";
    size_t data_size = strlen(data) + 1;
    
    std::cout << "Copying data to buffer" << std::endl;
    // Use memory operations instead of direct memcpy for consistency
    status_t write_status = mem_ops->copyHostToDevice(addr, data, data_size);
    if (write_status != status_t::SUCCESS) {
        std::cerr << "Failed to write data to buffer" << std::endl;
        mem_ops->freeBuffer(addr);
        delete mem_ops;
        return 1;
    }

    char host_buffer[1024] = {0};
    std::cout << "Reading data from buffer" << std::endl;
    // Use memory operations for reading
    status_t read_status = mem_ops->copyDeviceToHost(host_buffer, addr, data_size);
    if (read_status != status_t::SUCCESS) {
        std::cerr << "Failed to read data from buffer" << std::endl;
        mem_ops->freeBuffer(addr);
        delete mem_ops;
        return 1;
    }
    
    std::cout << "Memory Test Data: " << host_buffer << std::endl;

    // Clean up resources
    mem_ops->freeBuffer(addr);
    delete mem_ops;
    
    std::cout << "UCX memory test completed successfully" << std::endl;
    return 0;
}