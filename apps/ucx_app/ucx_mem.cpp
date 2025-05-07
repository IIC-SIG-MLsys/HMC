#include <hmc.h>
#include <iostream>
#include <string.h>

using namespace hmc;

int main() {
    // Simple test for UCX memory operations
    std::cout << "Starting UCX memory test" << std::endl;

    // Create memory with CPU type
    Memory *mem_ops = new Memory(0, MemoryType::CPU);
    void *addr;
    size_t buffer_size = 1024;
    
    std::cout << "Allocating buffer of size " << buffer_size << std::endl;
    mem_ops->allocateBuffer(&addr, buffer_size);
    
    if (!addr) {
        std::cerr << "Failed to allocate buffer" << std::endl;
        delete mem_ops;
        return 1;
    }
    
    std::cout << "Buffer allocated at " << addr << std::endl;

    // Test data operations
    const char* data = "Hello World via UCX!";
    size_t data_size = strlen(data) + 1;
    
    std::cout << "Copying data to buffer" << std::endl;
    // Direct memory copy
    memcpy(addr, data, data_size);

    char host_buffer[1024] = {0};
    std::cout << "Reading data from buffer" << std::endl;
    // Direct memory read
    memcpy(host_buffer, addr, data_size);
    
    std::cout << "Memory Test Data: " << host_buffer << std::endl;

    // Clean up resources
    mem_ops->freeBuffer(addr);
    delete mem_ops;
    
    std::cout << "UCX memory test completed successfully" << std::endl;
    return 0;
}