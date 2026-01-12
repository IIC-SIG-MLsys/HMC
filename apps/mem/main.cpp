#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr,                                   \
                    "CUDA error %s:%d: %s\n",                 \
                    __FILE__, __LINE__,                        \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(int argc, char **argv) {
    const int gpu_id = 6;
    const size_t buf_size = 128UL * 1024 * 1024; // 128MB

    printf("=== GPUDirect RDMA MR registration test ===\n");

    /* ------------------------------------------------------------------ */
    /* 1. Init CUDA                                                       */
    /* ------------------------------------------------------------------ */
    CHECK_CUDA(cudaSetDevice(gpu_id));
    CHECK_CUDA(cudaFree(0));  // force context creation

    int dev = -1;
    CHECK_CUDA(cudaGetDevice(&dev));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    char pci_bus_id[32];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), dev));

    printf("Using CUDA device %d\n", dev);
    printf("  Name           : %s\n", prop.name);
    printf("  PCI Bus ID     : %s\n", pci_bus_id);
    printf("  Integrated GPU : %d\n", prop.integrated);

    /* ------------------------------------------------------------------ */
    /* 2. Allocate GPU memory                                             */
    /* ------------------------------------------------------------------ */
    void *gpu_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&gpu_buf, buf_size));
    printf("Allocated GPU buffer at %p (%zu bytes)\n", gpu_buf, buf_size);

    /* ------------------------------------------------------------------ */
    /* 3. Get RDMA device list                                            */
    /* ------------------------------------------------------------------ */
    int num_devices = 0;
    ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        fprintf(stderr, "No RDMA devices found\n");
        return EXIT_FAILURE;
    }

    printf("Found %d RDMA devices\n", num_devices);

    /* ------------------------------------------------------------------ */
    /* 4. Open first RDMA device                                          */
    /* ------------------------------------------------------------------ */
    ibv_context *ctx = ibv_open_device(dev_list[0]);
    if (!ctx) {
        perror("ibv_open_device");
        return EXIT_FAILURE;
    }

    printf("Opened RDMA device: %s\n", ibv_get_device_name(dev_list[0]));

    /* ------------------------------------------------------------------ */
    /* 5. Allocate protection domain                                      */
    /* ------------------------------------------------------------------ */
    ibv_pd *pd = ibv_alloc_pd(ctx);
    if (!pd) {
        perror("ibv_alloc_pd");
        return EXIT_FAILURE;
    }

    /* ------------------------------------------------------------------ */
    /* 6. Register GPU memory                                             */
    /* ------------------------------------------------------------------ */
    int access = IBV_ACCESS_LOCAL_WRITE |
                 IBV_ACCESS_REMOTE_READ |
                 IBV_ACCESS_REMOTE_WRITE;

    ibv_mr *mr = ibv_reg_mr(pd, gpu_buf, buf_size, access);
    if (!mr) {
        perror("ibv_reg_mr (GPU memory)");
        printf("❌ GPUDirect RDMA NOT working\n");
        return EXIT_FAILURE;
    }

    printf("✅ ibv_reg_mr succeeded (GPUDirect RDMA OK)\n");
    printf("   MR lkey=0x%x rkey=0x%x\n", mr->lkey, mr->rkey);

    /* ------------------------------------------------------------------ */
    /* 7. Cleanup                                                         */
    /* ------------------------------------------------------------------ */
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    ibv_free_device_list(dev_list);
    CHECK_CUDA(cudaFree(gpu_buf));

    printf("Test completed successfully.\n");
    return 0;
}
