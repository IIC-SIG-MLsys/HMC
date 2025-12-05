
export SERVER_IP=192.168.2.244
export CLIENT_IP=192.168.2.244
export TCP_SERVER_IP=192.168.2.244

CUDA_VISIBLE_DEVICES=5 ./build/apps/uhm_app/uhm_server --mode uhm
CUDA_VISIBLE_DEVICES=6 ./build/apps/uhm_app/uhm_client --mode uhm > uhm.log 2>&1

CUDA_VISIBLE_DEVICES=5 ./build/apps/uhm_app/uhm_server --mode g2h2g
CUDA_VISIBLE_DEVICES=6 ./build/apps/uhm_app/uhm_client --mode g2h2g > g2h2g.log 2>&1

CUDA_VISIBLE_DEVICES=5 ./build/apps/uhm_app/uhm_server --mode rdma_cpu
CUDA_VISIBLE_DEVICES=6 ./build/apps/uhm_app/uhm_client --mode rdma_cpu > rdma_cpu.log 2>&1

CUDA_VISIBLE_DEVICES=5 ./build/apps/uhm_app/uhm_server --mode serial
CUDA_VISIBLE_DEVICES=6 ./build/apps/uhm_app/uhm_client --mode serial > serial.log 2>&1

export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1 # 如果遇到连接问题，换网卡
CUDA_VISIBLE_DEVICES=5 ./build/apps/uhm_app/uhm_server --mode ucx
CUDA_VISIBLE_DEVICES=6 ./build/apps/uhm_app/uhm_client --mode ucx