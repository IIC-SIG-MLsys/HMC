# 设置环境变量
export SERVER_IP=192.168.2.243
export CLIENT_IP=192.168.2.252
export TCP_SERVER_IP=192.168.2.243


# ./build/apps/uhm_app/uhm_server --mode serial
# ./build/apps/uhm_app/uhm_server --mode uhm
# ./build/apps/uhm_app/uhm_server --mode rdma_cpu
# ./build/apps/uhm_app/uhm_server --mode g2h2g

export LD_LIBRARY_PATH=/usr/local/ucx/lib:$LD_LIBRARY_PATH
./build/apps/uhm_app/uhm_server --mode ucx