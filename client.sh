# 设置环境变量
export SERVER_IP=192.168.2.236
export CLIENT_IP=192.168.2.249
export TCP_SERVER_IP=10.102.0.241

./build/apps/uhm_app/uhm_client --mode serial
# ./build/apps/uhm_app/uhm_client --mode uhm
# ./build/apps/uhm_app/uhm_client --mode rdma_cpu
# ./build/apps/uhm_app/uhm_client --mode g2h2g