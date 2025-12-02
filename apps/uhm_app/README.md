
export SERVER_IP=192.168.2.253
export CLIENT_IP=192.168.2.253
export TCP_SERVER_IP=192.168.2.253

./build/apps/uhm_app/uhm_server --mode uhm
./build/apps/uhm_app/uhm_client --mode uhm > uhm.log 2>&1

./build/apps/uhm_app/uhm_server --mode g2h2g
./build/apps/uhm_app/uhm_client --mode g2h2g > g2h2g.log 2>&1

./build/apps/uhm_app/uhm_server --mode rdma_cpu
./build/apps/uhm_app/uhm_client --mode rdma_cpu > rdma_cpu.log 2>&1

./build/apps/uhm_app/uhm_server --mode serial
./build/apps/uhm_app/uhm_client --mode serial > serial.log 2>&1