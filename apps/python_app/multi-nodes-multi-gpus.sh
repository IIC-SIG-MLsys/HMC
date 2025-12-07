export MASTER_ADDR=192.168.2.244
export MASTER_PORT=29500

export NNODES=2
export NODE_RANK=0

# 只暴露 5,6 两张卡给 torchrun，本机 LOCAL_RANK=0->5，LOCAL_RANK=1->6
export CUDA_VISIBLE_DEVICES=5,6

# HMC 用本机可被 B 访问的 IP
export HMC_IP=192.168.2.244
export HMC_BASE_PORT=25000

torchrun \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --nproc_per_node=2 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  collective_gpu.py
