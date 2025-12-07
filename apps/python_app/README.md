# perf.py
perftest for hmc

## cpu
```
# server
python3 perf.py --role server --bind-ip 192.168.2.244 --ctrl-bind-ip 192.168.2.244 --ucx-port 2025 --rdma-port 2026 --ctrl-port 2027 --verify
# client
python3 perf.py --role client --server-ip 192.168.2.244 --ctrl-ip 192.168.2.244 --ucx-port 2025 --rdma-port 2026 --ctrl-port 2027 --sizes 64,256,1k,4k,64k,1m,4m,16m,64m
```

## gpu
```
# server
CUDA_VISIBLE_DEVICES=5 python3 perf.py --role server --bind-ip 192.168.2.244 --ctrl-bind-ip 192.168.2.244 --ucx-port 2025 --rdma-port 2026 --ctrl-port 2027 --gpu --device 0 --verify
# client（默认只测 rdma；要都测用 --gpu-conn both）
CUDA_VISIBLE_DEVICES=6 python3 perf.py --role client --server-ip 192.168.2.244 --ctrl-ip 192.168.2.244 --ucx-port 2025 --rdma-port 2026 --ctrl-port 2027 --gpu --device 0 --gpu-conn both --sizes 4k,64k,1m,4m,16m,64m
```

```
# rocm
# server
CUDA_VISIBLE_DEVICES=0 python3 perf_rocm.py --role server --bind-ip 192.168.2.254 --ctrl-bind-ip 192.168.2.254 --ucx-port 2025 --rdma-port 2026 --ctrl-port 2027 --gpu --device 0 --verify
# client（默认只测 rdma；要都测用 --gpu-conn both）
CUDA_VISIBLE_DEVICES=1 python3 perf_rocm.py --role client --server-ip 192.168.2.254 --ctrl-ip 192.168.2.254 --ucx-port 2025 --rdma-port 2026 --ctrl-port 2027 --gpu --device 0 --gpu-conn ucx --sizes 4k,64k,1m,4m,16m,64m
```

> - ucx传输gpu数据时，小数据会走put short路径导致报错，建议传输大于4K的数据。 
> - 如果ucx遇到不通问题，指定网卡：export UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1


# collective
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=4 collective.py

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc_per_node=2 collective_gpu.py
