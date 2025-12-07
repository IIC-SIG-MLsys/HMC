import os

import hmc
import torch
import torch.distributed as dist


def pick_ip() -> str:
    return os.environ.get("HMC_IP", "192.168.2.244")


def main():
    dist.init_process_group(backend="gloo")

    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", -1))  # from torchrun
    if local_rank < 0:
        raise RuntimeError("LOCAL_RANK not set; please launch with torchrun")

    device = torch.device("cuda", local_rank)

    ip = pick_ip()
    base = int(os.environ.get("HMC_BASE_PORT", "25000"))

    port_ucx = base + local_rank
    port_rdma = base + 1000 + local_rank
    ctrl_tcp_port = int(os.environ.get("HMC_CTRL_TCP_PORT", str(base + 2000))) + local_rank

    bytes_per_peer = 1 * 1024 * 1024
    total = world * bytes_per_peer

    send_buf = torch.empty(total, dtype=torch.uint8, device=device)
    recv_buf = torch.empty_like(send_buf)

    send = send_buf.view(world, bytes_per_peer)
    recv = recv_buf.view(world, bytes_per_peer)

    # HMC session：按 collective.py 的默认布局，需要至少 2*total
    sess = hmc.create_session(
        device_id=local_rank,
        buffer_size=2 * total,
        mem_type=hmc.memory_type_from_torch_tensor(send),
    )

    g = hmc.collective.init_group(
        session=sess,
        my_ip=ip,
        port_ucx=port_ucx,
        port_rdma=port_rdma,
        port_ctrl=ctrl_tcp_port,
        group_id="ut",
    )

    dist.barrier()
    if rank == 0:
        print("members:", [(m.rank, m.ip, m.port_ucx, m.port_rdma, m.port_ctrl, int(m.mem_type)) for m in g.members])
        print("start alltoall(algo='direct', conn='auto') [GPU tensors, double buffer]")

    # 填充 send（每个 rank 填自己的 send 内容）
    peers = torch.arange(world, device=device, dtype=torch.int32)
    vals = ((rank * 10 + peers) & 0xFF).to(torch.uint8)  # [world]
    send[peers] = vals[:, None].expand(world, bytes_per_peer)

    # alltoall：send -> recv
    chunk_bytes = min(4 * 1024 * 1024, total)
    hmc.collective.alltoall(
        g,
        send,
        recv,
        algo="direct",
        conn="auto",
        chunk_bytes=chunk_bytes,
        do_self_copy=True,
    )

    # verify on GPU：recv[src, 0] 应该等于 (src*10 + rank) & 0xFF
    expect = (((torch.arange(world, device=device) * 10 + rank) & 0xFF).to(torch.uint8))
    got0 = recv[:, 0]
    mism = (got0 != expect).nonzero(as_tuple=False)
    ok = mism.numel() == 0

    dist.barrier()
    if not ok:
        peer = int(mism[0].item())
        print(f"[rank {rank}] MISMATCH peer={peer} expect={int(expect[peer])} got={int(got0[peer])}")
    else:
        print(f"[rank {rank}] direct alltoall OK (GPU)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
