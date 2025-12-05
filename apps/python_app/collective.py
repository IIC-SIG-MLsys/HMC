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

    ip = pick_ip()
    base = int(os.environ.get("HMC_BASE_PORT", "25000"))

    port_ucx = base + rank
    port_rdma = base + 1000 + rank
    # ctrl tcp port must be unique per rank on same host
    ctrl_tcp_port = int(os.environ.get("HMC_CTRL_TCP_PORT", str(base + 2000))) + rank

    sess = hmc.create_session(
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        buffer_size=64 * 1024 * 1024,
        mem_type=hmc.MemoryType.CPU,
        num_chs=1,
    )

    g = hmc.collective.init_group(
        session=sess,
        group_id="ut",
        my_ip=ip,
        port_ucx=port_ucx,
        port_rdma=port_rdma,
        port_ctrl=ctrl_tcp_port,
        mem_type=hmc.MemoryType.CPU,
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        num_chs=1,
        start_servers=True,
        server_barrier=True,
    )

    dist.barrier()
    if rank == 0:
        print(
            "members:",
            [(m.rank, m.ip, m.port_ucx, m.port_rdma, m.port_ctrl, int(m.mem_type)) for m in g.members],
        )
        print("start alltoall(algo='direct', conn='auto')")

    # ---- direct alltoall test ----
    bytes_per_peer = 1 * 1024 * 1024
    total_bytes = bytes_per_peer * world

    send = torch.empty(total_bytes, dtype=torch.uint8)
    recv = torch.empty_like(send)

    for peer in range(world):
        v = (rank * 10 + peer) & 0xFF
        send[peer * bytes_per_peer : (peer + 1) * bytes_per_peer].fill_(v)

    hmc.collective.alltoall(
        g,
        send,
        recv,
        algo="direct",
        conn=hmc.ConnType.RDMA,
        # conn="auto",
        chunk_bytes=4 * 1024 * 1024,
        do_self_copy=True,
    )

    ok = True
    for peer in range(world):
        expect = (peer * 10 + rank) & 0xFF
        got = int(recv[peer * bytes_per_peer].item())
        if got != expect:
            ok = False
            print(f"[rank {rank}] MISMATCH peer={peer} expect={expect} got={got}")
            break

    dist.barrier()
    if ok:
        print(f"[rank {rank}] direct alltoall OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
