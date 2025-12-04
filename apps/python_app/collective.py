import os
import hmc
import torch.distributed as dist


def pick_ip() -> str:
    return os.environ.get("HMC_IP", "192.168.2.244")


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    base = int(os.environ.get("HMC_BASE_PORT", "25000"))
    ip = pick_ip()

    sess = hmc.create_session(
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        buffer_size=8 * 1024 * 1024,
        mem_type=hmc.MemoryType.CPU,
        num_chs=1,
    )

    g = hmc.collective.init_group(
        session=sess,
        group_id="ut",
        my_ip=ip,
        port_ucx=base + rank,
        port_rdma=base + 1000 + rank,
        mem_type=hmc.MemoryType.CPU,
        device_id=int(os.environ.get("LOCAL_RANK", "0")),
        num_chs=1,
        start_servers=True,
        server_barrier=True,
    )

    if rank == 0:
        print("members:", [(m.rank, m.ip, m.port_ucx, m.port_rdma) for m in g.members])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
