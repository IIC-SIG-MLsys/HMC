from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from . import ConnType, MemoryType, Session


@dataclass(frozen=True)
class GroupMember:
    rank: int
    world: int
    ip: str
    port_ucx: int
    port_rdma: int
    mem_type: int
    device_id: int
    buffer_size: int
    num_chs: int = 1

    @property
    def memory_type(self) -> MemoryType:
        return MemoryType(int(self.mem_type))

    def port_for(self, conn: ConnType) -> int:
        return int(self.port_ucx if conn == ConnType.UCX else self.port_rdma)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": int(self.rank),
            "world": int(self.world),
            "ip": str(self.ip),
            "port_ucx": int(self.port_ucx),
            "port_rdma": int(self.port_rdma),
            "mem_type": int(self.mem_type),
            "device_id": int(self.device_id),
            "buffer_size": int(self.buffer_size),
            "num_chs": int(self.num_chs),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GroupMember":
        return cls(
            rank=int(d["rank"]),
            world=int(d["world"]),
            ip=str(d["ip"]),
            port_ucx=int(d["port_ucx"]),
            port_rdma=int(d["port_rdma"]),
            mem_type=int(d["mem_type"]),
            device_id=int(d["device_id"]),
            buffer_size=int(d["buffer_size"]),
            num_chs=int(d.get("num_chs", 1)),
        )


class Group:
    def __init__(self, *, session: Session, members: list[GroupMember], rank: int, group_id: str = "default"):
        self.session = session
        self.group_id = str(group_id)
        self.members = sorted(members, key=lambda m: int(m.rank))
        self.rank = int(rank)
        self.world = len(self.members)

        if self.world <= 0:
            raise ValueError("empty group")
        if not (0 <= self.rank < self.world):
            raise ValueError(f"bad rank: {self.rank} (world={self.world})")
        for i, m in enumerate(self.members):
            if int(m.rank) != i:
                raise ValueError("members must be contiguous ranks 0..world-1")
            if int(m.world) != self.world:
                raise ValueError("member.world mismatch")

        self.me = self.members[self.rank]

    def peer(self, r: int) -> GroupMember:
        return self.members[int(r)]

    @property
    def right(self) -> GroupMember:
        return self.members[(self.rank + 1) % self.world]

    @property
    def left(self) -> GroupMember:
        return self.members[(self.rank - 1 + self.world) % self.world]
    

def init_group(
    *,
    session: Session,
    group_id: str = "default",
    my_ip: str = "127.0.0.1",
    port_ucx: int = 0,
    port_rdma: int = 0,
    mem_type: MemoryType = MemoryType.CPU,
    device_id: int = 0,
    num_chs: int = 1,
    start_servers: bool = True,
    server_barrier: bool = True,
) -> Group:
    import torch.distributed as dist  # type: ignore

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized. Call dist.init_process_group first (or use torchrun).")

    rank = int(dist.get_rank())
    world = int(dist.get_world_size())

    me = GroupMember(
        rank=rank,
        world=world,
        ip=str(my_ip),
        port_ucx=int(port_ucx),
        port_rdma=int(port_rdma),
        mem_type=int(mem_type),
        device_id=int(device_id),
        buffer_size=int(session.buf.size),
        num_chs=int(num_chs),
    )

    # exchange roster (everyone learns everyone's ip/ports/mem_type)
    gathered: list[Optional[dict[str, Any]]] = [None for _ in range(world)]
    dist.all_gather_object(gathered, me.to_dict())
    members = [GroupMember.from_dict(x) for x in gathered]  # type: ignore[arg-type]

    g = Group(session=session, members=members, rank=rank, group_id=group_id)

    # start local servers (UCX + RDMA) once, in group init
    if start_servers:
        session.init_server(my_ip, int(port_ucx), conn=ConnType.UCX)
        session.init_server(my_ip, int(port_rdma), conn=ConnType.RDMA)

    # make sure everyone's server is up before any data-plane connect
    if start_servers and server_barrier:
        dist.barrier()

    return g


def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch  # type: ignore
        return torch.is_tensor(x)
    except Exception:
        return False


def _is_numpy_array(x: Any) -> bool:
    try:
        import numpy as np  # type: ignore
        return isinstance(x, np.ndarray)
    except Exception:
        return False


def _nbytes(x: Any) -> int:
    if _is_numpy_array(x):
        return int(x.nbytes)
    if _is_torch_tensor(x):
        # torch.Tensor
        return int(x.numel() * x.element_size())
    # fallback: buffer protocol
    return len(memoryview(x).cast("B"))

def _pick_conn(a: MemoryType, b: MemoryType) -> ConnType:
    """
    Link selection rules (as you specified):

    - both NVIDIA_GPU -> UCX
    - both AMD_GPU    -> UCX
    - both CPU        -> UCX
    - both CAMBRICON  -> RDMA
    - both MOORE      -> RDMA

    - GPU vs CPU:
        NVIDIA/AMD with CPU -> UCX
        CAMBRICON/MOORE with CPU -> RDMA

    Any other mixed GPU-vendor case (e.g., NVIDIA vs AMD, NVIDIA vs CAMBRICON, ...)
    is not explicitly specified; we choose:
      - if either side is CAMBRICON or MOORE -> RDMA
      - else -> UCX
    """
    a = MemoryType(int(a))
    b = MemoryType(int(b))

    if a == MemoryType.CPU and b == MemoryType.CPU:
        return ConnType.UCX

    if a == b:
        if a in (MemoryType.NVIDIA_GPU, MemoryType.AMD_GPU):
            return ConnType.UCX
        if a in (MemoryType.CAMBRICON_MLU, MemoryType.MOORE_GPU):
            return ConnType.RDMA
        # DEFAULT or others -> conservative: UCX
        return ConnType.UCX

    # one CPU one GPU
    if a == MemoryType.CPU or b == MemoryType.CPU:
        other = b if a == MemoryType.CPU else a
        if other in (MemoryType.NVIDIA_GPU, MemoryType.AMD_GPU):
            return ConnType.UCX
        if other in (MemoryType.CAMBRICON_MLU, MemoryType.MOORE_GPU):
            return ConnType.RDMA
        return ConnType.UCX

    # mixed GPU vendors not fully specified
    if (a in (MemoryType.CAMBRICON_MLU, MemoryType.MOORE_GPU)) or (b in (MemoryType.CAMBRICON_MLU, MemoryType.MOORE_GPU)):
        return ConnType.RDMA
    return ConnType.UCX


def _ring_alltoall_core(
    group: Group,
    *,
    send_base: int,
    recv_base: int,
    bytes_per_peer: int,
    chunk_bytes: int,
    do_self_copy: bool,
) -> None:
    sess = group.session
    r = int(group.rank)
    w = int(group.world)

    if w <= 1:
        if do_self_copy and bytes_per_peer > 0:
            tmp = sess.buf.buffer_get_cpu(nbytes=int(bytes_per_peer), offset=int(send_base))
            sess.buf.buffer_put_cpu(tmp, nbytes=int(bytes_per_peer), offset=int(recv_base))
        return

    me = group.me
    right = group.right
    left = group.left

    conn_right = _pick_conn(me.memory_type, right.memory_type)
    conn_left = _pick_conn(me.memory_type, left.memory_type)

    sess.connect(right.ip, right.port_for(conn_right), conn=conn_right)
    sess.connect(left.ip, left.port_for(conn_left), conn=conn_left)

    if do_self_copy and bytes_per_peer > 0:
        sess.buf.buffer_copy_within(
            dst_offset=int(recv_base + r * bytes_per_peer),
            src_offset=int(send_base + r * bytes_per_peer),
            nbytes=int(bytes_per_peer),
        )

    for s in range(1, w):
        send_peer = (r - s + w) % w
        recv_peer = (r + s) % w

        send_off = int(send_base + send_peer * bytes_per_peer)
        dst_off_on_right = int(recv_base + recv_peer * bytes_per_peer)

        src_off_on_left = int(send_base + recv_peer * bytes_per_peer)
        my_recv_off = int(recv_base + recv_peer * bytes_per_peer)

        moved = 0
        while moved < bytes_per_peer:
            n = min(int(chunk_bytes), int(bytes_per_peer - moved))

            sess.get_remote(
                left.ip,
                local_off=my_recv_off + moved,
                remote_off=src_off_on_left + moved,
                nbytes=n,
                conn=conn_left,
            )

            sess.put_remote(
                right.ip,
                local_off=send_off + moved,
                remote_off=dst_off_on_right + moved,
                nbytes=n,
                conn=conn_right,
            )

            moved += n

        sess.ctrl_send(right.ip, int(s))
        tag = sess.ctrl_recv(left.ip)
        if int(tag) != int(s):
            raise RuntimeError(f"ring step tag mismatch: expect={s} got={tag}")


def ring_alltoall(
    group: Group,
    send: Any,
    recv: Any,
    *,
    send_bias: int = 0,
    recv_bias: Optional[int] = None,
    chunk_bytes: int = 4 * 1024 * 1024,
    do_self_copy: bool = True,
) -> None:
    """
    High-level ring all-to-all that accepts numpy arrays or torch tensors.

    Semantics:
      - send is treated as a flat byte buffer of size N.
      - N must be divisible by world; each rank sends N/world bytes to every peer.
      - recv must have the same nbytes as send.

    Staging:
      - send is staged into ConnBuffer at [send_bias, send_bias+N)
      - recv is staged from ConnBuffer at [recv_bias, recv_bias+N) into 'recv'

    Notes:
      - No raw pointer API needed.
      - IOBuffer may be CPU/GPU; send/recv may be CPU/GPU; IOBuffer.put/get_into handle it.
      - Assumes servers already started (UCX+RDMA), like your perf testcase.
    """
    sess = group.session
    w = int(group.world)

    n_send = _nbytes(send)
    n_recv = _nbytes(recv)
    if n_send != n_recv:
        raise ValueError(f"send/recv nbytes mismatch: send={n_send} recv={n_recv}")
    if w <= 0:
        raise ValueError("bad group world")
    if n_send % w != 0:
        raise ValueError(f"send nbytes must be divisible by world: nbytes={n_send} world={w}")

    total = int(n_send)
    bytes_per_peer = total // w

    # choose recv_bias default right after send region
    if recv_bias is None:
        recv_bias = int(send_bias) + int(total)

    # check connbuf capacity
    need = int(recv_bias) + int(total)
    if need > int(sess.buf.size):
        raise ValueError(f"ConnBuffer too small: need={need} have={sess.buf.size}")

    # Stage send into ConnBuffer (auto path CPU/GPU)
    sess.buf.put(send, nbytes=total, offset=int(send_bias), device=None)

    # Run ring over ConnBuffer regions
    _ring_alltoall_core(
        group,
        send_base=int(send_bias),
        recv_base=int(recv_bias),
        bytes_per_peer=int(bytes_per_peer),
        chunk_bytes=int(chunk_bytes),
        do_self_copy=bool(do_self_copy),
    )

    # Stage recv out of ConnBuffer into user buffer (auto path CPU/GPU)
    sess.buf.get_into(recv, nbytes=total, offset=int(recv_bias), device=None)
