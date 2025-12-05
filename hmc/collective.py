# collective.py
from __future__ import annotations

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
    port_ctrl: int
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
            "port_ctrl": int(self.port_ctrl),
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
            port_ctrl=int(d["port_ctrl"]),
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
    port_ctrl: int = 0,
    ctrl_tcp_port: Optional[int] = None,
    ctrl_uds_dir: str = "/tmp",
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
        port_ctrl=int(port_ctrl),
        mem_type=int(mem_type),
        device_id=int(device_id),
        buffer_size=int(session.buf.size),
        num_chs=int(num_chs),
    )

    gathered: list[Optional[dict[str, Any]]] = [None for _ in range(world)]
    dist.all_gather_object(gathered, me.to_dict())
    members = [GroupMember.from_dict(x) for x in gathered]  # type: ignore[arg-type]

    g = Group(session=session, members=members, rank=rank, group_id=group_id)

    if start_servers:
        session.set_local_ip(str(my_ip))
        tcp_port = int(ctrl_tcp_port if ctrl_tcp_port is not None else me.port_ctrl)

        session.init_server(
            bind_ip=str(my_ip),
            ucx_port=int(port_ucx),
            rdma_port=int(port_rdma),
            ctrl_tcp_port=int(tcp_port),
            self_id=rank,
            ctrl_uds_dir=str(ctrl_uds_dir),
        )

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
        return int(x.numel() * x.element_size())
    return len(memoryview(x).cast("B"))


def _pick_conn(a: MemoryType, b: MemoryType) -> ConnType:
    a = MemoryType(int(a))
    b = MemoryType(int(b))

    UCX_GPU = {MemoryType.NVIDIA_GPU, MemoryType.AMD_GPU}
    RDMA_GPU = {MemoryType.CAMBRICON_MLU, MemoryType.MOORE_GPU}
    GPU = UCX_GPU | RDMA_GPU

    if a == b:
        if a == MemoryType.CPU:
            return ConnType.UCX
        if a in UCX_GPU:
            return ConnType.UCX
        if a in RDMA_GPU:
            return ConnType.RDMA
        return ConnType.UCX

    if a == MemoryType.CPU or b == MemoryType.CPU:
        other = b if a == MemoryType.CPU else a
        if other in RDMA_GPU:
            return ConnType.RDMA
        return ConnType.UCX

    if (a in GPU) and (b in GPU):
        if (a in RDMA_GPU) or (b in RDMA_GPU):
            return ConnType.RDMA
        return ConnType.RDMA

    return ConnType.UCX


def _conn_for_pair(group: Group, a_rank: int, b_rank: int, conn: ConnType | str) -> ConnType:
    if conn != "auto":
        return ConnType(conn)
    a = group.peer(a_rank).memory_type
    b = group.peer(b_rank).memory_type
    return _pick_conn(a, b)


def _connect_peer(group: Group, peer_rank: int, conn: ConnType) -> None:
    sess = group.session
    me = group.me
    peer = group.peer(peer_rank)
    sess.connect(
        peer_id=int(peer.rank),
        self_id=int(me.rank),
        peer_ip=str(peer.ip),
        data_port=int(peer.port_for(conn)),
        ctrl_tcp_port=int(peer.port_ctrl),
        conn=conn,
    )


def _copy_self_if_needed(group: Group, *, send_base: int, recv_base: int, bytes_per_peer: int, do_self_copy: bool) -> None:
    if not do_self_copy or bytes_per_peer <= 0:
        return
    sess = group.session
    r = int(group.rank)
    sess.buf.buffer_copy_within(
        dst_offset=int(recv_base + r * bytes_per_peer),
        src_offset=int(send_base + r * bytes_per_peer),
        nbytes=int(bytes_per_peer),
    )


def _put_chunked(
    sess: Session,
    ip: str,
    port: int,
    *,
    local_off: int,
    remote_off: int,
    nbytes: int,
    chunk_bytes: int,
    conn: ConnType,
) -> None:
    moved = 0
    while moved < nbytes:
        n = min(int(chunk_bytes), int(nbytes - moved))
        sess.put_remote(
            ip,
            int(port),
            local_off=int(local_off + moved),
            remote_off=int(remote_off + moved),
            nbytes=int(n),
            conn=conn,
        )
        moved += n


def _get_chunked(
    sess: Session,
    ip: str,
    port: int,
    *,
    local_off: int,
    remote_off: int,
    nbytes: int,
    chunk_bytes: int,
    conn: ConnType,
) -> None:
    moved = 0
    while moved < nbytes:
        n = min(int(chunk_bytes), int(nbytes - moved))
        sess.get_remote(
            ip,
            int(port),
            local_off=int(local_off + moved),
            remote_off=int(remote_off + moved),
            nbytes=int(n),
            conn=conn,
        )
        moved += n


def _direct_alltoall_core(
    group: Group,
    *,
    send_base: int,
    recv_base: int,
    bytes_per_peer: int,
    chunk_bytes: int,
    do_self_copy: bool,
    conn: ConnType | str = "auto",
) -> None:
    sess = group.session
    r = int(group.rank)
    w = int(group.world)

    if w <= 1:
        _copy_self_if_needed(group, send_base=send_base, recv_base=recv_base, bytes_per_peer=bytes_per_peer, do_self_copy=do_self_copy)
        return

    # connect to all peers (may vary per peer if conn="auto")
    for peer in range(w):
        if peer == r:
            continue
        c = _conn_for_pair(group, r, peer, conn)
        _connect_peer(group, peer, c)

    _copy_self_if_needed(group, send_base=send_base, recv_base=recv_base, bytes_per_peer=bytes_per_peer, do_self_copy=do_self_copy)

    for dst in range(w):
        if dst == r:
            continue

        c = _conn_for_pair(group, r, dst, conn)
        peer = group.peer(dst)
        peer_ip = str(peer.ip)
        peer_port = int(peer.port_for(c))

        local_src = int(send_base + dst * bytes_per_peer)
        remote_dst = int(recv_base + r * bytes_per_peer)

        _put_chunked(
            sess,
            peer_ip,
            peer_port,
            local_off=local_src,
            remote_off=remote_dst,
            nbytes=int(bytes_per_peer),
            chunk_bytes=int(chunk_bytes),
            conn=c,
        )

    import torch.distributed as dist  # type: ignore
    dist.barrier()


def _ring_alltoall_core(
    group: Group,
    *,
    send_base: int,
    recv_base: int,
    bytes_per_peer: int,
    chunk_bytes: int,
    do_self_copy: bool,
    conn: ConnType | str = "auto",
) -> None:
    sess = group.session
    r = int(group.rank)
    w = int(group.world)

    if w <= 1:
        _copy_self_if_needed(group, send_base=send_base, recv_base=recv_base, bytes_per_peer=bytes_per_peer, do_self_copy=do_self_copy)
        return

    right_rank = int(group.right.rank)
    left_rank = int(group.left.rank)

    conn_right = _conn_for_pair(group, r, right_rank, conn)
    conn_left = _conn_for_pair(group, r, left_rank, conn)

    _connect_peer(group, right_rank, conn_right)
    _connect_peer(group, left_rank, conn_left)

    right = group.peer(right_rank)
    left = group.peer(left_rank)
    right_ip, right_port = str(right.ip), int(right.port_for(conn_right))
    left_ip, left_port = str(left.ip), int(left.port_for(conn_left))

    _copy_self_if_needed(group, send_base=send_base, recv_base=recv_base, bytes_per_peer=bytes_per_peer, do_self_copy=do_self_copy)

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

            # left.get -> me.recv[recv_peer]
            sess.get_remote(
                left_ip,
                left_port,
                local_off=int(my_recv_off + moved),
                remote_off=int(src_off_on_left + moved),
                nbytes=int(n),
                conn=conn_left,
            )
            # me.send[send_peer] -> right.recv[recv_peer]
            sess.put_remote(
                right_ip,
                right_port,
                local_off=int(send_off + moved),
                remote_off=int(dst_off_on_right + moved),
                nbytes=int(n),
                conn=conn_right,
            )

            moved += n

        sess.ctrl_send(right_rank, int(s))
        tag = sess.ctrl_recv(left_rank)
        if int(tag) != int(s):
            raise RuntimeError(f"ring step tag mismatch: expect={s} got={tag}")


def alltoall(
    group: Group,
    send: Any,
    recv: Any,
    *,
    algo: str = "auto",               # "auto" | "direct" | "ring"
    conn: ConnType | str = "auto",    # "auto" | ConnType
    send_bias: int = 0,
    recv_bias: Optional[int] = None,
    chunk_bytes: int = 4 * 1024 * 1024,
    do_self_copy: bool = True,
) -> None:
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

    if recv_bias is None:
        recv_bias = int(send_bias) + total

    need = int(recv_bias) + total
    if need > int(sess.buf.size):
        raise ValueError(f"ConnBuffer too small: need={need} have={sess.buf.size}")

    if algo == "auto":
        algo = "direct" if w <= 8 else "ring"
    algo = algo.lower().strip()

    sess.buf.put(send, nbytes=total, offset=int(send_bias), device=None)

    if algo == "direct":
        _direct_alltoall_core(
            group,
            send_base=int(send_bias),
            recv_base=int(recv_bias),
            bytes_per_peer=int(bytes_per_peer),
            chunk_bytes=int(chunk_bytes),
            do_self_copy=bool(do_self_copy),
            conn=conn,
        )
    elif algo == "ring":
        _ring_alltoall_core(
            group,
            send_base=int(send_bias),
            recv_base=int(recv_bias),
            bytes_per_peer=int(bytes_per_peer),
            chunk_bytes=int(chunk_bytes),
            do_self_copy=bool(do_self_copy),
            conn=conn,
        )
    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    sess.buf.get_into(recv, nbytes=total, offset=int(recv_bias), device=None)
