# __init__.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal, Optional

from . import hmc as _core  # type: ignore


class status_t(IntEnum):
    SUCCESS = int(_core.status_t.SUCCESS)
    ERROR = int(_core.status_t.ERROR)
    UNSUPPORT = int(_core.status_t.UNSUPPORT)
    INVALID_CONFIG = int(_core.status_t.INVALID_CONFIG)
    NOT_FOUND = int(_core.status_t.NOT_FOUND)
    TIMEOUT = int(_core.status_t.TIMEOUT)

    @classmethod
    def _from_core(cls, st: Any) -> "status_t":
        try:
            return cls(int(st))
        except Exception:
            return cls.ERROR


class MemoryType(IntEnum):
    DEFAULT = int(_core.MemoryType.DEFAULT)
    CPU = int(_core.MemoryType.CPU)
    NVIDIA_GPU = int(_core.MemoryType.NVIDIA_GPU)
    AMD_GPU = int(_core.MemoryType.AMD_GPU)
    CAMBRICON_MLU = int(_core.MemoryType.CAMBRICON_MLU)
    MOORE_GPU = int(_core.MemoryType.MOORE_GPU)

    @classmethod
    def _from_core(cls, x: Any) -> "MemoryType":
        return cls(int(x))


class ConnType(IntEnum):
    RDMA = int(_core.ConnType.RDMA)
    UCX = int(_core.ConnType.UCX)

    @classmethod
    def _from_core(cls, x: Any) -> "ConnType":
        return cls(int(x))


class CtrlTransport(IntEnum):
    TCP = int(_core.CtrlTransport.TCP)
    UDS = int(_core.CtrlTransport.UDS)

    @classmethod
    def _from_core(cls, x: Any) -> "CtrlTransport":
        return cls(int(x))


# Pybind-exposed types
Memory = _core.Memory
ConnBuffer = _core.ConnBuffer
Communicator = _core.Communicator
Buffer = _core.Buffer
CtrlLink = _core.CtrlLink

memory_supported = _core.memory_supported


class HMCError(RuntimeError):
    pass


class HMCStatusError(HMCError):
    def __init__(self, st: Any, msg: str = ""):
        super().__init__(f"{msg} (status={st})")
        self.status = st


def _enum_int(x: Any) -> int:
    return int(getattr(x, "value", x))


def _to_core_memory_type(x: Any) -> Any:
    if isinstance(x, _core.MemoryType):
        return x
    if isinstance(x, MemoryType):
        return _core.MemoryType(int(x))
    try:
        return _core.MemoryType(_enum_int(x))
    except Exception as e:
        raise TypeError(f"Invalid MemoryType: {x!r}") from e


def _to_core_conn_type(x: Any) -> Any:
    if isinstance(x, _core.ConnType):
        return x
    if isinstance(x, ConnType):
        return _core.ConnType(int(x))
    try:
        return _core.ConnType(_enum_int(x))
    except Exception as e:
        raise TypeError(f"Invalid ConnType: {x!r}") from e


def _to_core_ctrl_transport(x: Any) -> Any:
    if isinstance(x, _core.CtrlTransport):
        return x
    if isinstance(x, CtrlTransport):
        return _core.CtrlTransport(int(x))
    return _core.CtrlTransport(_enum_int(x))


def _ok(st: Any) -> bool:
    return int(st) == int(status_t.SUCCESS)


def _ensure_ok(st: Any, msg: str):
    if not _ok(st):
        raise HMCStatusError(status_t._from_core(st), msg)


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception:
        return None


def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _to_memoryview(x: Any) -> memoryview:
    if isinstance(x, memoryview):
        return x
    return memoryview(x)


def _is_numpy_array(x: Any) -> bool:
    np = _try_import_numpy()
    return (np is not None) and isinstance(x, np.ndarray)


def _is_torch_tensor(x: Any) -> bool:
    torch = _try_import_torch()
    return (torch is not None) and torch.is_tensor(x)


def _torch_tensor_info(t: Any) -> dict[str, Any]:
    torch = _try_import_torch()
    assert torch is not None
    return {
        "device": str(t.device),
        "dtype": str(t.dtype),
        "shape": tuple(t.shape),
        "numel": int(t.numel()),
        "element_size": int(t.element_size()),
        "nbytes": int(t.numel() * t.element_size()),
        "is_contiguous": bool(t.is_contiguous()),
        "is_cuda": bool(t.is_cuda),
    }


def _as_cpu_contiguous_bytes_view(x: Any) -> memoryview:
    if _is_numpy_array(x):
        np = _try_import_numpy()
        assert np is not None
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        mv = memoryview(x)
        return mv.cast("B") if mv.format != "B" else mv

    if _is_torch_tensor(x):
        torch = _try_import_torch()
        assert torch is not None
        if x.is_cuda:
            raise ValueError("Got a CUDA torch.Tensor in CPU path; use device='cuda' or pass a CUDA tensor.")
        if not x.is_contiguous():
            x = x.contiguous()
        np = _try_import_numpy()
        if np is None:
            mv = memoryview(x.cpu().numpy().tobytes())
            return mv.cast("B") if mv.format != "B" else mv
        mv = memoryview(x.numpy())
        return mv.cast("B") if mv.format != "B" else mv

    mv = _to_memoryview(x)
    return mv.cast("B") if mv.format != "B" else mv


def _torch_cuda_ptr_and_nbytes(t: Any, nbytes: Optional[int]) -> tuple[int, int]:
    torch = _try_import_torch()
    if torch is None or not torch.is_tensor(t):
        raise TypeError("Expected a torch.Tensor")
    if not t.is_cuda:
        raise ValueError("Tensor is not CUDA tensor.")
    if not t.is_contiguous():
        t = t.contiguous()
    ptr = int(t.data_ptr())
    n = int(t.numel() * t.element_size()) if nbytes is None else int(nbytes)
    return ptr, n


def memory_type_from_torch_tensor(t: Any, *, default: "MemoryType" = None) -> "MemoryType":
    """
    Map a torch.Tensor -> hmc.MemoryType (wrapper enum).

    - CPU tensor -> MemoryType.CPU
    - CUDA tensor -> MemoryType.NVIDIA_GPU or MemoryType.AMD_GPU (best-effort)
    - Other device types -> best-effort mapping if known, else DEFAULT (or provided default)

    Args:
        t: torch.Tensor
        default: fallback when device type is unknown; if None, uses MemoryType.DEFAULT
    """
    torch = _try_import_torch()
    if torch is None or not torch.is_tensor(t):
        raise TypeError("Expected a torch.Tensor")

    if default is None:
        default = MemoryType.DEFAULT

    dev = getattr(t, "device", None)
    dev_type = getattr(dev, "type", None)

    # CPU
    if dev_type == "cpu" or not bool(getattr(t, "is_cuda", False)) and dev_type is None:
        return MemoryType.CPU

    # CUDA / HIP
    if dev_type == "cuda" or bool(getattr(t, "is_cuda", False)):
        # Best-effort: distinguish NVIDIA vs AMD (HIP)
        hip = getattr(torch.version, "hip", None)
        cuda = getattr(torch.version, "cuda", None)

        # On ROCm builds, torch.version.hip is usually a non-empty string; torch.version.cuda is often None.
        if hip:
            return MemoryType.AMD_GPU
        if cuda:
            return MemoryType.NVIDIA_GPU

        return default

    # Cambricon MLU: some builds use device.type == "mlu"
    if dev_type == "mlu":
        return MemoryType.CAMBRICON_MLU

    # Moore Threads (MUSA) often appears as device.type == "musa"
    if dev_type == "musa":
        return MemoryType.MOORE_GPU

    return default


DeviceHint = Optional[Literal["cpu", "cuda", "ptr"]]


@dataclass
class IOBuffer:
    core: _core.ConnBuffer

    @classmethod
    def create(
        cls,
        device_id: int,
        buffer_size: int,
        mem_type: MemoryType = MemoryType.DEFAULT,
    ) -> "IOBuffer":
        return cls(_core.ConnBuffer(int(device_id), int(buffer_size), _to_core_memory_type(mem_type)))

    @property
    def ptr(self) -> int:
        return int(self.core.ptr)
    
    @property
    def mem_type(self) -> MemoryType:
        return MemoryType._from_core(self.core.mem_type)
    
    @property
    def device_id(self) -> int:
        return int(self.core.device_id)

    @property
    def size(self) -> int:
        return int(self.core.buffer_size)

    def buffer_put_cpu(self, src: Any, *, nbytes: Optional[int] = None, offset: int = 0) -> int:
        mv = _as_cpu_contiguous_bytes_view(src)
        n = len(mv) if nbytes is None else int(nbytes)
        st = self.core.writeFromCpu(mv, int(n), int(offset))
        _ensure_ok(st, "ConnBuffer.writeFromCpu failed")
        return n

    def buffer_get_cpu_into(self, dst: Any, *, nbytes: int, offset: int = 0) -> None:
        if _is_numpy_array(dst):
            if not dst.flags["C_CONTIGUOUS"]:
                raise ValueError("Destination numpy array must be C-contiguous")
            if not dst.flags["WRITEABLE"]:
                raise ValueError("Destination numpy array must be writeable")
            mv = memoryview(dst).cast("B")
        elif _is_torch_tensor(dst):
            if dst.is_cuda:
                raise ValueError("Destination is CUDA tensor; use device='cuda' or pass CUDA tensor.")
            if not dst.is_contiguous():
                raise ValueError("Destination torch CPU tensor must be contiguous")
            np = _try_import_numpy()
            if np is None:
                tmp = bytearray(int(nbytes))
                st = self.core.readToCpu(tmp, int(nbytes), int(offset))
                _ensure_ok(st, "ConnBuffer.readToCpu failed")
                import torch as _torch  # type: ignore
                dst.view(_torch.uint8)[: int(nbytes)].copy_(
                    _torch.frombuffer(tmp, dtype=_torch.uint8)[: int(nbytes)]
                )
                return
            mv = memoryview(dst.numpy()).cast("B")
        else:
            mv = _to_memoryview(dst).cast("B")
            if mv.readonly:
                raise ValueError("Destination buffer must be writeable (e.g. bytearray)")

        if len(mv) < int(nbytes):
            raise ValueError(f"Destination too small: have={len(mv)} need={int(nbytes)}")

        st = self.core.readToCpu(mv, int(nbytes), int(offset))
        _ensure_ok(st, "ConnBuffer.readToCpu failed")

    def buffer_get_cpu(self, *, nbytes: int, offset: int = 0) -> bytearray:
        out = bytearray(int(nbytes))
        st = self.core.readToCpu(out, int(nbytes), int(offset))
        _ensure_ok(st, "ConnBuffer.readToCpu failed")
        return out

    def buffer_put_gpu_ptr(self, src_ptr: int, *, nbytes: int, offset: int = 0) -> None:
        st = self.core.writeFromGpu(int(src_ptr), int(nbytes), int(offset))
        _ensure_ok(st, "ConnBuffer.writeFromGpu failed")

    def buffer_get_gpu_ptr(self, dst_ptr: int, *, nbytes: int, offset: int = 0) -> None:
        st = self.core.readToGpu(int(dst_ptr), int(nbytes), int(offset))
        _ensure_ok(st, "ConnBuffer.readToGpu failed")

    def buffer_put_cuda(self, t: Any, *, nbytes: Optional[int] = None, offset: int = 0) -> int:
        ptr, n = _torch_cuda_ptr_and_nbytes(t, nbytes)
        self.buffer_put_gpu_ptr(ptr, nbytes=n, offset=offset)
        return n

    def buffer_get_cuda(self, t: Any, *, nbytes: Optional[int] = None, offset: int = 0) -> int:
        ptr, n = _torch_cuda_ptr_and_nbytes(t, nbytes)
        self.buffer_get_gpu_ptr(ptr, nbytes=n, offset=offset)
        return n

    def buffer_copy_within(self, *, dst_offset: int, src_offset: int, nbytes: int) -> None:
        st = self.core.copyWithin(int(dst_offset), int(src_offset), int(nbytes))
        _ensure_ok(st, "ConnBuffer.copyWithin failed")

    def put(
        self,
        data: Any,
        *,
        nbytes: Optional[int] = None,
        offset: int = 0,
        device: DeviceHint = None,
    ) -> int:
        if device == "ptr":
            if not isinstance(data, int):
                raise TypeError("device='ptr' requires an int pointer")
            if nbytes is None:
                raise ValueError("device='ptr' requires nbytes")
            self.buffer_put_gpu_ptr(int(data), nbytes=int(nbytes), offset=offset)
            return int(nbytes)

        if device == "cuda" or (device is None and _is_torch_tensor(data) and bool(getattr(data, "is_cuda", False))):
            return self.buffer_put_cuda(data, nbytes=nbytes, offset=offset)

        return self.buffer_put_cpu(data, nbytes=nbytes, offset=offset)

    def get_into(
        self,
        dst: Any,
        *,
        nbytes: int,
        offset: int = 0,
        device: DeviceHint = None,
    ) -> None:
        if device == "ptr":
            if not isinstance(dst, int):
                raise TypeError("device='ptr' requires an int pointer")
            self.buffer_get_gpu_ptr(int(dst), nbytes=int(nbytes), offset=offset)
            return

        if device == "cuda" or (device is None and _is_torch_tensor(dst) and bool(getattr(dst, "is_cuda", False))):
            self.buffer_get_cuda(dst, nbytes=nbytes, offset=offset)
            return

        self.buffer_get_cpu_into(dst, nbytes=int(nbytes), offset=int(offset))


def _env_str(name: str, default: str) -> str:
    import os
    v = os.getenv(name)
    return v if v else default


def _normalize_ip(s: str) -> str:
    return str(s).strip()


def _pick_ctrl_transport(local_ip: str, peer_ip: str) -> CtrlTransport:
    t = _env_str("CTRL_TRANSPORT", "").strip().lower()
    if t:
        return CtrlTransport.UDS if t == "uds" else CtrlTransport.TCP

    li = _normalize_ip(local_ip)
    pi = _normalize_ip(peer_ip)
    if li and li != "0.0.0.0" and pi and (li == pi):
        return CtrlTransport.UDS
    return CtrlTransport.TCP


def _build_ctrl_link(
    *,
    transport: CtrlTransport,
    peer_rank: int,
    tcp_ip: str,
    tcp_port: int,
    uds_dir: str,
) -> Any:
    """
    CtrlLink.UDS needs a FULL PATH to peer's UDS file, not the directory.
    """
    link = CtrlLink()
    if transport == CtrlTransport.UDS:
        link.transport = _core.CtrlTransport.UDS
        link.uds_path = Communicator.udsPathFor(str(uds_dir), int(peer_rank))
        link.ip = ""
        link.port = 0
    else:
        link.transport = _core.CtrlTransport.TCP
        link.ip = str(tcp_ip)
        link.port = int(tcp_port)
        link.uds_path = ""
    return link


def _require_port(port: Any) -> int:
    try:
        p = int(port)
    except Exception as e:
        raise TypeError(f"port must be int-like, got {port!r}") from e
    if not (0 < p < 65536):
        raise ValueError(f"port out of range: {p}")
    return p


class Session:
    """
    - init_server(bind_ip=...): bring up data-plane + ctrl (TCP and/or UDS)
    - connect(peer_ip=...): auto-pick ctrl UDS/TCP by comparing local_ip vs peer_ip (or env override)
    """

    def __init__(self, buf: IOBuffer, num_chs: int = 1, *, local_ip: str = "", ctrl_uds_dir: str = "/tmp"):
        self.buf = buf
        self.comm = Communicator(self.buf.core, int(num_chs))
        self.local_ip: str = _normalize_ip(local_ip)
        self.ctrl_uds_dir: str = str(ctrl_uds_dir)
        self.id = None  # rank / CtrlId

    def set_local_ip(self, ip: str) -> None:
        self.local_ip = _normalize_ip(ip)

    def set_ctrl_uds_dir(self, d: str) -> None:
        self.ctrl_uds_dir = str(d)

    def init_server(
        self,
        *,
        bind_ip: str,
        ucx_port: int,
        rdma_port: int,
        ctrl_tcp_port: int,
        self_id: int,
        ctrl_uds_dir: Optional[str] = None,
    ) -> None:
        bind_ip = _normalize_ip(bind_ip)
        uds_dir = self.ctrl_uds_dir if ctrl_uds_dir is None else str(ctrl_uds_dir)
        self.ctrl_uds_dir = uds_dir
        self.id = self_id

        uds_path = Communicator.udsPathFor(str(uds_dir), int(self_id))

        _ensure_ok(
            self.comm.initServer(
                str(bind_ip),
                int(rdma_port),
                int(ctrl_tcp_port),
                str(uds_path),
                _to_core_conn_type(ConnType.RDMA),
            ),
            "Communicator.initServer RDMA failed",
        )
        _ensure_ok(
            self.comm.initServer(
                str(bind_ip),
                int(ucx_port),
                int(ctrl_tcp_port),
                str(uds_path),
                _to_core_conn_type(ConnType.UCX),
            ),
            "Communicator.initServer UCX failed",
        )

        self.local_ip = bind_ip

    def close_server(self) -> None:
        _ensure_ok(self.comm.closeServer(), "Communicator.closeServer failed")

    def connect(
        self,
        *,
        peer_id: int,
        self_id: int,
        peer_ip: str,
        data_port: int,
        ctrl_tcp_port: int,
        uds_dir: Optional[str] = None,
        conn: ConnType = ConnType.RDMA,
    ) -> None:
        peer_ip = _normalize_ip(peer_ip)
        data_port = _require_port(data_port)

        # If already connected for that (ip,port), skip
        if self.check_conn(peer_ip, data_port, conn):
            return

        transport = _pick_ctrl_transport(self.local_ip, peer_ip)
        use_dir = self.ctrl_uds_dir if uds_dir is None else str(uds_dir)

        ctrl_link = _build_ctrl_link(
            transport=transport,
            peer_rank=int(peer_id),
            tcp_ip=str(peer_ip),
            tcp_port=int(ctrl_tcp_port),
            uds_dir=str(use_dir),
        )

        _ensure_ok(
            self.comm.connectTo(
                int(peer_id),
                int(self_id),
                str(peer_ip),
                int(data_port),
                ctrl_link,
                _to_core_conn_type(conn),
            ),
            "Communicator.connectTo failed",
        )

    # ---- updated: ip + port indexed ----
    def disconnect(self, ip: str, port: int, conn: ConnType = ConnType.RDMA) -> None:
        p = _require_port(port)
        _ensure_ok(
            self.comm.disConnect(str(ip), int(p), _to_core_conn_type(conn)),
            "Communicator.disConnect failed",
        )

    def check_conn(self, ip: str, port: int, conn: ConnType = ConnType.RDMA) -> bool:
        p = _require_port(port)
        st = self.comm.checkConn(str(ip), int(p), _to_core_conn_type(conn))
        return int(st) == int(status_t.SUCCESS)

    def put_remote(
        self,
        ip: str,
        port: int,
        local_off: int,
        remote_off: int,
        nbytes: int,
        conn: ConnType = ConnType.RDMA,
    ) -> None:
        p = _require_port(port)
        _ensure_ok(
            self.comm.put(str(ip), int(p), int(local_off), int(remote_off), int(nbytes), _to_core_conn_type(conn)),
            "Communicator.put failed",
        )

    def get_remote(
        self,
        ip: str,
        port: int,
        local_off: int,
        remote_off: int,
        nbytes: int,
        conn: ConnType = ConnType.RDMA,
    ) -> None:
        p = _require_port(port)
        _ensure_ok(
            self.comm.get(str(ip), int(p), int(local_off), int(remote_off), int(nbytes), _to_core_conn_type(conn)),
            "Communicator.get failed",
        )

    def put_nb(self, ip: str, port: int, local_off: int, remote_off: int, nbytes: int, conn: ConnType = ConnType.RDMA) -> int:
        p = _require_port(port)
        wr_id_box = [0]
        st = self.comm.putNB(
            str(ip),
            int(p),
            int(local_off),
            int(remote_off),
            int(nbytes),
            wr_id_box,
            _to_core_conn_type(conn),
        )
        _ensure_ok(st, "Communicator.putNB failed")
        return int(wr_id_box[0])

    def get_nb(self, ip: str, port: int, local_off: int, remote_off: int, nbytes: int, conn: ConnType = ConnType.RDMA) -> int:
        p = _require_port(port)
        wr_id_box = [0]
        st = self.comm.getNB(
            str(ip),
            int(p),
            int(local_off),
            int(remote_off),
            int(nbytes),
            wr_id_box,
            _to_core_conn_type(conn),
        )
        _ensure_ok(st, "Communicator.getNB failed")
        return int(wr_id_box[0])

    def wait(self, wr_id: int | list[int]) -> None:
        if isinstance(wr_id, list):
            _ensure_ok(self.comm.wait([int(x) for x in wr_id]), "Communicator.wait(wr_ids) failed")
        else:
            _ensure_ok(self.comm.wait(int(wr_id)), "Communicator.wait(wr_id) failed")

    def ctrl_send(self, peer: int, tag: int) -> None:
        _ensure_ok(self.comm.ctrlSend(int(peer), int(tag)), "Communicator.ctrlSend failed")

    def ctrl_recv(self, peer: int) -> int:
        st, tag = self.comm.ctrlRecv(int(peer))
        _ensure_ok(st, "Communicator.ctrlRecv failed")
        return int(tag)

    def buffer_put_then_put(
        self,
        ip: str,
        port: int,
        data: Any,
        *,
        local_off: int = 0,
        remote_off: int = 0,
        conn: ConnType = ConnType.RDMA,
        nbytes: Optional[int] = None,
        device: DeviceHint = None,
    ) -> int:
        n = self.buf.put(data, nbytes=nbytes, offset=local_off, device=device)
        self.put_remote(ip, port, local_off=local_off, remote_off=remote_off, nbytes=n, conn=conn)
        return n

    def get_then_buffer_get_into(
        self,
        ip: str,
        port: int,
        dst: Any,
        *,
        local_off: int = 0,
        remote_off: int = 0,
        nbytes: Optional[int] = None,
        conn: ConnType = ConnType.RDMA,
        device: DeviceHint = None,
    ) -> int:
        if nbytes is None:
            if _is_numpy_array(dst):
                nbytes = int(dst.nbytes)
            elif _is_torch_tensor(dst):
                nbytes = int(_torch_tensor_info(dst)["nbytes"])
            else:
                nbytes = len(_to_memoryview(dst).cast("B"))

        self.get_remote(ip, port, local_off=local_off, remote_off=remote_off, nbytes=int(nbytes), conn=conn)
        self.buf.get_into(dst, nbytes=int(nbytes), offset=local_off, device=device)
        return int(nbytes)

    # ---- high-level RDMA-only: updated sendDataTo requires port; recvDataFrom stays ip-only ----
    def send_data_to(
        self,
        ip: str,
        port: int,
        send_buf_ptr: int,
        buf_size: int,
        buf_type: MemoryType,
        conn: ConnType = ConnType.RDMA,
    ) -> None:
        p = _require_port(port)
        _ensure_ok(
            self.comm.sendDataTo(
                str(ip),
                int(p),
                int(send_buf_ptr),
                int(buf_size),
                _to_core_memory_type(buf_type),
                _to_core_conn_type(conn),
            ),
            "Communicator.sendDataTo failed",
        )

    def recv_data_from(
        self,
        ip: str,
        port: int,
        recv_buf_ptr: int,
        buf_size: int,
        buf_type: MemoryType,
        flag_ptr: int,
        conn: ConnType = ConnType.RDMA,
    ) -> None:
        _ensure_ok(
            self.comm.recvDataFrom(
                str(ip),
                int(port),
                int(recv_buf_ptr),
                int(buf_size),
                _to_core_memory_type(buf_type),
                int(flag_ptr),
                _to_core_conn_type(conn),
            ),
            "Communicator.recvDataFrom failed",
        )


def create_session(
    *,
    device_id: int = 0,
    buffer_size: int = 128 * 1024 * 1024,
    mem_type: MemoryType = MemoryType.CPU,
    num_chs: int = 1,
    local_ip: str = "",
) -> Session:
    buf = IOBuffer.create(device_id=device_id, buffer_size=buffer_size, mem_type=mem_type)
    return Session(buf, num_chs=num_chs, local_ip=local_ip)


__all__ = [
    "status_t",
    "MemoryType",
    "ConnType",
    "CtrlTransport",
    "Memory",
    "ConnBuffer",
    "Communicator",
    "Buffer",
    "memory_supported",
    "HMCError",
    "HMCStatusError",
    "IOBuffer",
    "Session",
    "create_session",
    "memory_type_from_torch_tensor",
]

from .collective import Group, init_group, alltoall  # noqa: E402

__all__ += [
    "Group",
    "init_group",
    "alltoall",
]
