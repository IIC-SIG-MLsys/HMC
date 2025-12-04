#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import hmc


def _try_import_torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


# ---------------- TCP control channel helpers ----------------

def _sendall(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = sock.send(view)
        if n <= 0:
            raise ConnectionError("control socket send failed")
        view = view[n:]


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv(n - got)
        if not r:
            raise ConnectionError("control socket recv failed")
        view[got:got + len(r)] = r
        got += len(r)
    return bytes(buf)


# Message framing: [u32 len][payload]
def ctrl_send_msg(sock: socket.socket, payload: bytes) -> None:
    _sendall(sock, struct.pack("!I", len(payload)) + payload)


def ctrl_recv_msg(sock: socket.socket) -> bytes:
    (n,) = struct.unpack("!I", _recvall(sock, 4))
    return _recvall(sock, n)


# ---------------- Perf logic ----------------

@dataclass
class Result:
    conn: str
    size: int
    iters: int
    seconds: float
    gbps: float
    mib_s: float
    avg_rtt_us: float


def mib_per_s(bytes_total: float, seconds: float) -> float:
    return (bytes_total / seconds) / (1024.0 * 1024.0)


def gbps(bytes_total: float, seconds: float) -> float:
    return (bytes_total * 8.0) / seconds / 1e9


def make_payload(size: int, fill: int = 0x41) -> bytes:
    return bytes([fill]) * size


def parse_sizes(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        p = part.strip().lower()
        if not p:
            continue
        mult = 1
        if p.endswith("k"):
            mult = 1024
            p = p[:-1]
        elif p.endswith("m"):
            mult = 1024 * 1024
            p = p[:-1]
        elif p.endswith("g"):
            mult = 1024 * 1024 * 1024
            p = p[:-1]
        out.append(int(float(p) * mult))
    return out


def _accept_ctrl(ctrl_bind_ip: str, ctrl_port: int) -> Tuple[socket.socket, Tuple[str, int], socket.socket]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((ctrl_bind_ip, ctrl_port))
    s.listen(1)
    conn, addr = s.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s, addr, conn


def _torch_gpu_ptr(t) -> int:
    # torch on ROCm still uses torch.cuda APIs; device backend is HIP.
    if not getattr(t, "is_cuda", False):
        raise ValueError("tensor is not on GPU (torch.cuda/ROCm)")
    if not t.is_contiguous():
        t = t.contiguous()
    return int(t.data_ptr())


def _ensure_gpu_tensor(torch, device: int, need_bytes: int, buf):
    """
    Reuse a single GPU uint8 tensor big enough (>= need_bytes).
    Returns (tensor, ptr).
    """
    if buf is None or int(buf.numel()) < int(need_bytes):
        buf = torch.empty((int(need_bytes),), device=f"cuda:{device}", dtype=torch.uint8)
    return buf, _torch_gpu_ptr(buf)


def run_server(
    bind_ip: str,
    ucx_port: int,
    rdma_port: int,
    ctrl_bind_ip: str,
    ctrl_port: int,
    buffer_size: int,
    verify: bool,
    gpu: bool,
    device: int,
) -> None:
    torch = _try_import_torch()
    if gpu and torch is None:
        raise SystemExit("GPU mode requires torch (ROCm)")

    mem_type = hmc.MemoryType.AMD_GPU if gpu else hmc.MemoryType.CPU
    sess = hmc.create_session(
        device_id=device,
        buffer_size=buffer_size,
        mem_type=mem_type,
        num_chs=1,
    )

    # Start servers on different ports
    sess.init_server(bind_ip, ucx_port, conn=hmc.ConnType.UCX)
    sess.init_server(bind_ip, rdma_port, conn=hmc.ConnType.RDMA)

    # Control socket
    s, addr, conn = _accept_ctrl(ctrl_bind_ip, ctrl_port)

    print(f"[server] UCX  listening on {bind_ip}:{ucx_port}")
    print(f"[server] RDMA listening on {bind_ip}:{rdma_port}")
    print(f"[server] CTRL listening on {ctrl_bind_ip}:{ctrl_port}")
    print(f"[server] CTRL accepted from {addr[0]}:{addr[1]}")

    current_mode: Optional[str] = None
    current_conn: Optional[hmc.ConnType] = None

    # verification buffers
    head_cpu = bytearray(64)
    head_gpu = None  # torch tensor on GPU

    # sequence for DONE messages
    last_done_seq: int = 0

    try:
        hello = ctrl_recv_msg(conn).decode(errors="ignore")
        print(f"[server] {hello}")

        while True:
            msg = ctrl_recv_msg(conn)

            if msg == b"BYE":
                ctrl_send_msg(conn, b"OK")
                break

            if msg.startswith(b"MODE "):
                m = msg.split(b" ", 1)[1].decode(errors="ignore")
                if m == "ucx":
                    current_mode, current_conn = "ucx", hmc.ConnType.UCX
                elif m == "rdma":
                    current_mode, current_conn = "rdma", hmc.ConnType.RDMA
                else:
                    ctrl_send_msg(conn, b"ERR_BAD_MODE")
                    continue
                last_done_seq = 0
                print(f"[server] switch mode -> {current_mode}")
                ctrl_send_msg(conn, b"OK")
                continue

            if msg.startswith(b"DONE "):
                # DONE <size> <seq>
                if current_conn is None:
                    ctrl_send_msg(conn, b"ERR_NO_MODE")
                    continue
                parts = msg.split()
                if len(parts) != 3:
                    ctrl_send_msg(conn, b"ERR_BAD_DONE")
                    continue
                try:
                    sz = int(parts[1])
                    seq = int(parts[2])
                except Exception:
                    ctrl_send_msg(conn, b"ERR_BAD_DONE")
                    continue

                if seq != last_done_seq + 1:
                    print(f"[server] WARN: DONE seq jump: got={seq} expected={last_done_seq + 1} (size={sz})")
                last_done_seq = seq

                # If verify is enabled, touch local ConnBuffer before ACK (forces a small read path).
                if verify:
                    n = min(64, sz)
                    if gpu:
                        head_gpu, head_ptr = _ensure_gpu_tensor(torch, device, n, head_gpu)
                        sess.buf.buffer_get_gpu_ptr(head_ptr, nbytes=int(n), offset=0)
                        _ = int(head_gpu[0].item()) if n > 0 else 0
                    else:
                        sess.buf.buffer_get_cpu_into(head_cpu, nbytes=int(n), offset=0)
                        _ = head_cpu[0] if n > 0 else 0

                ctrl_send_msg(conn, b"ACK")
                continue

            if msg.startswith(b"CASE_END "):
                if current_conn is None:
                    ctrl_send_msg(conn, b"ERR_NO_MODE")
                    continue
                try:
                    sz = int(msg.split(b" ", 1)[1])
                except Exception:
                    ctrl_send_msg(conn, b"ERR_BAD_CASE")
                    continue

                if verify:
                    # Verify by reading LOCAL ConnBuffer only (no read_from remote!)
                    n = min(64, sz)
                    if gpu:
                        head_gpu, head_ptr = _ensure_gpu_tensor(torch, device, n, head_gpu)
                        sess.buf.buffer_get_gpu_ptr(head_ptr, nbytes=int(n), offset=0)
                        ok = bool((head_gpu[:n].cpu() == 0x41).all().item())
                    else:
                        sess.buf.buffer_get_cpu_into(head_cpu, nbytes=int(n), offset=0)
                        ok = all(b == 0x41 for b in head_cpu[:n])

                    print(f"[server] verify size={sz} mode={current_mode}: {'PASS' if ok else 'FAIL'}")

                ctrl_send_msg(conn, b"OK")
                continue

            ctrl_send_msg(conn, b"ERR_UNKNOWN")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass
        sess.close_server()


def run_client(
    server_ip: str,
    ucx_port: int,
    rdma_port: int,
    ctrl_ip: str,
    ctrl_port: int,
    buffer_size: int,
    sizes: List[int],
    iters: int,
    warmup: int,
    gpu: bool,
    device: int,
    gpu_conn: str,
) -> List[Result]:
    torch = _try_import_torch()
    if gpu and torch is None:
        raise SystemExit("GPU mode requires torch (ROCm)")

    mem_type = hmc.MemoryType.AMD_GPU if gpu else hmc.MemoryType.CPU
    sess = hmc.create_session(
        device_id=device,
        buffer_size=buffer_size,
        mem_type=mem_type,
        num_chs=1,
    )

    # Connect control channel first
    ctrl = socket.create_connection((ctrl_ip, ctrl_port))
    ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ctrl_send_msg(ctrl, f"HELLO client={socket.gethostname()} server={server_ip} gpu={int(gpu)}".encode())

    results: List[Result] = []

    # Transport cases (each uses its own port)
    all_cases: List[Tuple[str, hmc.ConnType, int]] = [
        ("rdma", hmc.ConnType.RDMA, rdma_port),
        ("ucx", hmc.ConnType.UCX, ucx_port),
    ]
    if gpu:
        if gpu_conn == "rdma":
            cases = [c for c in all_cases if c[0] == "rdma"]
        elif gpu_conn == "ucx":
            cases = [c for c in all_cases if c[0] == "ucx"]
        else:
            cases = all_cases
    else:
        cases = all_cases

    payload = b""
    payload_size = 0

    gpu_buf = None
    gpu_ptr = 0
    gpu_size = 0

    done_seq = 0  # monotonically increasing across all sends

    for conn_name, conn_type, port in cases:
        sess.connect(server_ip, port, conn=conn_type)

        ctrl_send_msg(ctrl, f"MODE {conn_name}".encode())
        ack = ctrl_recv_msg(ctrl)
        if ack != b"OK":
            raise RuntimeError(f"server refused mode switch: {ack!r}")

        for sz in sizes:
            if sz > buffer_size:
                raise ValueError(f"payload size {sz} > buffer_size {buffer_size}")

            if gpu:
                if gpu_buf is None or gpu_size < sz:
                    gpu_buf = torch.empty((sz,), device=f"cuda:{device}", dtype=torch.uint8)
                    gpu_ptr = _torch_gpu_ptr(gpu_buf)
                    gpu_size = sz
                gpu_buf[:sz].fill_(0x41)
                sess.buf.buffer_put_gpu_ptr(gpu_ptr, nbytes=sz, offset=0)
            else:
                if payload_size != sz:
                    payload = make_payload(sz, 0x41)
                    payload_size = sz

            # warmup
            for _ in range(warmup):
                if not gpu:
                    sess.put(server_ip, payload, conn=conn_type, bias=0)
                else:
                    sess.write_to(server_ip, 0, sz, conn=conn_type)

                done_seq += 1
                ctrl_send_msg(ctrl, f"DONE {sz} {done_seq}".encode())
                _ = ctrl_recv_msg(ctrl)  # ACK

            # timed
            t0 = time.perf_counter()
            rtt_sum = 0.0

            for _ in range(iters):
                iter0 = time.perf_counter()

                if not gpu:
                    sess.put(server_ip, payload, conn=conn_type, bias=0)
                else:
                    # strict correctness: re-put pointer region each iter (keeps local buffer hot)
                    sess.buf.buffer_put_gpu_ptr(gpu_ptr, nbytes=sz, offset=0)
                    sess.write_to(server_ip, 0, sz, conn=conn_type)

                done_seq += 1
                ctrl_send_msg(ctrl, f"DONE {sz} {done_seq}".encode())
                _ = ctrl_recv_msg(ctrl)  # ACK

                iter1 = time.perf_counter()
                rtt_sum += (iter1 - iter0)

            t1 = time.perf_counter()
            elapsed = t1 - t0
            total_bytes = float(sz * iters)

            res = Result(
                conn=conn_name,
                size=sz,
                iters=iters,
                seconds=elapsed,
                gbps=gbps(total_bytes, elapsed),
                mib_s=mib_per_s(total_bytes, elapsed),
                avg_rtt_us=(rtt_sum / iters) * 1e6,
            )
            results.append(res)

            print(
                f"[client] {conn_name} port={port} size={sz} iters={iters}  "
                f"{res.mib_s:.2f} MiB/s  {res.gbps:.2f} Gbps"
            )

            ctrl_send_msg(ctrl, f"CASE_END {sz}".encode())
            _ = ctrl_recv_msg(ctrl)

        sess.disconnect(server_ip, conn=conn_type)

    ctrl_send_msg(ctrl, b"BYE")
    _ = ctrl_recv_msg(ctrl)
    ctrl.close()
    return results


def print_results(results: List[Result]) -> None:
    def fmt_size(n: int) -> str:
        if n >= 1024 * 1024:
            return f"{n / (1024*1024):.0f} MiB"
        if n >= 1024:
            return f"{n / 1024:.0f} KiB"
        return f"{n} B"

    print("\n=== Results ===")
    print("conn   size     iters    MiB/s       Gbps      avg RTT (us)")
    print("-----  -------  ------  ----------  ---------  ------------")
    for r in results:
        print(
            f"{r.conn:<5}  {fmt_size(r.size):<7}  {r.iters:<6}  "
            f"{r.mib_s:>10.2f}  {r.gbps:>9.2f}  {r.avg_rtt_us:>12.2f}"
        )

    print("\n=== UCX vs RDMA (MiB/s ratio) ===")
    by_size = {}
    for r in results:
        by_size.setdefault(r.size, {})[r.conn] = r
    for sz in sorted(by_size.keys()):
        e = by_size[sz]
        if "ucx" in e and "rdma" in e:
            ratio = e["ucx"].mib_s / max(e["rdma"].mib_s, 1e-9)
            print(f"size={sz:>10}  ucx/rdma = {ratio:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="HMC UCX vs RDMA perf benchmark (ROCm GPU/CPU, separate ports)")
    ap.add_argument("--role", choices=["server", "client"], required=True)

    ap.add_argument("--bind-ip", default="0.0.0.0", help="HMC bind IP (server role)")
    ap.add_argument("--server-ip", default="", help="Server IP (client role)")

    ap.add_argument("--ucx-port", type=int, default=2025, help="UCX HMC port")
    ap.add_argument("--rdma-port", type=int, default=2026, help="RDMA HMC port")

    ap.add_argument("--ctrl-bind-ip", default="0.0.0.0", help="Control bind IP (server role)")
    ap.add_argument("--ctrl-ip", default="", help="Control server IP (client role)")
    ap.add_argument("--ctrl-port", type=int, default=2027, help="TCP control port")

    ap.add_argument("--buffer-size", type=int, default=128 * 1024 * 1024, help="ConnBuffer size bytes")
    ap.add_argument("--sizes", default="64,256,1k,4k,64k,1m,4m,16m,64m", help="Comma sizes (e.g. 1k,4k,1m)")
    ap.add_argument("--iters", type=int, default=200, help="Iterations per size")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations per size")
    ap.add_argument("--verify", action="store_true", help="Server verifies payload header once per size")

    ap.add_argument("--gpu", action="store_true", help="Enable AMD GPU path (torch ROCm)")
    ap.add_argument("--device", type=int, default=0, help="GPU device id")
    ap.add_argument("--gpu-conn", choices=["rdma", "ucx", "both"], default="rdma",
                    help="In GPU mode: which transport(s) to test. Default rdma (safer).")

    args = ap.parse_args()

    if args.role == "server":
        run_server(
            bind_ip=args.bind_ip,
            ucx_port=args.ucx_port,
            rdma_port=args.rdma_port,
            ctrl_bind_ip=args.ctrl_bind_ip,
            ctrl_port=args.ctrl_port,
            buffer_size=args.buffer_size,
            verify=args.verify,
            gpu=args.gpu,
            device=args.device,
        )
        return

    if not args.server_ip:
        raise SystemExit("--server-ip is required for client role")
    if not args.ctrl_ip:
        raise SystemExit("--ctrl-ip is required for client role")

    sizes = parse_sizes(args.sizes)
    results = run_client(
        server_ip=args.server_ip,
        ucx_port=args.ucx_port,
        rdma_port=args.rdma_port,
        ctrl_ip=args.ctrl_ip,
        ctrl_port=args.ctrl_port,
        buffer_size=args.buffer_size,
        sizes=sizes,
        iters=args.iters,
        warmup=args.warmup,
        gpu=args.gpu,
        device=args.device,
        gpu_conn=args.gpu_conn,
    )
    print_results(results)


if __name__ == "__main__":
    main()
