#!/bin/bash
set -e

# ================================
# Auto RoCE Environment Fix (All ACTIVE devices)
# ================================
# Detects all ACTIVE RDMA ports, disables firewalls,
# sets MTU, and prints GID tables.
# ================================

MTU="1500"  # Use 1500 for stable tests, 9000 for max performance

echo "=== [1] Detecting ACTIVE RDMA devices ==="
found_any=false

for dev in /sys/class/infiniband/mlx5_*; do
  [ -d "$dev" ] || continue
  DEV=$(basename "$dev")

  # 遍历每个端口
  for port in $(ls $dev/ports 2>/dev/null); do
    state_file="$dev/ports/$port/state"
    state=$(cat "$state_file" 2>/dev/null || echo "DOWN")
    link_type=$(cat "$dev/ports/$port/link_layer" 2>/dev/null || echo "N/A")

    if [[ "$state" == *"ACTIVE"* ]]; then
      IFACE=$(cat $dev/ports/$port/gid_attrs/ndevs/0 2>/dev/null || true)
      if [ -n "$IFACE" ]; then
        echo "  ✔ Found ACTIVE device: $DEV (port=$port, iface=$IFACE, link=$link_type)"
        found_any=true

        echo "    → Setting MTU=$MTU on $IFACE"
        sudo ip link set dev "$IFACE" mtu "$MTU" || true
        ip link show "$IFACE" | grep mtu
      fi
    fi
  done
done

if [ "$found_any" = false ]; then
  echo "[ERROR] No ACTIVE RDMA ports found. Check cables, switch, or driver."
  exit 1
fi

echo "=== [2] Disabling firewalls (UDP 4791) ==="
sudo ufw disable 2>/dev/null || true
sudo systemctl stop firewalld 2>/dev/null || true
sudo iptables -F 2>/dev/null || true
sudo nft flush ruleset 2>/dev/null || true

echo "=== [3] Summary of RDMA device states ==="
ibv_devinfo -v | egrep 'hca_id|link_layer|port:|state|active_mtu' || true

sudo show_gids

echo "=== ✅ All ACTIVE RDMA links configured (MTU=$MTU) ==="
echo "You can test with:"
echo "  Server: ibv_rc_pingpong -d <dev> -i <port> -g <gid_index>"
echo "  Client: ibv_rc_pingpong -d <dev> -i <port> -g <gid_index> <server_ip>"
echo "  ibv_rc_pingpong -d mlx5_0 -i 1 -g 3"
echo "  ibv_rc_pingpong -d mlx5_0 -i 1 -g 3 ip"