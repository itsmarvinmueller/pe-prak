#!/bin/bash

# Check which .env file should be used
if [ $# -ne 1 ]; then
  echo "Usage: $0 <path-to-env-file>"
  echo "Example: $0 ./configs/balanced.env"
  exit 1
fi

ENV_FILE=$1

# Check if the env file exists
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: Environment file '$ENV_FILE' not found!"
  exit 2
fi

echo "Loading environment from file: $ENV_FILE"
source "$ENV_FILE"

echo "[1/3] Setting TCP congestion control to DCTCP..."
sudo sysctl -w net.ipv4.tcp_congestion_control=dctcp

echo "[2/3] Enabling ECN in the Linux Kernel..."
sudo sysctl -w net.ipv4.tcp_ecn=1

echo "[3/3] Configuring ECN marking on Mininet switch interfaces..."

# List of Mininet switch interfaces to configure ECN marking on
# Update these if you're using different switches or interface names
SWITCH_INTERFACES=(
  "r1-eth1"  # r1 <-> s1
  "r1-eth2"  # r1 <-> s2
  "r1-eth3"  # r1 <-> r2
  "r2-eth1"  # r2 <-> r1
  "r2-eth2"  # r2 <-> s3
  "r2-eth3"  # r2 <-> s4
)

# Configure each interface using RED with ECN enabled
for iface in "${SWITCH_INTERFACES[@]}"; do
  if ip link show "$iface" > /dev/null 2>&1; then
    echo "Configuring ECN on $iface..."
    sudo tc qdisc replace dev "$iface" root red \
         limit $LIMIT min $MIN max $MAX \
         avpkt $AVPKT burst $BURST bandwidth $BANDWIDTH \
         probability $PROBABILITY ecn
  else
    echo "Interface $iface not found."
  fi
done

echo "DCTCP and ECN configuration complete."