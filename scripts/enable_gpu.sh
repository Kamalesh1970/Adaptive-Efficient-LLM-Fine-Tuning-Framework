#!/usr/bin/env bash
# ==============================================================================
# GPU Recovery & Power-On Script for ASUS TUF Gaming & Hybrid Laptop Hardware
# ==============================================================================
# This script re-enables the discrete NVIDIA GPU when powered down by ASUS
# hardware firmware (dgpu_disable=1), triggers a PCIe bus rescan, and loads
# the NVIDIA kernel drivers.
# ==============================================================================

set -e

echo "=== ASUS TUF / NVIDIA GPU Recovery Tool ==="

# 1. Enable ASUS discrete GPU power gate
if [ -f "/sys/devices/platform/asus-nb-wmi/dgpu_disable" ]; then
    CURRENT_STATE=$(cat /sys/devices/platform/asus-nb-wmi/dgpu_disable)
    echo "[1/4] Checking ASUS dGPU power gate (current: dgpu_disable=$CURRENT_STATE)..."
    if [ "$CURRENT_STATE" = "1" ]; then
        echo "      Powering on discrete GPU via asus-nb-wmi..."
        echo 0 | sudo tee /sys/devices/platform/asus-nb-wmi/dgpu_disable >/dev/null
    else
        echo "      dGPU power gate is already enabled (0)."
    fi
else
    echo "[1/4] No asus-nb-wmi dgpu_disable gate found (skipping)."
fi

# 2. Rescan PCIe bus so the kernel discovers the newly powered GPU
echo "[2/4] Rescanning PCI bus to detect NVIDIA GPU on PCIe..."
echo 1 | sudo tee /sys/bus/pci/rescan >/dev/null
sleep 1

# 3. Load NVIDIA kernel modules
echo "[3/4] Loading NVIDIA driver kernel modules..."
sudo modprobe nvidia 2>/dev/null || true
sudo modprobe nvidia_modeset 2>/dev/null || true
sudo modprobe nvidia_uvm 2>/dev/null || true

# 4. Verify nvidia-smi and PyTorch
echo "[4/4] Verifying NVIDIA driver and CUDA availability..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

echo ""
echo "=== PyTorch CUDA Detection Check ==="
python3 -c '
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM Capacity:   {props.total_memory / (1024**3):.2f} GB")
    print("SUCCESS: GPU is active and ready for training and evaluation!")
else:
    print("WARNING: GPU is not yet visible to PyTorch. A system reboot may be required if the driver was freshly upgraded.")
'

