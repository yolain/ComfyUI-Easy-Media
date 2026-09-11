"""Optional memory snapshots without resetting shared allocator statistics."""

import logging
import os

import psutil
import torch


def log_memory(label, device):
    if os.environ.get("SELFLIFT_MEMORY_LOG", "0") != "1":
        return
    device = torch.device(device)
    fields = []
    try:
        fields.append(f"process_rss={psutil.Process().memory_info().rss / 2**20:.2f} MiB")
        fields.append(f"system_available={psutil.virtual_memory().available / 2**20:.2f} MiB")
    except (psutil.Error, OSError) as error:
        fields.append(f"host_memory_unavailable={error}")
    if device.type == "cuda" and torch.cuda.is_initialized():
        try:
            free, total = torch.cuda.mem_get_info(device)
            fields.extend([
                f"allocated={torch.cuda.memory_allocated(device) / 2**20:.2f} MiB",
                f"reserved={torch.cuda.memory_reserved(device) / 2**20:.2f} MiB",
                f"process_peak_allocated={torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB",
                f"device_used={((total - free) / 2**20):.2f} MiB",
                f"device_free={free / 2**20:.2f} MiB",
            ])
        except RuntimeError as error:
            fields.append(f"cuda_memory_unavailable={error}")
    logging.info("[SelfLift memory] %s device=%s; %s", label, device, "; ".join(fields))
