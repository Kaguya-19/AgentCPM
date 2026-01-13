"""Cross entropy + DTensor support quick test.

Usage examples:
  torchrun --nproc_per_node=2 tests/cross_entropy_dtensor.py --batch 32 --classes 17
  python tests/cross_entropy_dtensor.py  # falls back to world_size=1

What it does:
  1. Initializes torch.distributed (supports torchrun or single-process fallback)
  2. Builds sharded DTensors (Shard(0)) for logits & targets
  3. Tries F.cross_entropy directly on DTensor inputs
  4. Gathers a replicated version as baseline reference
  5. Compares forward values (and gradients if supported) against baseline

Exit status 0 always (even if unsupported) — prints a summary. This is a diagnostic script.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

try:
	from torch.distributed._tensor import DeviceMesh, distribute_tensor, Shard, Replicate
except Exception as e:  # pragma: no cover
	raise RuntimeError("Your PyTorch build does not have DTensor experimental API.") from e


def init_dist_if_needed(backend_preference: str | None = None) -> Tuple[int, int]:
	if dist.is_available() and dist.is_initialized():
		return dist.get_rank(), dist.get_world_size()

	backend = backend_preference
	if backend is None:
		if torch.cuda.is_available():
			backend = "nccl"
		else:
			backend = "gloo"

	# If launched by torchrun, env:// works; else create a file init
	world_size = int(os.environ.get("WORLD_SIZE", "1"))
	rank = int(os.environ.get("RANK", "0"))
	if world_size > 1:
		dist.init_process_group(backend=backend, init_method="env://")
	else:
		# single-process fallback
		with tempfile.NamedTemporaryFile() as tf:
			init_method = f"file://{tf.name}"
			dist.init_process_group(backend=backend, init_method=init_method, rank=0, world_size=1)
	return rank, world_size


def build_mesh(device_type: str, world_size: int) -> DeviceMesh:
	return DeviceMesh(device_type, list(range(world_size)))


def main():  # noqa: C901
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch", type=int, default=16, help="Global batch size (must be divisible by world size)")
	parser.add_argument("--classes", type=int, default=10, help="Number of classes")
	parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Logit dtype")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--grad-check", action="store_true", help="Also compare gradients vs baseline replicate")
	parser.add_argument("--manual-fallback", action="store_true", help="Force manual distributed CE (sum/all-reduce) for reference/gradient when native DTensor path mismatches or lacks grad")
	args = parser.parse_args()

	torch.manual_seed(args.seed)

	rank, world_size = init_dist_if_needed()
	device_type = "cuda" if torch.cuda.is_available() else "cpu"
	device = torch.device(device_type, rank if device_type == "cuda" else 0)

	if args.batch % world_size != 0:
		if rank == 0:
			print(f"[WARN] batch {args.batch} not divisible by world size {world_size}; reducing batch to nearest divisible")
		new_batch = (args.batch // world_size) * world_size
		if new_batch == 0:
			new_batch = world_size
		args.batch = new_batch

	local_batch = args.batch // world_size
	if local_batch == 0:
		local_batch = 1
		args.batch = world_size

	mesh = build_mesh(device_type, world_size)

	dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
	logit_dtype = dtype_map[args.dtype]

	# Create local shard for logits (requires grad)
	local_logits = torch.randn(local_batch, args.classes, device=device, dtype=logit_dtype, requires_grad=True)
	# Targets: integer class indices; keep as regular tensor then wrap
	local_targets = torch.randint(0, args.classes, (local_batch,), device=device)

	# Wrap into DTensor (sharded along batch dim)
	logits_dt = distribute_tensor(local_logits, mesh, placements=[Shard(0)])
	targets_dt = distribute_tensor(local_targets, mesh, placements=[Shard(0)])

	# Baseline replicate (full) tensors
	full_logits_dt = logits_dt.redistribute(placements=[Replicate()])
	full_targets_dt = targets_dt.redistribute(placements=[Replicate()])
	if rank == 0:
		ref_loss = F.cross_entropy(full_logits_dt.to_local(), full_targets_dt.to_local())
	else:
		ref_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
	# Broadcast reference scalar to all ranks
	dist.broadcast(ref_loss, src=0)

	supported = True
	forward_diff = math.nan
	grad_max_diff = math.nan
	error_msg = ""
	manual_used = False
	manual_forward_diff = math.nan
	manual_grad_max_diff = math.nan

	try:
		# Native path
		test_loss_local_mean = F.cross_entropy(logits_dt, targets_dt)
		# We don't know if native returns global mean or local mean; assume local mean and aggregate to global.
		weighted = test_loss_local_mean.detach().float() * (local_batch / args.batch)
		if dist.is_initialized() and world_size > 1:
			dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
		global_mean_from_native = weighted.item()
		forward_diff = abs(global_mean_from_native - ref_loss.float().item())

		grad_supported = False
		if args.grad_check:
			# Backward on native path
			test_loss_local_mean.backward()
			obtained_grad = local_logits.grad  # use original tensor
			if obtained_grad is not None:
				grad_supported = True
				local_logits_baseline = full_logits_dt.to_local().detach().clone().requires_grad_(True)
				baseline_loss = F.cross_entropy(local_logits_baseline, full_targets_dt.to_local())
				baseline_loss.backward()
				baseline_grad_full = local_logits_baseline.grad
				start = rank * local_batch
				end = start + local_batch
				baseline_grad_slice = baseline_grad_full[start:end]
				grad_max_diff = (obtained_grad - baseline_grad_slice).abs().max().item()
			else:
				grad_max_diff = math.inf

		need_manual = (
			args.manual_fallback
			or (forward_diff > (5e-3 if logit_dtype in (torch.float16, torch.bfloat16) else 1e-5))
			or (args.grad_check and not grad_supported)
		)
		if need_manual:
			manual_used = True
			# Manual forward (global mean)
			local_sum = F.cross_entropy(local_logits.detach(), local_targets, reduction="sum").float()
			if dist.is_initialized() and world_size > 1:
				dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
			manual_loss = local_sum / args.batch
			manual_forward_diff = abs(manual_loss.item() - ref_loss.float().item())
			# Manual gradient
			if args.grad_check:
				local_logits_m = local_logits.detach().clone().requires_grad_(True)
				local_sum_m = F.cross_entropy(local_logits_m, local_targets, reduction="sum").float()
				(local_sum_m / args.batch).backward()
				manual_grad = local_logits_m.grad
				# Need baseline grad slice if not computed yet
				if not grad_supported:
					local_logits_baseline = full_logits_dt.to_local().detach().clone().requires_grad_(True)
					baseline_loss = F.cross_entropy(local_logits_baseline, full_targets_dt.to_local())
					baseline_loss.backward()
					baseline_grad_full = local_logits_baseline.grad
					start = rank * local_batch
					end = start + local_batch
					baseline_grad_slice = baseline_grad_full[start:end]
				manual_grad_max_diff = (manual_grad - baseline_grad_slice).abs().max().item()
	except Exception as e:  # noqa: BLE001
		supported = False
		error_msg = str(e)

	# Summarize (rank 0)
	if rank == 0:
		print("==== Cross Entropy DTensor Support Test ====")
		print(f"World Size        : {world_size}")
		print(f"Device Type       : {device_type}")
		print(f"Global Batch      : {args.batch}")
		print(f"Local Batch       : {local_batch}")
		print(f"Num Classes       : {args.classes}")
		print(f"Logits DType      : {logit_dtype}")
		print(f"Grad Check        : {args.grad_check}")
		print(f"Baseline Loss     : {ref_loss.item():.6f}")
		if supported:
			print("Direct DTensor CE : SUPPORTED (forward)")
			print(f"Forward |Δ|       : {forward_diff:.3e}")
			if args.grad_check:
				if math.isfinite(grad_max_diff):
					print("Gradient Support  : YES")
					print(f"Grad max |Δ|      : {grad_max_diff:.3e}")
				else:
					print("Gradient Support  : NO (grad remained None)")
					print("Grad max |Δ|      : N/A")
			if manual_used:
				print("-- Manual Distributed CE Fallback --")
				print(f"Manual Forward |Δ| : {manual_forward_diff:.3e}")
				if args.grad_check and math.isfinite(manual_grad_max_diff):
					print(f"Manual Grad max |Δ|: {manual_grad_max_diff:.3e}")
				elif args.grad_check:
					print("Manual Grad max |Δ|: N/A")
			# Decide verdict prioritizing manual path if native mismatch
			ol = 5e-3 if logit_dtype in (torch.float16, torch.bfloat16) else 1e-5
			if manual_used:
				grad_ok = (not args.grad_check) or (math.isfinite(manual_grad_max_diff) and manual_grad_max_diff < 5 * ol)
				status = "PASS(manual)" if (manual_forward_diff < ol and grad_ok) else "MISMATCH"
			else:
				if not args.grad_check:
					status = "PASS" if forward_diff < ol else "MISMATCH"
				elif math.isfinite(grad_max_diff):
					status = "PASS" if (forward_diff < ol and grad_max_diff < 5 * ol) else "MISMATCH"
				else:
					status = "FORWARD_ONLY"
			print(f"Verdict           : {status}")
		else:
			print("Direct DTensor CE : NOT SUPPORTED")
			print(f"Error             : {error_msg}")
			print("(You can still use replicate() workaround before calling F.cross_entropy.)")

	# Clean up
	dist.barrier()
	if dist.is_initialized():
		dist.destroy_process_group()


if __name__ == "__main__":  # pragma: no cover
	main()
