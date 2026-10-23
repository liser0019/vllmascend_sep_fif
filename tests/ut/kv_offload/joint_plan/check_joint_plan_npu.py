# SPDX-License-Identifier: Apache-2.0
"""A5 Plan/Transfer mapping audit + event timing, with HBM source buffers.

This isolates the operator from MemFabric/URMA. Production Host DDR and SFA
integration still require an end-to-end run. No package-global imports required.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_wrapper():
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/sparse_kv_simt_lru.py"
    spec = importlib.util.spec_from_file_location("joint_plan_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SparseKVSimtLru


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, help="built _C shared library")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--rows-per-request", type=int, default=3)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.ops.load_library(args.extension)
    cls = load_wrapper()
    c, t, w, b = args.capacity, args.topk, args.rows_per_request, args.requests
    if not 0 < t <= c or c % 2 or min(w, b, args.repeat) <= 0:
        parser.error("require 0 < topk <= capacity, even capacity, positive rows/requests/repeat")
    rows, block_size, dim = b * w, 2, 16
    n = max(c * 2, w * t + 16)
    n += n % block_size
    source_k = torch.empty((b * n // block_size, block_size, 1, dim), dtype=torch.bfloat16, device="npu")
    source_v = torch.empty_like(source_k)
    values = torch.arange(b * n, dtype=torch.int32, device="npu").remainder(113).to(torch.bfloat16)
    source_k.view(b * n, dim).copy_(values[:, None].expand(-1, dim))
    source_v.copy_(source_k + 1)
    block_table = torch.arange(b * n // block_size, dtype=torch.int32, device="npu").view(b, -1)
    token_to_req = torch.arange(b, dtype=torch.int32, device="npu").repeat_interleave(w)
    req_ids = torch.arange(b, dtype=torch.int64, device="npu")
    visible = torch.full((rows,), n, dtype=torch.int32, device="npu")
    stable = torch.full_like(visible, n - w)
    topk = torch.empty((rows, t), dtype=torch.int32, device="npu")
    cases = {}
    for enabled in (False, True):
        current = torch.empty((rows, t), dtype=torch.int32, device="npu")
        pages = torch.arange(rows * c // block_size, dtype=torch.int32, device="npu").view(rows, -1)
        rk = torch.empty((rows, c, 1, dim), dtype=torch.bfloat16, device="npu")
        rv = torch.empty_like(rk)
        state = cls(
            max_rows=rows,
            topk=t,
            capacity=c,
            max_token=n,
            block_size=block_size,
            host_num_blocks=b * n // block_size,
            token_size_bytes_k=dim * 2,
            token_size_bytes_v=dim * 2,
            host_k_bases=[source_k.data_ptr()],
            host_v_bases=[source_v.data_ptr()],
            current_slots=current,
            device=rk.device,
            resident_block_table=pages,
            joint_mtp=enabled,
        )
        state.set_active_rows(rows)
        kwargs = dict(
            layer_id=0,
            req_ids=req_ids,
            topk_indices=topk,
            stable_prefix_lens=stable,
            visible_seq_lens=visible,
            token_to_req=token_to_req,
            block_table=block_table,
            resident_k=rk,
            resident_v=rv,
        )
        cases[enabled] = state, kwargs, rk, rv, pages
    # Exercise sharing -> overflow -> sharing, causal masking, duplicate slots,
    # request identity reset, and overwritten suffix on both eager and replay.
    for capture in (False, True):
        graphs = {}
        topk.copy_(torch.arange(t, dtype=torch.int32, device="npu").expand(rows, -1))
        for enabled, (state, kw, *_rest) in cases.items():
            state.plan_and_transfer(**kw)
            if capture:
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g):
                    state.plan_and_transfer(**kw)
                graphs[enabled] = g
        for step in range(8):
            selection = torch.arange(t, dtype=torch.int32).repeat(rows, 1)
            if step % 3 == 1:
                selection += (torch.arange(rows) % w)[:, None] * t
            # The same suffix can be visible to a later query and invalid for an earlier one.
            selection[:, -1] = n - 1
            visible.copy_((n - w + 1 + torch.arange(rows) % w).to(torch.int32).npu())
            topk.copy_(selection.npu())
            source_k.view(b, n, dim)[:, n - w :].fill_(step + 120)
            source_v.view(b, n, dim)[:, n - w :].fill_(step + 121)
            if step == 4:
                req_ids.add_(b)
            expected_valid = (topk >= 0) & (topk < visible[:, None])
            for enabled, (state, kw, rk, rv, pages) in cases.items():
                if capture:
                    graphs[enabled].replay()
                else:
                    state.plan_and_transfer(**kw)
                slots = state.current_slots.long()
                owners = state.resident_rows.long()
                assert torch.equal(slots >= 0, expected_valid)
                row_index = torch.arange(rows, device="npu")[:, None].expand(rows, t)
                # Resolve through SFA's page table as well as Transfer's owner table.
                page = pages.long()[row_index, slots.clamp_min(0) // block_size]
                linear = page * block_size + slots.clamp_min(0) % block_size
                assert torch.equal(linear, owners[:, None] * c + slots.clamp_min(0))
                for buf, src in ((rk, source_k), (rv, source_v)):
                    got = buf.view(rows * c, dim)[linear]
                    want = src.view(b * n, dim)[token_to_req.long()[:, None] * n + topk.long()]
                    assert torch.equal(got[expected_valid], want[expected_valid])
        # Fixed latest workload; timings include Plan + Transfer, not full model throughput.
        for enabled, (state, kw, *_rest) in cases.items():
            start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(args.repeat):
                if capture:
                    graphs[enabled].replay()
                else:
                    state.plan_and_transfer(**kw)
            end.record()
            torch.npu.synchronize()
            print(f"joint={enabled} replay={capture} mean_us={start.elapsed_time(end) * 1000 / args.repeat:.3f}")
    print("PASS: operator payload and page-table checks (HBM source, not Host/PD validation)")


if __name__ == "__main__":
    main()
