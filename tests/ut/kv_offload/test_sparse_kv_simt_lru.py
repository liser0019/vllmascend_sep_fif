# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_simt_lru import (
    SparseKVSimtLru,
    _validate_simt_lru_config,
    sparse_kv_plan_hash_capacity,
    sparse_kv_plan_workspace_elements,
)


def _make_backend() -> SparseKVSimtLru:
    backend = SparseKVSimtLru.__new__(SparseKVSimtLru)
    backend.topk = 4
    backend.capacity = 8
    backend.max_token = 64
    backend.block_size = 2
    backend.host_num_blocks = 32
    backend.token_size_bytes_k = 16
    backend.token_size_bytes_v = 8
    backend.active_rows = "active_rows"
    backend.current_slots = "current_slots"
    backend.compact_workspace = "workspace"
    backend.plan_owner_layer_id = None
    backend.layers = [
        SimpleNamespace(
            last_req_ids=f"last_req_ids_{layer}",
            slot_to_token=f"slot_to_token_{layer}",
            lru_slots=f"lru_slots_{layer}",
            miss_count=f"miss_count_{layer}",
            miss_tokens=f"miss_tokens_{layer}",
            miss_slots=f"miss_slots_{layer}",
            host_cache_bases=f"host_cache_bases_{layer}",
        )
        for layer in range(2)
    ]
    backend._plan_transfer_op = Mock()
    backend._transfer_op = Mock()
    return backend


@pytest.mark.parametrize(
    ("topk", "expected"),
    [(1, 32), (16, 32), (17, 64), (2048, 4096)],
)
def test_plan_hash_capacity_is_power_of_two_with_half_load(topk, expected):
    assert sparse_kv_plan_hash_capacity(topk) == expected


def test_plan_workspace_matches_kernel_layout():
    # keys/first-position/owner hashes + evict/hit slots + miss positions.
    assert sparse_kv_plan_workspace_elements(topk=2048, capacity=4096) == 3 * 4096 + 2 * 4096 + 2048


def test_config_rejects_topk_larger_than_resident_capacity():
    with pytest.raises(ValueError, match="topk must not exceed capacity"):
        _validate_simt_lru_config(
            max_rows=2,
            topk=9,
            capacity=8,
            max_token=64,
            block_size=2,
            host_num_blocks=32,
            token_size_bytes_k=16,
            token_size_bytes_v=8,
            host_k_bases=[0x1000],
            host_v_bases=[0x2000],
            current_slots=torch.empty((2, 9), dtype=torch.int32),
        )


def test_plan_and_transfer_records_owner_and_uses_layer_state():
    backend = _make_backend()
    inputs = {
        "req_ids": "req_ids",
        "topk_indices": "topk_indices",
        "stable_prefix_lens": "stable_prefix_lens",
        "visible_seq_lens": "visible_seq_lens",
        "token_to_req": "token_to_req",
        "block_table": "block_table",
        "resident_k": "resident_k_0",
        "resident_v": "resident_v_0",
    }

    backend.plan_and_transfer(layer_id=0, **inputs)

    assert backend.plan_owner_layer_id == 0
    call = backend._plan_transfer_op.call_args.args
    assert call[:7] == (
        "req_ids",
        "topk_indices",
        "stable_prefix_lens",
        "visible_seq_lens",
        "token_to_req",
        "block_table",
        "active_rows",
    )
    assert call[7:16] == (
        "last_req_ids_0",
        "slot_to_token_0",
        "lru_slots_0",
        "current_slots",
        "miss_count_0",
        "miss_tokens_0",
        "miss_slots_0",
        "workspace",
        "host_cache_bases_0",
    )


def test_skip_topk_reuses_plan_but_targets_current_layer_buffers():
    backend = _make_backend()
    backend.plan_owner_layer_id = 0

    backend.transfer_reused_plan(
        layer_id=1,
        token_to_req="token_to_req",
        block_table="block_table",
        resident_k="resident_k_1",
        resident_v="resident_v_1",
    )

    call = backend._transfer_op.call_args.args
    assert call[:7] == (
        "miss_count_0",
        "miss_tokens_0",
        "miss_slots_0",
        "token_to_req",
        "block_table",
        "host_cache_bases_1",
        "active_rows",
    )
    assert call[7:9] == ("resident_k_1", "resident_v_1")


def test_skip_topk_requires_an_existing_plan():
    backend = _make_backend()
    with pytest.raises(RuntimeError, match="before a Plan has run"):
        backend.transfer_reused_plan(
            layer_id=1,
            token_to_req="token_to_req",
            block_table="block_table",
            resident_k="resident_k_1",
            resident_v="resident_v_1",
        )
