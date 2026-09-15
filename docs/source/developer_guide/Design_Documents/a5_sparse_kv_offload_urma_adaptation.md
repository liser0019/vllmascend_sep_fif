# Atlas A5 Sparse KV Offload over URMA

## 1. Overview

### 1.1 Summary

Sparse KV offload originally used one Host KV pool shared by all Decode tensor
parallel (TP) ranks on Atlas A3. The pool address could be used by the CPU,
MemFabric transfers, and NPU sparse-copy operators. Atlas A5 uses a
connection-based Host pool whose Host virtual address (HVA) and Device virtual
address (DVA) have different meanings. Reusing the A3 address and TP ownership
rules can make PD disaggregation start successfully while storing or reading KV
data from the wrong location.

This design gives each A5 TP process a rank-local, URMA-compatible MemFabric
pool. PD transfer descriptors use the pool HVA. NPU sparse-copy descriptors use
the DVA returned by `offload.get_dva()`. Because the pool is no longer shared,
each A5 rank pulls a complete copy of the replicated main MLA KV.

### 1.2 Motivation

An earlier A5 experiment allocated memory with `aclrtMallocHost`, registered it
again with `aclrtHostRegisterV2`, and created a CPU tensor from the returned
device pointer. This has three problems:

1. `aclrtMallocHost` already returns managed Host memory. The CANN documentation
   does not permit treating it as ordinary unlocked memory for a second
   `ACL_HOST_REG_PINNED` registration.
2. `aclrtHostGetDevicePointer` returns an address for Device access. It does not
   turn the address into CPU-owned tensor storage, and a mapped DVA must not be
   used as the address of an ACL memory-copy operation.
3. Changing from one TP-shared pool to one pool per rank without changing PD
   block ownership leaves every local pool only partially populated. Missing KV
   blocks can corrupt attention results without making the transfer API fail.

MemFabric release 1.2 already owns the supported A5 memory lifecycle. It creates
the Host allocation, performs the device mapping, and exposes both address
forms. vLLM Ascend should consume that API instead of reproducing CANN memory
management with `ctypes`.

### 1.3 Goals

- Run sparse KV offload with PD disaggregation on Atlas A5 through
  `TransDataOpType.DEVICE_URMA`.
- Preserve the existing Atlas A3 shared-pool behavior.
- Keep HVA and DVA usage explicit at every call site.
- Populate every A5 rank-local pool with complete main MLA KV data.
- Fail during startup when the MemFabric version or configuration is
  incompatible.

### 1.4 Scope limits

- A5 `use_fused_overlap` remains disabled because its membership and planning
  storage still assumes one TP-shared Host pool.
- The change does not add a second CANN allocator or Host-registration wrapper
  to vLLM Ascend.
- Performance tuning of the CPU LRU planner and attention graph boundaries is
  separate from the correctness adaptation.

## 2. Use Case Analysis

The primary use case is a GLM sparse-attention model with Prefill and Decode on
different A5 nodes. Prefill publishes registered HBM KV. Each Decode TP rank
pulls the required KV into its own Host pool, updates the pool with newly
decoded KV, and loads TopK rows into its NPU resident buffer.

The implementation must also retain these behaviors:

- A3 Decode TP ranks continue to share one pool and divide the main KV pull.
- A5 and A3 use the same CPU LRU planner implementation.
- Invalid protocol selection, missing MemFabric APIs, failed pool creation, and
  failed HVA-to-DVA translation stop startup with actionable errors.
- Address conversion occurs during cache registration, outside the per-layer
  inference hot path.

## 3. Design

### 3.1 Address and ownership model

| Platform | Host pool ownership | PD transfer address | NPU operator address | Main KV pull |
| --- | --- | --- | --- | --- |
| A3 | One pool shared by Decode TP ranks | Shared HVA/GVA | Same mapped address | Split across Decode TP ranks |
| A5 | One complete pool in every Decode TP process | Rank-local HVA | Rank-local DVA from `offload.get_dva()` | Full copy on every Decode TP rank |

The A5 data path is:

```text
Prefill HBM (registered source)
        |
        | MemFabric DEVICE_URMA pull, destination = HVA
        v
Decode rank-local MemFabric Host pool (registered destination)
        |
        | offload.get_dva(HVA), once during registration
        v
Sparse-copy / LRU load descriptors, source or destination = DVA
        |
        v
Decode NPU resident TopK KV buffer
```

The HVA remains the address of the CPU tensor and the local destination passed
to `batch_transfer_sync_read`. The DVA is stored only in the operator address
tables. This separation follows the CANN mapped-Host-memory contract and the
MemFabric `offload_get_dva` API contract.

After the Decode TransferEngine is initialized, every rank registers its Host
K/V pool HVA and rank-local indexer HBM destinations before starting the pull
thread. Registration publishes the buffers to `smem_trans` and prepares them
for DEVICE_URMA. The DVA is never passed to `register_memory`.

### 3.2 Pool initialization

Atlas A5 initializes the offload pool with:

```python
config.scene = offload.Scene.LOCAL
config.flags = offload.OFFLOAD_FLAG_GIANT_PAGE
config.reserve_size = pool_size_bytes
config.alloc_size = pool_size_bytes
config.world_size = 1
config.rank_id = 0
```

Each TP worker is a separate process, so `Scene.LOCAL` creates one local pool
per worker. All workers allocate their K/V Host tensors from their own pool.
The existing A3 `Scene.SHARED` initialization and TP0-only allocation remain
unchanged.

When a KV layer is registered, the manager records two address tables:

- `hvas_k_bases` and `hvas_v_bases` for CPU access and PD transfer.
- `dvas_k_bases` and `dvas_v_bases` for NPU sparse-copy and onload operations.

For A3, the DVA tables alias the shared HVA tables. For A5, every HVA is
translated with `offload.get_dva()`. A zero result is a startup error.

### 3.3 PD transfer layout

Main MLA KV is replicated on the Prefill TP ranks. With the A3 shared pool,
Decode ranks can split the destination block range because all ranks write into
one physical pool. With A5 rank-local pools, each Decode rank needs every
requested main block in its own physical pool. The descriptor builder therefore
skips the Decode-TP block split when `replicate_main_pool` is enabled.

For unequal Prefill and Decode TP sizes, only the first Prefill contributor in a
Decode group supplies main MLA KV. This avoids duplicate writes while still
filling the complete rank-local pool. Indexer KV keeps its existing rank-local
partition logic.

### 3.4 Decode writeback and onload

Decode-produced K/V is replicated across TP ranks. On A3, TP0 writes it once to
the shared pool. On A5, every TP rank writes the same new K/V into its own pool.
The attention implementation therefore returns local CPU K/V tensors on every
A5 rank.

The writeback and onload descriptors use DVA bases:

- NPU-to-Host sparse-copy destinations use the layer DVA plus the token-slot
  byte offset.
- Host-to-NPU resident loads use DVA bases in the LRU address planner.
- `skip_topk` layer offsets are computed from DVA tables, so reused descriptors
  remain valid for the Device mapping.

### 3.5 Build integration

The sparse KV LRU implementation in `vllm_ascend_C` is Host-side C++ with
OpenMP. It is enabled for `ascend950.*` as well as `ascend910_93.*`. No new A5
AscendC kernel is introduced by this adaptation.

### 3.6 Compatibility checks

Startup on A5 requires:

- `memfabric_transfer_protocol` set to `device_urma` for PD deployment.
- MemFabric Python `offload.OFFLOAD_FLAG_GIANT_PAGE`.
- MemFabric Python `offload.get_dva`.
- Successful local pool initialization and nonzero DVA translation.

A5 fused-overlap mode is rejected before allocation because silently applying
its shared-pool assumptions would risk incorrect KV contents.

## 4. Alternatives Considered

### 4.1 Register Host memory directly from Python

Calling CANN through `ctypes` duplicates the allocator and mapping lifecycle
already owned by MemFabric. It also makes it easy to confuse HVA and DVA or to
register already managed Host memory with incompatible flags. This approach is
not used.

### 4.2 Broadcast one A5 Host address to all TP ranks

An A5 `Scene.LOCAL` pool belongs to one process and one device mapping. Other TP
ranks cannot safely use that process's HVA or DVA. Broadcasting the address is
not used.

### 4.3 Keep the A3 block split with rank-local pools

This fills only a fraction of each rank's KV pool. Later attention can read
uninitialized or stale blocks, which explains how the transfer calls can return
success while generated text is corrupt. This approach is not used.

### 4.4 Pass DVA to the PD transfer API

The mapped DVA is for Device access to Host memory. The local buffer supplied to
the transfer API remains the CPU-visible HVA. This approach is not used.

## 5. Risks and Drawbacks

- A5 Host memory per DP rank grows by approximately the Decode TP size because
  every TP process owns a complete pool.
- Main MLA P-to-D traffic is also replicated across Decode TP ranks.
- The static implementation cannot prove CANN, firmware, driver, huge-page,
  and URMA compatibility without an A5 environment.
- Registering many per-layer Host and indexer regions can add startup time and
  consume registration resources. Registration occurs once before the pull
  thread starts and is kept out of the inference hot path.

## 6. Validation Plan

### 6.1 Static and Host-only checks

- Python compilation for all modified modules and tests.
- Ruff lint and format checks.
- Unit tests for A3 shared-pool initialization, A5 local-pool initialization,
  protocol rejection, HVA/DVA separation, per-rank CPU-cache ownership, and
  full main-KV descriptor generation and destination registration on every A5
  TP rank.

### 6.2 A5 build checks

- Build vLLM Ascend with an `ascend950.*` `SOC_VERSION` and verify that
  `_C_ascend` exports all sparse KV CPU planner operators.
- Build and install the matching MemFabric release 1.2 package.
- Verify at startup that each Decode rank logs rank-local URMA pool creation and
  that `offload.get_dva()` succeeds for every K/V layer.

### 6.3 End-to-end correctness

Use greedy decoding and a fixed prompt, then compare token IDs with sparse
offload disabled and with the A3/CPU reference where available. Cover:

1. TP1 and TP greater than one.
2. Prompts spanning several KV blocks.
3. At least two concurrent requests with different sequence lengths.
4. Prefix-cache reuse followed by continued decode.
5. Enough generated tokens to exercise Decode writeback and subsequent onload.

Before performance profiling, verify transferred K/V bytes for at least one
layer and request at three points: Prefill source HBM, Decode Host HVA after the
URMA pull, and the Decode resident NPU buffer after sparse onload. A successful
transfer return code alone is not a content-integrity check.

## 7. References

- [CANN `aclrtHostRegisterV2` documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/runtimeapi/aclcppdevg_03_2128.html)
- [CANN `aclrtHostGetDevicePointer` documentation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/runtimeapi/aclcppdevg_03_2129.html)
- MemFabric `src/acc_offload/include/host/acc_offload.h` for
  `OFFLOAD_FLAG_GIANT_PAGE` and `offload_get_dva` semantics.
