# Sparse KV SIMT LRU operators

`SparseKvPlan` and `SparseKvTransfer` are derived from the corresponding
MemFabric Hybrid runtime kernels. The derived kernel files retain their Mulan
PSL v2 copyright and license notices. The vLLM Ascend host integration and
operator definitions use this repository's Apache-2.0 license.

The operators are built only for Ascend 950. `SparseKvPlan` keeps LRU state in
NPU tensors and emits resident slots plus compact miss descriptors.
`SparseKvTransfer` consumes those descriptors and copies BF16 K/V payloads
from a MemFabric registered Host DVA into the layer's resident buffer.
