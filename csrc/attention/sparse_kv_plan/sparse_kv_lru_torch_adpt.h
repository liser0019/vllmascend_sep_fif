/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef VLLM_ASCEND_SPARSE_KV_LRU_TORCH_ADPT_H
#define VLLM_ASCEND_SPARSE_KV_LRU_TORCH_ADPT_H

#include <torch/extension.h>

#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {
namespace {
inline void CheckSparseKvRuntimeInputs(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                                       const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                                       const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                                       const at::Tensor& activeRows, int64_t topk, int64_t capacity) {
  TORCH_CHECK(reqIds.dim() == 1 && reqIds.scalar_type() == at::kLong, "sparse KV req_ids must be 1-D int64");
  TORCH_CHECK(topkIndices.dim() == 2 && topkIndices.scalar_type() == at::kInt,
              "sparse KV topk_indices must be 2-D int32");
  TORCH_CHECK(topkIndices.size(1) == topk, "sparse KV topk shape mismatch");
  TORCH_CHECK(stablePrefixLens.numel() == topkIndices.size(0) && visibleSeqLens.numel() == topkIndices.size(0) &&
                  tokenToReq.numel() == topkIndices.size(0),
              "sparse KV row metadata length mismatch");
  TORCH_CHECK(stablePrefixLens.scalar_type() == at::kInt && visibleSeqLens.scalar_type() == at::kInt &&
                  tokenToReq.scalar_type() == at::kInt,
              "sparse KV row metadata must be int32");
  TORCH_CHECK(blockTable.dim() == 2 && blockTable.scalar_type() == at::kInt, "sparse KV block_table must be 2-D int32");
  TORCH_CHECK(blockTable.size(0) == reqIds.size(0), "sparse KV request ids and block table row counts must match");
  TORCH_CHECK(activeRows.numel() == 1 && activeRows.scalar_type() == at::kInt,
              "sparse KV active_rows must be a scalar int32 tensor");
  TORCH_CHECK(capacity > 0, "sparse KV resident capacity must be positive");
}

inline void RunSparseKvTransfer(const at::Tensor& missCount, const at::Tensor& missTokens, const at::Tensor& missSlots,
                                const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                                const at::Tensor& hostCacheBases, const at::Tensor& activeRows,
                                const at::Tensor& residentK, const at::Tensor& residentV, int64_t topk,
                                int64_t capacity, int64_t blockSize, int64_t hostNumBlocks, int64_t tokenSizeBytesK,
                                int64_t tokenSizeBytesV) {
  EXEC_NPU_CMD(aclnnSparseKvTransfer, missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases,
               activeRows, residentK, residentV, topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK,
               tokenSizeBytesV);
}
}  // namespace

inline void npu_sparse_kv_plan_transfer(const at::Tensor& reqIds, const at::Tensor& topkIndices,
                                        const at::Tensor& stablePrefixLens, const at::Tensor& visibleSeqLens,
                                        const at::Tensor& tokenToReq, const at::Tensor& blockTable,
                                        const at::Tensor& activeRows, const at::Tensor& lastReqIds,
                                        const at::Tensor& slotToToken, const at::Tensor& lruSlots,
                                        const at::Tensor& currentSlots, const at::Tensor& missCount,
                                        const at::Tensor& missTokens, const at::Tensor& missSlots,
                                        const at::Tensor& compactWorkspace, const at::Tensor& hostCacheBases,
                                        const at::Tensor& residentK, const at::Tensor& residentV, int64_t topk,
                                        int64_t capacity, int64_t maxToken, int64_t blockSize, int64_t hostNumBlocks,
                                        int64_t tokenSizeBytesK, int64_t tokenSizeBytesV) {
#ifndef NDEBUG
  CheckSparseKvRuntimeInputs(reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable, activeRows,
                             topk, capacity);
#endif
  EXEC_NPU_CMD(aclnnSparseKvPlan, reqIds, topkIndices, stablePrefixLens, visibleSeqLens, tokenToReq, blockTable,
               activeRows, lastReqIds, slotToToken, lruSlots, currentSlots, missCount, missTokens, missSlots,
               compactWorkspace, topk, capacity, maxToken, blockSize, hostNumBlocks);
  RunSparseKvTransfer(missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases, activeRows, residentK,
                      residentV, topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK, tokenSizeBytesV);
}

inline void npu_sparse_kv_transfer(const at::Tensor& missCount, const at::Tensor& missTokens,
                                   const at::Tensor& missSlots, const at::Tensor& tokenToReq,
                                   const at::Tensor& blockTable, const at::Tensor& hostCacheBases,
                                   const at::Tensor& activeRows, const at::Tensor& residentK,
                                   const at::Tensor& residentV, int64_t topk, int64_t capacity, int64_t blockSize,
                                   int64_t hostNumBlocks, int64_t tokenSizeBytesK, int64_t tokenSizeBytesV) {
#ifndef NDEBUG
  TORCH_CHECK(missCount.dim() == 1 && missCount.scalar_type() == at::kInt, "sparse KV miss_count must be 1-D int32");
  TORCH_CHECK(hostCacheBases.numel() == 2 && hostCacheBases.scalar_type() == at::kLong,
              "sparse KV host_cache_bases must contain two int64 addresses");
#endif
  RunSparseKvTransfer(missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases, activeRows, residentK,
                      residentV, topk, capacity, blockSize, hostNumBlocks, tokenSizeBytesK, tokenSizeBytesV);
}
}  // namespace vllm_ascend

#endif
