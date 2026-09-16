/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MemFabric_Hybrid is licensed under Mulan PSL v2.
 *
 * 昇腾 950：64 个 AIV 的 Sparse KV Transfer。
 * 本实现通过 vLLM Ascend 自定义算子接口接收 Host DVA 和 resident 目标。
 *
 * Plan 使用 SIMT 做不规则索引/哈希；这里沿用 Ascend C 的 SIMD 编程外壳，
 * 由标量单元计算地址、MTE2/MTE3 执行 GM -> UB -> GM 搬运。
 * Transfer 不启动 SIMT VF，所以这里没有“2048 个搬运线程”的概念。
 * Host K/V 必须已注册为 NPU 可访问的 DVA；普通 CPU malloc 地址不能直接传入。
 *
 * 主要修改：
 * 1. 固定 launch 64 个 AIV block，batch 与所有 shape 仍由运行时传入。
 * 2. 跨请求累计 miss 编号后按 %64 分配，避免每条请求都从 core 0 重新分配。
 *    例如 32 条请求各有 2 个 miss，正好形成 64 个任务，每核一个。
 * 3. 两块 16 KiB UB、深度为 2 的绑定队列，实现搬入下一块/搬出上一块的流水。
 * 4. 删除每个 K/V payload 后的 PIPE_ALL，只在全部任务提交并排空队列后收尾。
 * 5. K/V 大小可不同；超过 16 KiB 自动分块，非 32 字节整数倍使用 DataCopyPad。
 *
 * 语义与使用约束：
 * - source = HostBase + physicalBlock*blockSize*tokenBytes + offset*tokenBytes。
 * - destination = DeviceBase + (request*capacity + slot)*tokenBytes。
 * - Plan 应保证每行有效 miss 的 slot 互不重复，各行目标区间独立。
 * - Plan 完成后才能执行 Transfer；同 stream 依次 launch 已满足此依赖。
 * - tiling 携带 Host pool 物理块总数，Transfer 会拒绝越界 blockIndex。
 *   上层仍需保证 DVA 注册范围和所有目标 buffer 大小合法。
 * - 均衡的是 miss 任务数；无效任务被跳过、总 miss 太少时，不保证 64 核都有搬运。
 * - 16 KiB 是初始调优值，不代表所有链路/payload 的最优值，可实测 8/16/32 KiB。
 *
 * API 依据（Huawei 官方开发仓 master 预览，部署时以对应 CANN 版本为准）：
 * https://asc.gitcode.com/api/SIMD-API/basic_api/resource_management/TQueBind/TQueBind_intro.html
 * https://asc.gitcode.com/api/SIMD-API/basic_api/resource_management/TQueBind/DeQue.html
 * https://asc.gitcode.com/api/SIMD-API/basic_api/resource_management/TQueBind/FreeTensor.html
 */

#include <cstdint>
#include "kernel_operator.h"

namespace {

constexpr uint32_t TRANSFER_BUFFER_COUNT = 2U;
constexpr uint32_t TRANSFER_CHUNK_BYTES = 16U * 1024U;
static_assert(TRANSFER_CHUNK_BYTES % 32U == 0U, "UB chunk must be 32-byte aligned");

struct SparseKvTransferTilingData {
  int64_t maxRows;
  int64_t maxRequests;
  int64_t maxNumBlocks;
  int64_t topk;
  int64_t capacity;
  int64_t hostNumBlocks;
  int32_t blockSize;
  int32_t tokenSizeBytesK;
  int32_t tokenSizeBytesV;
};

class SparseKvTransferRuntimeKernel {
 public:
  // 每个 AIV 构造自己的对象，UB 与队列互不共享；所有可写目标按 miss 分片。
  __aicore__ inline void Init(GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots, GM_ADDR tokenToReq,
                              GM_ADDR blockTable, GM_ADDR hostCacheBases, GM_ADDR deviceKBase, GM_ADDR deviceVBase,
                              int64_t numRows, int64_t maxRequests, int64_t topk, int64_t capacity,
                              int64_t maxNumBlocks, int64_t hostNumBlocks, int32_t blockSize, int32_t tokenSizeBytesK,
                              int32_t tokenSizeBytesV) {
    aivNum_ = AscendC::GetBlockNum();
    aivIndex_ = AscendC::GetBlockIdx();
    missCount_ = reinterpret_cast<__gm__ int32_t*>(missCount);
    missTokens_ = reinterpret_cast<__gm__ int32_t*>(missTokens);
    missSlots_ = reinterpret_cast<__gm__ int32_t*>(missSlots);
    tokenToReq_ = reinterpret_cast<__gm__ int32_t*>(tokenToReq);
    blockTable_ = reinterpret_cast<__gm__ int32_t*>(blockTable);
    const __gm__ int64_t* bases = reinterpret_cast<__gm__ int64_t*>(hostCacheBases);
    hostKBase_ = static_cast<uint64_t>(bases[0]);
    hostVBase_ = static_cast<uint64_t>(bases[1]);
    deviceKBase_ = reinterpret_cast<uint64_t>(deviceKBase);
    deviceVBase_ = reinterpret_cast<uint64_t>(deviceVBase);
    numRows_ = numRows;
    maxRequests_ = maxRequests;
    topk_ = topk;
    capacity_ = capacity;
    maxNumBlocks_ = maxNumBlocks;
    hostNumBlocks_ = hostNumBlocks;
    blockSize_ = blockSize;
    tokenSizeBytesK_ = tokenSizeBytesK;
    tokenSizeBytesV_ = tokenSizeBytesV;
    blockBytesK_ = static_cast<uint64_t>(blockSize) * tokenSizeBytesK;
    blockBytesV_ = static_cast<uint64_t>(blockSize) * tokenSizeBytesV;

    // InitBuffer 的 num=2 才真正分配两块 UB；仅把队列模板 depth 改成 2 不够。
    // 总 UB = 2*16 KiB，metadata 只有两个地址和两个长度，无 GM 描述符数组。
    pipe_.InitBuffer(copyQueue_, TRANSFER_BUFFER_COUNT, TRANSFER_CHUNK_BYTES);
    head_ = 0U;
    tail_ = 0U;
    pending_ = 0U;
  }

  /**
   * P 为之前所有请求的有效 count 之和，本行 miss i 由 (P+i)%aivNum 号核处理。
   * 不需要先生成全局 prefix 数组：每核遍历 count 时只维护 P%aivNum 即可。
   * 同一核在本行的第一个 i = (coreId+aivNum-P%aivNum)%aivNum。
   *
   * 全部 count 按相同规则钳位，各核计算出的 P 一致，每个任务恰好有一个 owner。
   * B 可以小于、等于或大于 64；各请求 miss 数不同也不影响正确性。
   */
  __aicore__ inline void Process() {
    uint32_t prefixMod = 0U;
    for (int64_t row = 0; row < numRows_; ++row) {
      int32_t count = missCount_[row];
      if (count < 0) {
        count = 0;
      } else if (static_cast<int64_t>(count) > topk_) {
        // 仅在 topk<count<=INT32_MAX 时执行，向 int32 转换安全。
        count = static_cast<int32_t>(topk_);
      }
      const uint32_t first = (aivIndex_ + aivNum_ - prefixMod) % aivNum_;
      prefixMod = (prefixMod + static_cast<uint32_t>(count) % aivNum_) % aivNum_;
      const int64_t rowMissBase = row * topk_;
      const int32_t request = tokenToReq_[row];
      if (request < 0 || request >= maxRequests_) {
        continue;
      }
      const int64_t rowBlockBase = static_cast<int64_t>(request) * maxNumBlocks_;

      for (int64_t index = first; index < count; index += aivNum_) {
        const int32_t token = missTokens_[rowMissBase + index];
        const int32_t slot = missSlots_[rowMissBase + index];
        if (token < 0 || slot < 0 || static_cast<int64_t>(slot) >= capacity_) {
          continue;
        }
        // 支持任意正 blockSize，不假定它是 2 的幂。
        const int32_t blockId = token / blockSize_;
        if (static_cast<int64_t>(blockId) >= maxNumBlocks_) {
          continue;
        }
        const int32_t blockIndex = blockTable_[rowBlockBase + blockId];
        if (blockIndex < 0 || blockIndex >= hostNumBlocks_) {
          continue;
        }
        // 商已知，用乘减获得余数，避免代码中再次写一个独立除法。
        const int32_t offsetInBlock = token - blockId * blockSize_;
        const uint64_t sourceK = hostKBase_ + static_cast<uint64_t>(blockIndex) * blockBytesK_ +
                                 static_cast<uint64_t>(offsetInBlock) * tokenSizeBytesK_;
        const uint64_t sourceV = hostVBase_ + static_cast<uint64_t>(blockIndex) * blockBytesV_ +
                                 static_cast<uint64_t>(offsetInBlock) * tokenSizeBytesV_;
        const uint64_t linearSlot = static_cast<uint64_t>(row) * capacity_ + slot;
        const uint64_t destinationK = deviceKBase_ + linearSlot * tokenSizeBytesK_;
        const uint64_t destinationV = deviceVBase_ + linearSlot * tokenSizeBytesV_;

        // CopyBytes 只向流水提交 chunk，不在 K/V 或 request 边界清空流水。
        // 这样 K、V，以及下一个 miss 的搬运都可共用两个 UB buffer。
        CopyBytes(sourceK, destinationK, static_cast<uint32_t>(tokenSizeBytesK_));
        CopyBytes(sourceV, destinationV, static_cast<uint32_t>(tokenSizeBytesV_));
      }
    }
    while (pending_ != 0U) {
      DrainOne();
    }
    // 队列排空表示 MTE3 写指令已提交；此处再保证本核所有写入完成。
    AscendC::PipeBarrier<PIPE_ALL>();
  }

 private:
  // 大 payload 自动拆分，长度与地址偏移按字节计算。
  __aicore__ inline void CopyBytes(uint64_t source, uint64_t destination, uint32_t bytes) {
    uint32_t offset = 0U;
    while (offset < bytes) {
      const uint32_t remaining = bytes - offset;
      const uint32_t chunk = remaining < TRANSFER_CHUNK_BYTES ? remaining : TRANSFER_CHUNK_BYTES;
      SubmitChunk(source + offset, destination + offset, chunk);
      offset += chunk;
    }
  }

  /**
   * 保持至多一块已入队但尚未提交搬出的数据；每次先搬入下一块，再搬出前一块。
   * 瞬间队列可以有两块，故 depth=2 且 bufferCount=2。
   *
   * AllocTensor/FreeTensor 管理 buffer 的回收依赖，EnQue/DeQue 管理
   * 搬入到搬出的依赖；不能绕过队列手工重用同一 UB 地址。
   * metadata 环形队列与 LocalTensor FIFO 一一对应，尤其支持 K/V 长度不同。
   */
  __aicore__ inline void SubmitChunk(uint64_t source, uint64_t destination, uint32_t bytes) {
    AscendC::GlobalTensor<uint8_t> sourceGm;
    sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(source), bytes);
    AscendC::LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
    AscendC::DataCopyExtParams copyParams(1, bytes, 0, 0, 0);
    // 显式清零 pad 参数，避免原版未初始化字段；无需读取补齐后的无效字节。
    AscendC::DataCopyPadExtParams<uint8_t> padParams{};
    AscendC::DataCopyPad(local, sourceGm, copyParams, padParams);
    copyQueue_.EnQue(local);

    destinations_[tail_] = destination;
    lengths_[tail_] = bytes;
    tail_ = (tail_ + 1U) & 1U;
    ++pending_;
    if (pending_ == TRANSFER_BUFFER_COUNT) {
      DrainOne();
    }
  }

  // 只搬出 FIFO 最早的一块；下次 AllocTensor 复用它之前由队列保证写后读依赖。
  __aicore__ inline void DrainOne() {
    const uint32_t bytes = lengths_[head_];
    AscendC::GlobalTensor<uint8_t> destinationGm;
    destinationGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(destinations_[head_]), bytes);
    AscendC::LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
    AscendC::DataCopyExtParams copyParams(1, bytes, 0, 0, 0);
    // 精确写 bytes 字节，最后一个非对齐 chunk 不覆盖相邻 slot 的有效数据。
    AscendC::DataCopyPad(destinationGm, local, copyParams);
    copyQueue_.FreeTensor(local);
    head_ = (head_ + 1U) & 1U;
    --pending_;
  }

  AscendC::TPipe pipe_;
  AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> copyQueue_;
  uint64_t destinations_[TRANSFER_BUFFER_COUNT] = {};
  uint32_t lengths_[TRANSFER_BUFFER_COUNT] = {};
  uint32_t head_ = 0U;
  uint32_t tail_ = 0U;
  uint32_t pending_ = 0U;
  uint32_t aivNum_ = 0U;
  uint32_t aivIndex_ = 0U;
  __gm__ int32_t* missCount_ = nullptr;
  __gm__ int32_t* missTokens_ = nullptr;
  __gm__ int32_t* missSlots_ = nullptr;
  __gm__ int32_t* tokenToReq_ = nullptr;
  __gm__ int32_t* blockTable_ = nullptr;
  uint64_t hostKBase_ = 0U;
  uint64_t hostVBase_ = 0U;
  uint64_t deviceKBase_ = 0U;
  uint64_t deviceVBase_ = 0U;
  int64_t numRows_ = 0;
  int64_t maxRequests_ = 0;
  int64_t topk_ = 0;
  int64_t capacity_ = 0;
  int64_t maxNumBlocks_ = 0;
  int64_t hostNumBlocks_ = 0;
  int32_t blockSize_ = 0;
  int32_t tokenSizeBytesK_ = 0;
  int32_t tokenSizeBytesV_ = 0;
  uint64_t blockBytesK_ = 0U;
  uint64_t blockBytesV_ = 0U;
};

}  // namespace

extern "C" __global__ __aicore__ void sparse_kv_transfer(GM_ADDR missCount, GM_ADDR missTokens, GM_ADDR missSlots,
                                                         GM_ADDR tokenToReq, GM_ADDR blockTable, GM_ADDR hostCacheBases,
                                                         GM_ADDR activeRows, GM_ADDR residentK, GM_ADDR residentV,
                                                         GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  REGISTER_TILING_DEFAULT(SparseKvTransferTilingData);
  GET_TILING_DATA_WITH_STRUCT(SparseKvTransferTilingData, tilingData, tiling);
  const int32_t requestedRows = *reinterpret_cast<__gm__ int32_t*>(activeRows);
  const int64_t numRows = requestedRows > 0 && requestedRows < tilingData.maxRows
                              ? requestedRows
                              : (requestedRows > 0 ? tilingData.maxRows : 0);
  SparseKvTransferRuntimeKernel kernel;
  kernel.Init(missCount, missTokens, missSlots, tokenToReq, blockTable, hostCacheBases, residentK, residentV, numRows,
              tilingData.maxRequests, tilingData.topk, tilingData.capacity, tilingData.maxNumBlocks,
              tilingData.hostNumBlocks, tilingData.blockSize, tilingData.tokenSizeBytesK, tilingData.tokenSizeBytesV);
  kernel.Process();
}
