# A5 MTP 按请求联合 Sparse KV Plan

## 分支与目的

基于 `chore/simtv3-dependency-audit` 的 `7d2e639a0`（ABCDF、1024 SIMT 线程），
在 `feat/mtp-joint-plan` 实现实验性 MTP 联合规划。未修改 MemFabric、PD 或 Host 池所有权。
目标是减少同一请求多条 query 的重复 resident 扫描和重复缺失 KV 搬运，不改变每条 query 的 TopK 选择。

## 数据通路

保持 `Indexer → Plan → Transfer → SFA`，没有新增 kernel launch 或设备结果的 D2H 判断。
manager 在配置了草稿 token 时启用 `joint_mtp`。不开 MTP 时走原 ABCDF 逐行规划。

Plan 在设备端将 `token_to_req` 中连续属于同一请求的行组成一组。
按组序号分配 AIV，避免用组首行对核数取模造成宽度为 4 时只利用四分之一核的情况。
每组由一个 AIV 的 1024 个 SIMT 线程处理，不使用跨 AIV 同步。
非连续的同请求片段独立处理，仍然正确，但不跨片段共享。

1. 逐行验证 token 范围、可见长度和 Host block table；无效项不参与联合需求。
2. 在哈希表中对所有合法 TopK 去重。每个 tile 做一次计数，发现并集大于 capacity 即停止。
3. 并集能放下时，只扫描首行的一套 LRU 状态，以各行稳定前缀的最小值失效旧后缀。
4. 为联合 miss 安排槽位，保留本轮所有行需要的 hit，更新一次 LRU。
5. 各行保留原 TopK 顺序及重复位置，输出各自的 `currentSlots`。重复 token 可以指向同一槽位，
   但不减少 SFA 的 TopK 条目数。未来位置即使被其他行加载，本行仍输出 -1。
6. `residentRows` 指向组首行；Plan 同时写 SFA 的 resident block table，使多行引用同一物理 resident 区。
7. Transfer 根据 `residentRows` 选择目标区，每个唯一 miss 只搬一次。

联合 miss 最多 C 项，也不超过组内 W*T 项，所以沿用该组已有的 W*T 个 miss 输出位置。
每个逻辑行承载最多 T 条搬运记录，记录对应的源请求相同；所有记录的目标都指向共享首行。
这不是每个 follower 都重新搬一次。Transfer 完成整个组后，SFA 才读取结果。

## 容量溢出和状态切换

并集大于 C 时，联合尝试尚未改动任何 resident 状态，整个组回退原 ABCDF 逐行路径，绝不截断联合 TopK。
单行组也走 ABCDF。此实现的溢出回退由组所在 AIV 顺序处理各行；低重合率下可能比原逐行多 AIV 更慢。

每个规划层有持久化 `rowModes`，区分 private、joint leader、follower。
private/joint 转换、follower 重新获得私有区时，重置相应行的缓存身份，强制冷加载，
防止将旧布局的元数据误用于已覆盖或没有填充的 resident 数据。
请求身份变化仍使用 `lastReqIds` 重置。首行位置改变可能降低命中率，不会将别的请求 KV 当作 hit。

普通 `skip_topk` 层仍只执行 Transfer，复用 Plan 所属层的 miss 列表、currentSlots 和共享行映射；
目标基址换成本层 resident 区。MTP 层沿用原有规则，行映射压缩后重新 Plan。

联合路径使用桶顺序分配新 miss，保留旧 hit 的 LRU 相对顺序。
因此缓存策略和具体槽号不要求与逐行 ABCDF 相同；正确性比较应检查每行实际读取的 KV，不能比较整份 LRU 状态逐字相等。

## 内存与图模式

不缩小或扩张已有 KV resident 分配；各行备用区域保留用于溢出回退。此版本目标是降低扫描和传输，
不声称节省 resident HBM 容量。增加的持久元数据为每层 R 个 int32 模式和全局 R 个 int32 owner。
SFA block table 使用原有缓冲区，不额外分配。

联合临时区为 `keys[H] + slots[H] + partition[C]`，H 为至少 2*C 的 2 次幂。
实际 scratch 大小取联合路径与 ABCDF 路径需求的最大值，二者不同时存活。
C=4096、T=2048 时联合临时区 80 KiB，ABCDF 72 KiB，取 80 KiB，加原有 288 字节 Scan 区，仍走 UB。
超过 112 KiB 预算时按同样大小分配持久 GM workspace；Host tiling 与 Python 分配公式一致。

所有新增 tensor 在 manager 初始化时分配。是否合并及溢出选择均在 kernel 中完成，
Torch schema 标记新增可写 tensor，fake/meta 签名同步更新。没有热路径 .item() 或新增动态 tensor 分配。
这维持图捕获所需的地址稳定性，不代表整条 Attention 的图兼容性已通过 A5 验证。

## 本地验证

```bash
python tests/ut/kv_offload/joint_plan/check_joint_plan.py --sanitize
```

测试提取实际 ABCDF、联合 Plan 辅助函数，用 32 个 CPU 线程模拟 barrier、shuffle 和原子操作。
包含重复 TopK、未来/无效位置、不同可见长度、覆盖旧后缀、请求身份变更、不同组宽度、
共享→溢出→共享、active rows 变化、第二层复用同一搬运计划，以及 LRU 槽位排列和页表检查。
本地 ASAN/UBSAN 回归通过：6495 个有效位置校验、316 次共享组、33 次溢出组。
Python lint、语法、格式及 diff 检查通过，Python/C++/Torch schema/meta 参数数目核对一致。
Plan 和 Transfer 通过本机 CANN 9.2 beta 的 bisheng 语法检查；该独立检查使用临时 glibc 兼容头与
模拟生成的 tiling 读取宏，不能替代正式 OPP 构建，未生成或验证可部署的算子包。

ASAN/UBSAN 可检查 CPU 访问范围；不模拟 NPU UB/GM 地址空间、DCache、1024 线程资源和硬件同步。

## 服务器验收

本次变更了自定义算子参数，必须重建配套的 OPP/ACLNN 库和 `_C` 扩展，不能混用旧二进制。
按当前部署的正常 A5 构建安装流程操作，再运行：

```bash
python tests/ut/kv_offload/joint_plan/check_joint_plan_npu.py \
  --extension /absolute/path/to/_C.abi3.so --device 0
python tests/ut/kv_offload/joint_plan/check_joint_plan_npu.py \
  --extension /absolute/path/to/_C.abi3.so --device 0 \
  --capacity 4096 --topk 2048 --requests 32 --rows-per-request 3
```

脚本同一二进制对照 `joint_mtp=False/True`，校验 eager 和 graph replay 下真实 Transfer 后的内容，
并通过 SFA 页表解析槽位。使用 HBM 源数据隔离地址/搬运逻辑，不能替代 Host DDR、PD、实际 SFA 或整网测试。
报告的 event 时间为重复调用 Plan+Transfer（或图回放），不是整网 token 延迟。

后续整网验收需覆盖 TP=1 PD、MTP 接受与拒绝、多请求换位、Indexer 跨层复用、真实 SFA 输出，
并对比每输出 token 的端到端耗时、Plan/Transfer 时间和重复搬运量。
高并集重合率有望受益；溢出、请求数少造成的核利用率下降、组宽变化导致的冷重置都可能抵消收益。
当前未进行 A5 上机运行或整网性能测量，不承诺加速比例。
