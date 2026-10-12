# v2_optimized —— (B,M) 多核切分版，附真实问题清单

`colbert_maxsim_v2.cpp` 单文件：tiling 结构体 + tiling 计算 + device
kernel + host `main()`。Matmul/ReduceMax 调用方式原样复用 v1（docx 真实
参考）验证过的 API，新增部分是 design.md 要求、v1 没做的多核切分。

## 先回答："这个方案到底用没用到 batch、并行这些优化？"

**用了，但不是"同时处理整批"的那种 batch 优化，是"把 batch 维度也
摊开当并行单位"——这是两回事，容易搞混，重新说清楚：**

- **没有用的优化**："batch 化计算"，即把多个 batch 的 Q/Doc 拼在一起
  一次性喂给 Cube 做更大的矩阵乘（提高 Cube 利用率/减少 kernel 发射
  次数的常见手法）。v1、v2 都没做——每个 (batch, M子块) 任务块仍然是
  独立的一次 `Matmul::SetTensorA/SetTensorB/IterateAll`，batch 之间
  没有在 Cube 层面被"打包"成一次更大的矩阵乘。
- **真正用到的优化**：**把 batch 当成任务并行的一个维度，和 M 轴联合
  切分后扔给硬件的多个 AI Core 并行跑**——即 `idx = b * mTilesPerBatch
  + mTileIdx`，`for (idx = GetBlockIdx(); idx < totalBlocks; idx +=
  coreNum)`。这是"任务级并行"（task parallelism），不是"数据级
  batching"。design.md 说的"多核把 (B,M) 联合切分，打散到全部 AI
  Core"指的就是这种并行，v2 做到了，v1 没做到（v1 是"一个 batch 占一
  个核，串行发射"，并行度 = min(B, 等待硬件调度)，而不是
  min(totalBlocks, 核数)）。

**结论**：v2 相对 v1 用上的优化是"任务并行度从 B 提升到 B×mTilesPerBatch"，
不是"batching 矩阵乘计算量"。如果想要后一种优化（真正的 batch 化矩阵
乘，比如把多个小 batch 的 Q 拼成一个更大的 M 维度一次性进 Cube），
v2 现在的结构做不到——这是一个明确的、留给下一步的优化方向，不是
已经做了的事，之前的 README 没把这两者分开讲清楚，是之前文档不对的
地方。

## 重新审视代码本身，发现的真实问题（之前 README 没提到）

1. **M 轴尾块和 Cube tiling 形状不匹配（已修复）**：
   `mTile`（调度层 M 分块大小）原来直接取 `baseM*2` 再 cap 到 M，
   不保证整除 M——但 `GenerateCubeTiling` 只在 host 侧生成一次固定的
   `TCubeTiling`（shape 写死为 `mTile x baseK x D`），所有 `(B,M)`
   任务块共用这一份 tiling。如果 `mTile` 不整除 `M`，最后一个 M 子块
   的真实 `mLen < mTile`，但 Matmul 对象配置的 shape 还是整段 `mTile`
   ——这会导致尾块算出错误结果（不是编译错误，是运行时数值错误，
   比 placeholder 代码的"看得出来的 bug"更隐蔽）。
   **修复**：现在 `mTile` 的选取逻辑改成和 `baseK` 一样——找一个能整除
   `M` 的候选值，找不到就退化成 `mTile = M`（每个 batch 一个块，没有
   尾块）。代价是牺牲了一点调度灵活性（`mTile` 不再总是 `baseM*2`），
   换来"不整除就等价于没切"，彻底消灭这类尾块-tiling 不匹配问题，而
   不是指望运行时凑巧对上。

2. **Query 没有本地驻留复用，比 `code/ascendc_colbert_maxsim` 的设计
   倒退了一步**：`code/ascendc_colbert_maxsim/op_kernel/
   colbert_maxsim_kernel.cpp` 里显式用 `qBuf_`（`TBuf`）把 Q 子块
   `DataCopy` 进 UB 一次，K 轴循环里反复复用这块本地内存——对应
   design.md"Query 常驻，一次搬运、全程复用"的优化。但 `v2_optimized`
   里 `matmulObj_.SetTensorA(qGm_[...])` 直接传 GM 地址，K 轴每次
   迭代都重新 `SetTensorA`，Q 的实际搬运时机和次数完全交给 Matmul
   内部处理——**没有验证过 Matmul 是否会对同一个 A 地址做重复 GM 读取**。
   如果会（大概率会，因为每次 `IterateAll` 调用在 Matmul 眼里是独立
   一次矩阵乘），那么 Q 这部分数据会被重复搬运 `K/baseK` 次，design.md
   强调的"Query 复用省 MTE 带宽"这个优化在 v2 里实际上**没有**真正
   生效，这是之前 README 没发现、需要承认的问题。
   这部分不在本次改动范围内直接修（涉及把 AType 从 GM 换成 VECIN、
   额外管理一块 Q 的本地缓冲、改 `GenerateCubeTiling` 的 `SetAType`，
   改动面比刚才的 M 尾块 bug 大），列入下面的后续修复清单。

3. **Host 侧 `<<<coreNum, 0, stream>>>` 这种三尖括号 kernel 发射写法，
   是 docx 参考（v1）里就有的写法**，适合单算子编译测试场景（配合
   `aclrtlaunch_` 宏或类似机制），不是正式算子打包进 `msOpGen`
   工程、注册给 `torch_npu`/`ACLNN` 之后的标准调用方式。这不是 bug，
   但要提醒：`v2_optimized/colbert_maxsim_v2.cpp` 目前是"单算子裸测
   试"形态，不能直接当成最终产物接入生产 pipeline，还需要走
   `code/ascendc_colbert_maxsim/README.md` 里写的 `msopgen` 流程重新
   打包。

## 已验证（无需 NPU）

`(B,M)` 多核切分 + 跨核原子加这部分调度逻辑，用
`test/cpu_twin_reference_v2.py` 纯 Python 精确复刻验证过，三种切分
配置全部通过（含多核不整除、K 轴不对齐尾块）：

```bash
# 从 code2/ 目录下运行
PYTHONPATH=../code python3 v2_optimized/test/cpu_twin_reference_v2.py
```
```
cfg={'m_tile': 16, 'k_base': 64, 'core_num': 32}  max_diff=0.000244  PASS
cfg={'m_tile': 64, 'k_base': 256, 'core_num': 1}  max_diff=0.000244  PASS
cfg={'m_tile': 16, 'k_base': 37, 'core_num': 32}  max_diff=0.000244  PASS
结论: v2 的多核切分 + 跨核原子加汇总逻辑正确
```

**但要说明这份验证的边界**：`cpu_twin_reference_v2.py` 验证的是
"调度分配 + 数值结果对不对"，不验证"Matmul 内部有没有重复搬运 Q"
这类性能层面的问题——第 2 条 Query 复用缺失的问题，纯 Python 逻辑
孪生测不出来，必须上真机跑 `msprof` 看 MTE 带宽才能确认。

## 下一步（按优先级）

1. 真机编译 `colbert_maxsim_v2.cpp`，先排查编译报错。
2. 用 `msprof` 确认 Query 是否被重复搬运（上面第 2 条），如果确实是
   热点，把 Q 换成显式本地缓冲复用（参考
   `code/ascendc_colbert_maxsim` 的 `qBuf_` 写法）。
3. 对比 v1（按 batch 发射）和 v2（(B,M) 联合切分）在 `B=4` 这种小
   batch 场景下的 Task Duration / AI Core 利用率，验证"任务并行度
   提升"这个优化是否真的带来了预期的加速。
4. 如果后续需要"真正的 batch 化矩阵乘"（见本文开头的区分），那是
   在当前结构之上的另一层优化，需要重新设计 Cube 侧的 shape 规划，
   不是这一版的范围。
