# DSV4.1 MultiConnector 与 P 端异步 Store

## 适配范围

在 `ced-debug` 上扩展固定 DP 通信策略，支持 `MultiConnector` / `AscendMultiConnector`
包装同一 P/D 角色的 `MooncakeHybridConnector` 和 `AscendStoreConnector`，支持嵌套组合。
只包含 Store、未知子 connector、P/D 角色混合、D Store 开启 `consumer_is_to_load`
时不建立固定策略。其余 A5、TP/PP/CP=1、模型拓扑、容量、DSpark 和通信算法检查保持生效。

固定策略在启动时做一次 rank 间一致性检查，实际请求省去 runner 的 DP metadata allreduce。
它不取消 EngineCore 的协调，也不取消启动 profiling/warmup 的原有通信。

## 用户 D 配置

配置为 DP=16、TP=1、56 个请求、DSpark 5、最大图档位 336、token budget 600：

| 项目 | 每 rank token 容量 | global_bs |
| --- | --- | --- |
| target MC2 | 56 × 6 = 336 | 5376 |
| draft MC2，默认从 anchor 采样 | 56 × 5 = 280 | 4480 |

保持 `max_num_batched_tokens=600`、scheduler 的 speculative 预留预算、最大请求数和图档位。
`synthetic_acceptance_length` 不改变最大执行预算。

`AscendStoreConnector` 的 `kv_consumer` 默认不读取也不写入。
`load_async=true` 只控制已启用加载的执行方式，不能单独开启 Store 读取。
private pool 的 scheduler 会将 KV 加载失败设为终止请求；不会按外层
`kv_load_failure_policy=recompute` 就地恢复完整 prefill。
普通 D 抢占也仍保留现有错误保护。

成功建立策略后，日志包含：

```text
PD execution skips DP metadata synchronization: target=MC2 capacity=336 global_bs=5376,
draft=MC2 capacity=280 global_bs=4480; runtime_only=True.
```

## 用户 P 配置

将脚本中的 `--no-async-scheduling` 改为 `--async-scheduling` 后，以下流程需要分别验证：

- target eager、最大 chunk 24576、DP/EP=8，`mega_moe_max_tokens=196608` 对应每 rank 24576。
- DSpark eager / probabilistic，16 请求、默认 anchor 模式最大 80 draft tokens，独立 MC2 通信域。
- private pool + tail-window，P Store 可以读取共享 KV，但不会导入 private SWA ring；命中后仍需 bounded replay。
- **dynamic EPLB 必须关闭才能满足当前固定 MegaMoE 策略。** 开启时保留 Gloo 与原算法选择。

当前本地 `EPLBConfig` 不声明 `enable_omni_eplb` 和 `omni_config_file`。
`expert_map_record_path` 会使 `dynamic_eplb=True`，并且要求 `.json` 文件路径。
因此贴出的 EPLB 配置还需与实际部署版本核对；本次不放开其 MegaMoE 互斥条件。

## P Store 的异步保存与生命周期

原先非 layerwise 保存虽把 IO 投递到后台线程，但模型线程随后执行 `request_queue.join()`。
新路径仅在 private pool、异步 P、TP/PP/PCP/DCP=1、非 layerwise 时解除该等待。
同步 P、D、layerwise 与其他并行配置保留原有保存路径。

1. scheduler 在发布 Store metadata 时，对完整共享前缀 block 增加引用，记录 event ID。
   sender 可能重写已被 Store 淘汰的旧 chunk，因此不能只固定本轮新增 block。
   partial chunk、private SWA ring 和 compressor state 不进入异步保存范围。
2. 模型完成 KV 写入后记录 device event，投递保存任务后继续下发后续任务。
3. IO 线程等待 event，再读 KV 并执行 put。空 lookup 或异常也必须确认 event 后才发布 block 释放。
4. scheduler 汇总 worker 的 event ID 完成通知后，释放额外 block 引用。
   请求结束/抢占不能提前释放这些引用；接收线程也不清除尚未完成的异步保存计数。
5. 保存计数和 finished 标志在同一个锁内更新。新 chunk 入队时清除旧完成标志，避免旧 chunk 的
   完成通知导致新 chunk 的缓存提前释放。

DSV4.1 不使用 Mamba，因此 worker 的 event 完成上报不能只由 `use_mamba` 控制；
异步 private-pool P 同样需要上报。发送线程先发布 event 完成，再发布请求保存完成，
确保最后一个请求结束、EngineCore 停止轮询之前，引用释放信息已经可读取。

同步 Store load 的 replay put 仍等待 scheduler 验证加载成功；未验证的 put 不使用上述异步 block pin。
已验证 backlog 保留原先的等待方式。用户配置 `load_async=true` 时，加载阶段不执行对应的 replay put，
请求加载完成后才调度 bounded replay，不应把同步 load 的 backlog 等待泛化为每次 Store 命中。
本次没有重写 Store lookup/load 协议。

持有 block 引用不复制 KV，但会延长 block 的可回收时间。Store 吞吐不足时，KV 内存回收可能成为瓶颈。
输入准备中的 `_tail_query_start_loc` 也改为 CPU mirror 规划和 padding，复用 pinned staging
上传 QSL 与 tail 行号，不再直接构造 device tensor，也不回读 device 标量作为填充值。
每步独立 staging 保护未完成的 DMA；该修改同时用于 HBM 命中、Store 命中和未命中的 tail prefill。

## 验证与限制

CPU 测试执行真实策略、scheduler、worker 和发送线程方法。覆盖组合角色、嵌套、P/D 容量、
EPLB、同步路径、保存入队、后台完成、连续 chunk、并发计数、请求结束/抢占、失败和 replay 验证。
CPU 环境缺少完整 vLLM/NPU 安装，部分测试隔离导入或使用测试替身，不能替代精度和性能验证。

针对本次修改，110 项 DP 策略回归和 167 项 Store 回归通过，另有 68 个 Store 子用例通过。
tail-QSL 与共享 pinned 上传路径另有 77 项 CPU 回归，覆盖 int32/int64、padding、混合 batch、
纯 decode、未裁剪和连续 step 的延迟 DMA。
改动 Python 文件通过 Ruff 和 diff 空白检查。扩大到 Store 目录时，两个文件因缺少硬件构建信息/
vLLM 模块无法收集，两个未修改的 coordinator 用例在 mock 返回值解包处失败，不能报告全量通过。
`bash format.sh ci` 因 Node 18 不满足 markdownlint-cli 的 Node >=20 要求而中断。

NPU 上需要覆盖：P/D 异步开关组合、完整/部分 Store 命中、连续 chunk、put 慢于模型、加载失败、
KV 压力与请求结束，比较正常请求的模型输出和 KV 内容。prof 应确认 P 模型线程不再等待普通
Store put；tail-QSL 路径应不再出现由 device scalar fill 引起的同步，需要实测 D FULL graph capture/replay。

## 关闭 EPLB 后的四条路径检视

| 场景 | Store 执行与缓存保护 | 结论范围 |
| --- | --- | --- |
| P 同步 | 普通 put 保留模型线程等待；完成前请求块不能回收 | 原有同步保存语义保留 |
| P 异步 | 普通完整 chunk 投递后台；event ID 完成信息释放额外 block 引用；请求完成通知最后发布 | 修复非 Mamba 上报遗漏及最后一次轮询竞态；replay backlog 仍保留等待 |
| D 同步 | 当前 consumer 配置下 Store 不读取、不写入；Mooncake 拉取共享 KV 与 private ring | 固定 target/draft MC2 策略不依赖异步开关 |
| D 异步 | Store 默认仍为旁路；D 输入和 accepted-count 设备路径沿用现有实现 | 图执行保留；profiling/warmup 仍使用原有容量和同步规则 |

以上按用户脚本的 `consumer_is_to_load=false`、`consumer_is_to_put=false` 默认值检查。
打开 D Store 读取不在此 decode-only 策略内；private pool 初始化也会拒绝这项配置。
没有 NPU 端到端运行和精度评测结果，不能据此宣称四种组合已实机验证。
