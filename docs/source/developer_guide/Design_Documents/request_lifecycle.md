---
title: Request Lifecycle
---

# 请求全链路

完整链路（含 Router 代理、PD 拆分、Mooncake 池化与传输、reasoning parser 与 tool call parser）见独立页面：

<a class="md-button md-button--primary" href="../../request_lifecycle.html" target="_blank" rel="noopener">
  打开请求全链路文档（HTML）
</a>

该页面覆盖：

- `curl` → Router → vLLM API Server → EngineCore → NPU Worker → Client 的 48 个调用节点
- 普通路由与 prefill/decode 两阶段路由的分叉
- Mooncake 两种形态：`AscendStoreConnector` 共享 KV 池 vs `MooncakeConnectorV1` 直连 P2P
- 池化 key 构造、lookup/get/put 时序与失败语义
- `reasoning_ended` 对结构化输出（grammar bitmask）的影响
- 非流式 `parser.parse()` 与流式 `parser.parse_delta()` 的解析顺序

也可以直接在浏览器中打开仓库文件 `docs/source/request_lifecycle.html`。
