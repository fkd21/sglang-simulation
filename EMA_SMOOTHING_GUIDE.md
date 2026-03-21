# EMA Smoothing for Throughput-Based Policy

## 概述

我们为throughput-based policy添加了**自适应EMA（指数移动平均）平滑**功能，以解决工作负载不稳定导致的频繁切换问题。

## 核心特性

### 1. **自适应Alpha参数**

EMA使用可变的alpha参数，根据工作负载变化自动调整响应速度：

- **稳定期** (`base_alpha = 0.3`)：当工作负载平稳变化时，使用较小的alpha进行平滑，过滤噪声
- **剧烈变化期** (`max_alpha = 0.7`)：当检测到比例（ratio）变化超过阈值时，使用较大的alpha快速响应

### 2. **灵敏度检测**

通过比较当前ratio和EMA ratio的变化百分比来检测剧烈变化：

```python
ratio_change = |current_ratio - ema_ratio| / ema_ratio

if ratio_change > sensitivity_threshold:
    使用 max_alpha (快速响应)
else:
    使用 base_alpha (平滑噪声)
```

### 3. **工作原理**

```
原始throughput → EMA平滑 → 计算ratio → 确定P/D配置 → 切换决策
   ↓                ↓
 有噪声         平滑但灵敏
```

## 配置参数

### 方法1：通过config.py配置（推荐）

在`config.py`中添加参数（需要先更新SimConfig类）：

```python
config = SimConfig(
    trace_path="your_trace.jsonl",
    num_prefill_instances=2,
    num_decode_instances=2,
    enable_switching=True,
    switch_policy="throughput",
    monitor_interval_s=5.0,

    # EMA平滑参数
    ema_base_alpha=0.3,              # 默认平滑强度 (0-1, 越小越平滑)
    ema_max_alpha=0.7,               # 剧烈变化时的响应速度 (0-1, 越大越灵敏)
    ema_sensitivity_threshold=0.2,   # 变化检测阈值 (0-1, 超过此阈值认为是剧烈变化)
)
```

### 方法2：通过PolicyMonitor直接配置

```python
from mechanisms.policy_monitor import PolicyMonitor

policy_monitor = PolicyMonitor(
    policy="throughput",
    monitor_interval_s=5.0,

    # EMA参数
    ema_base_alpha=0.3,
    ema_max_alpha=0.7,
    ema_sensitivity_threshold=0.2,
)
```

## 参数调优指南

### `ema_base_alpha` (基础平滑系数)

- **范围**: 0.0 - 1.0
- **默认**: 0.3
- **效果**:
  - **越小 (0.1-0.2)**: 更平滑，对短期波动不敏感，适合噪声大的workload
  - **越大 (0.4-0.5)**: 更快响应，但可能被噪声影响
- **推荐**:
  - 高噪声workload: 0.2
  - 中等噪声: 0.3 (默认)
  - 低噪声但需要快速响应: 0.4

### `ema_max_alpha` (最大响应速度)

- **范围**: 0.0 - 1.0
- **默认**: 0.7
- **效果**:
  - **越小 (0.5-0.6)**: 即使检测到剧烈变化也保持一定平滑
  - **越大 (0.8-0.9)**: 剧烈变化时几乎立即跟随新值
- **推荐**:
  - 保守策略: 0.6
  - 平衡策略: 0.7 (默认)
  - 激进策略: 0.8-0.9

### `ema_sensitivity_threshold` (灵敏度阈值)

- **范围**: 0.0 - 1.0
- **默认**: 0.2 (ratio变化20%)
- **效果**:
  - **越小 (0.1)**: 更容易触发高alpha，对小变化也快速响应
  - **越大 (0.3-0.5)**: 只有大的变化才触发高alpha
- **推荐**:
  - 频繁小波动: 0.3-0.5 (避免过度响应)
  - 中等变化: 0.2 (默认)
  - 需要快速捕捉小趋势: 0.1

## 使用场景

### 场景1: 稳定workload，偶尔有突发

```python
ema_base_alpha=0.2,              # 平时很平滑
ema_max_alpha=0.8,               # 突发时快速响应
ema_sensitivity_threshold=0.25,  # 25%变化才算突发
```

### 场景2: 持续波动的workload

```python
ema_base_alpha=0.3,              # 适度平滑
ema_max_alpha=0.6,               # 即使剧烈变化也保持一定平滑
ema_sensitivity_threshold=0.3,   # 较高阈值，避免频繁切换alpha
```

### 场景3: 需要快速响应

```python
ema_base_alpha=0.4,              # 基础就比较灵敏
ema_max_alpha=0.8,               # 剧烈变化时更快
ema_sensitivity_threshold=0.15,  # 低阈值，快速检测变化
```

### 场景4: 极度不稳定，需要强力平滑

```python
ema_base_alpha=0.1,              # 强力平滑
ema_max_alpha=0.5,               # 即使检测到变化也保持克制
ema_sensitivity_threshold=0.4,   # 高阈值，只响应真正的大变化
```

## 验证效果

### 1. 查看日志输出

运行仿真时，context中会包含平滑后的值：

```python
{
    "input_throughput": 1050.0,           # 原始值
    "output_throughput": 525.0,           # 原始值
    "smoothed_input_throughput": 1015.0,  # EMA平滑后
    "smoothed_output_throughput": 507.5,  # EMA平滑后
    "ratio": 2.0,                         # 基于平滑值计算的ratio
}
```

### 2. 对比切换频率

记录添加EMA前后的切换次数：

```python
# 运行后检查
switches = engine.metrics_collector.switches
print(f"Total switches: {len(switches)}")
```

预期：添加EMA后，切换次数应该减少，特别是在workload波动但整体趋势稳定的情况下。

### 3. 运行测试

```bash
python3 test_ema_manual.py
```

应该看到所有4个测试通过：
- ✓ EMA初始化
- ✓ 稳定workload下的平滑
- ✓ 剧烈变化时的高灵敏度
- ✓ 自定义参数

## 技术细节

### EMA公式

```python
new_ema = alpha * current_value + (1 - alpha) * old_ema
```

- alpha = 0.3: 新值占30%，旧EMA占70%
- alpha = 0.7: 新值占70%，旧EMA占30%

### 自适应逻辑

```python
current_ratio = input_throughput / output_throughput
ema_ratio = ema_input / ema_output

ratio_change = |current_ratio - ema_ratio| / ema_ratio

if ratio_change > sensitivity_threshold:
    alpha = max_alpha  # 检测到剧烈变化
else:
    alpha = base_alpha  # 正常平滑
```

### 初始化

第一次evaluate时，直接使用原始值初始化EMA：

```python
if _ema_input_throughput is None:
    _ema_input_throughput = input_throughput
    _ema_output_throughput = output_throughput
```

## 与其他策略的对比

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **无平滑** | 响应最快 | 噪声导致频繁切换 | 非常稳定的workload |
| **固定EMA** | 简单平滑 | 无法应对突变 | 噪声小且变化缓慢 |
| **自适应EMA** (当前) | 平滑+灵敏 | 需要调参 | 大多数场景 ✓ |
| **ARIMA预测** | 可预测趋势 | 复杂，需要历史数据 | 有明显周期性pattern |

## 故障排查

### 问题1: 仍然切换太频繁

**解决方案**:
- 降低 `ema_base_alpha` (如 0.2 或 0.1)
- 提高 `ema_sensitivity_threshold` (如 0.3 或 0.4)
- 增加 `global_cooldown_s` (如 60.0)

### 问题2: 响应太慢，错过负载变化

**解决方案**:
- 提高 `ema_base_alpha` (如 0.4 或 0.5)
- 提高 `ema_max_alpha` (如 0.8 或 0.9)
- 降低 `ema_sensitivity_threshold` (如 0.1 或 0.15)

### 问题3: 不确定是否在工作

**验证方法**:
```python
# 在policy.evaluate()后打印
print(f"Raw input: {input_throughput}, Smoothed: {smoothed_input}")
print(f"Raw output: {output_throughput}, Smoothed: {smoothed_output}")
```

如果smoothed值和raw值不同，说明EMA在工作。

## 未来扩展

可能的改进方向：

1. **动态调整阈值**: 根据历史波动性自动调整sensitivity_threshold
2. **多时间尺度EMA**: 短期EMA检测快速变化，长期EMA确定趋势
3. **结合趋势检测**: 在EMA基础上增加趋势方向判断
4. **ARIMA集成**: 在EMA基础上添加预测能力

## 相关文件

- [mechanisms/policy_throughput_sim.py](mechanisms/policy_throughput_sim.py) - EMA实现
- [mechanisms/policy_monitor.py](mechanisms/policy_monitor.py) - 参数传递
- [tests/test_throughput_policy.py](tests/test_throughput_policy.py) - 单元测试
- [test_ema_manual.py](test_ema_manual.py) - 手动测试脚本

## 总结

EMA平滑通过自适应alpha参数，在平滑噪声和保持灵敏度之间取得了良好的平衡：

- ✅ 减少因短期波动导致的不必要切换
- ✅ 保持对真实负载变化的快速响应
- ✅ 简单易用，参数直观
- ✅ 无需历史数据存储，计算开销小

对于不稳定的workload，这是一个比完整ARIMA更轻量但同样有效的解决方案。
