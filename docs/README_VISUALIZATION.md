# 可视化图表使用指南

## 概述

模拟运行时会自动生成时间序列监控图表，所有图表保存在按运行名称组织的目录中。

## 输出目录结构

```
result/
└── YYYYMMDD_HHMMSS_xPxD_数据集名/
    ├── config.json                      # 运行配置
    ├── decode_00.jsonl                  # Decode 实例迭代日志
    ├── decode_01.jsonl
    ├── prefill_00.jsonl                 # Prefill 实例迭代日志
    ├── prefill_01.jsonl
    ├── request_traces.jsonl             # 请求追踪日志
    └── plots/                           # 可视化图表目录
        ├── throughput_over_time.png
        ├── queue_metrics_prefill_0.png
        ├── queue_metrics_prefill_1.png
        ├── queue_metrics_decode_0.png
        ├── queue_metrics_decode_1.png
        ├── sla_attainment_over_time.png
        ├── memory_usage_over_time.png
        └── time_series_samples.jsonl    # 采样数据
```

### 示例目录名
- `20260319_173142_2P2D_azure_code_500/`
  - `20260319_173142`: 运行时间戳 (YYYYMMDD_HHMMSS)
  - `2P2D`: 2个Prefill实例 + 2个Decode实例
  - `azure_code_500`: 数据集名称（从trace文件名提取）

## 可视化图表说明

### 1. throughput_over_time.png - 系统吞吐量
显示输入和输出吞吐量随时间的变化：
- **蓝线**: Input Throughput (prefill tokens/s)
- **橙线**: Output Throughput (decode tokens/s)

**用途**:
- 识别吞吐量瓶颈
- 观察负载变化模式
- 分析系统性能趋势

### 2. queue_metrics_{instance_id}.png - 队列指标（每个实例）
显示各个队列的长度变化：
- **waiting queue**: 等待队列长度
- **running**: 运行中的批次大小
- **bootstrap queue**: Bootstrap队列（prefill实例）
- **prealloc queue**: 预分配队列（decode实例）
- **transfer queue**: 传输队列（decode实例）
- **prealloc reserved**: 预留队列（decode实例）

**特殊标记**:
- **红色虚线**: 角色切换时刻
- **黄色注释框**: 切换方向 (P→D 或 D→P)

**用途**:
- 识别队列积压问题
- 验证角色切换效果
- 调试调度策略

### 3. sla_attainment_over_time.png - SLA达成率
显示SLA达成率随时间变化（1000请求滚动窗口）：
- **蓝线**: TTFT SLA达成率
- **橙线**: ITL SLA达成率
- **绿色虚线**: 总体SLA达成率（两者都满足）

**用途**:
- 监控SLA降级
- 识别性能问题时段
- 评估系统稳定性

### 4. memory_usage_over_time.png - 内存使用率
显示每个实例的KV cache利用率：
- 每个实例一条线
- Prefill实例用实线
- Decode实例用虚线
- Y轴: 0-100% 利用率

**用途**:
- 识别内存压力
- 调整内存分配策略
- 验证内存优化效果

## 配置参数

在 `config.py` 中的相关配置：

```python
# 流式统计（节省内存）
enable_streaming_metrics: bool = True
metrics_reservoir_size: int = 10000

# 时间序列监控
enable_monitoring: bool = True
monitoring_sample_interval: float = 1.0    # 1秒采样一次
monitoring_max_samples: int = 10000        # 最多保存10,000个样本
monitoring_sla_window: int = 1000          # SLA滚动窗口1000个请求
```

## 内存使用

所有监控功能使用固定内存（O(1)）：
- MetricsCollector: ~1-2 MB（Reservoir Sampling）
- TimeSeriesMonitor: ~2-3 MB（10K样本 + 1K SLA窗口）
- **总计**: 无论模拟多长，只占用 ~5 MB

这使得可以运行百万级甚至更长的模拟而不会内存爆炸！

## 使用示例

```python
from config import SimConfig
from core.engine import SimulationEngine

# 配置
config = SimConfig(
    trace_path="azure_code_500.csv",
    num_prefill_instances=2,
    num_decode_instances=2,
    # 监控功能默认启用
    enable_iteration_logging=True,  # 如需要详细的 iteration log，可启用
)

# 运行模拟
engine = SimulationEngine(config)
results = engine.run()

# 图表自动生成在 result/YYYYMMDD_HHMMSS_2P2D_azure_code_500/plots/
```

## 禁用监控（如果需要）

```python
config = SimConfig(
    trace_path="trace.csv",
    enable_monitoring=False,  # 禁用时间序列监控
)
```

## 访问采样数据

所有采样数据保存在 `plots/time_series_samples.jsonl`，可以用于自定义分析：

```python
import json

samples = []
with open('result/20260319_173142_2P2D_azure_code_500/plots/time_series_samples.jsonl', 'r') as f:
    for line in f:
        samples.append(json.loads(line))

# 分析采样数据
for sample in samples:
    print(f"Time: {sample['time']}")
    print(f"Input throughput: {sample['input_throughput']} tokens/s")
    print(f"SLA attainment: {sample['sla_attainment']}")
```

## 故障排查

如果图表没有生成：
1. 检查 `enable_monitoring=True`
2. 检查 `enable_iteration_logging=True`
3. 查看控制台输出中的错误信息
4. 确认 `result/` 目录有写权限

如果内存仍然爆炸：
1. 确认 `enable_streaming_metrics=True`
2. 检查是否有其他组件保留了请求引用
3. 验证 `cleanup_after_completion()` 被正确调用
