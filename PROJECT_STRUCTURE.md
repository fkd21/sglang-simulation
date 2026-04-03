# 项目结构说明

本文档描述了 sglang-simulation 项目的目录结构和组织方式。

## 目录结构

```
sglang-simulation/
├── core/                    # 核心引擎和调度逻辑
├── mechanisms/              # 各种策略机制实现
│   ├── policy_alpha_sim_v2.py
│   ├── policy_alpha_sim_v3.py
│   ├── policy_alpha_sim_v4.py
│   ├── policy_alpha_sim_v5.py
│   ├── partial_offload.py
│   ├── policy_monitor.py
│   └── kalman_filter_1d.py
├── metrics/                 # 指标收集和分析
├── memory/                  # 内存管理
├── request/                 # 请求处理
├── scheduling/              # 调度算法
├── policy/                  # 策略定义
├── instances/               # 实例配置
├── experiments/             # 实验脚本
│   ├── run_alpha_v2_only.py
│   ├── run_alpha_v3_only.py
│   ├── run_alpha_v4_only.py
│   ├── run_alpha_v5_only.py
│   ├── run_with_baseline.py
│   └── run_offload_with_protection_only.py
├── tests/                   # 测试文件（新增）
│   ├── test_baseline.py
│   ├── test_multiprocess.py
│   ├── test_single_config.py
│   ├── test_offload_pd_ratio_integration.py
│   ├── test_streaming.py
│   └── ... (其他单元测试)
├── docs/                    # 文档和分析报告（新增）
│   ├── EMA_SMOOTHING_GUIDE.md
│   ├── ITERATION_LOGGING.md
│   ├── OPTIMIZATION_FEATURES.md
│   ├── PERFORMANCE_ANALYSIS.md
│   ├── simulation_rule.md
│   ├── DECODE_PROTECTION_ANALYSIS.md
│   ├── OFFLOAD_PROTECTION_BUG.md
│   ├── SWITCHING_ANALYSIS_FINAL.md
│   └── ... (问题诊断和修复文档)
├── utils/                   # 工具函数
├── result/                  # 实验结果输出
├── config.py               # 配置文件
├── main.py                 # 主入口
└── policy_alpha.py         # Alpha策略主文件
```

## 主要模块说明

### core/
核心引擎实现，包括：
- 调度引擎
- 请求处理逻辑
- 资源管理

### mechanisms/
不同版本的策略机制：
- **policy_alpha_sim_v2.py**: Alpha策略v2版本
- **policy_alpha_sim_v3.py**: Alpha策略v3版本（改进预测）
- **policy_alpha_sim_v4.py**: Alpha策略v4版本（增强预算管理）
- **policy_alpha_sim_v5.py**: Alpha策略v5版本（集成Kalman滤波）
- **partial_offload.py**: 部分卸载机制
- **policy_monitor.py**: 策略监控
- **kalman_filter_1d.py**: 一维Kalman滤波器

### experiments/
实验运行脚本：
- **run_alpha_v2_only.py**: 仅运行Alpha v2
- **run_alpha_v3_only.py**: 仅运行Alpha v3
- **run_alpha_v4_only.py**: 仅运行Alpha v4
- **run_alpha_v5_only.py**: 仅运行Alpha v5
- **run_with_baseline.py**: 运行基线对比实验
- **run_offload_with_protection_only.py**: 运行带保护的卸载实验

### tests/
单元测试和集成测试：
- 基线测试
- 多进程测试
- 流式处理测试
- Offload和PD ratio集成测试
- 其他功能测试

### docs/
项目文档分为两类：

**指南和规则：**
- `simulation_rule.md`: 模拟规则说明
- `EMA_SMOOTHING_GUIDE.md`: EMA平滑指南
- `ITERATION_LOGGING.md`: 迭代日志说明
- `OPTIMIZATION_FEATURES.md`: 优化特性文档
- `README_VISUALIZATION.md`: 可视化说明

**问题分析和修复：**
- `DECODE_PROTECTION_ANALYSIS.md`: Decode保护分析
- `OFFLOAD_PROTECTION_BUG.md`: Offload保护问题
- `SWITCHING_ANALYSIS_FINAL.md`: 切换分析
- `INFEASIBLE_PROBLEM.md`: 不可行性问题
- `LP_PHASE1_PROBLEM.md`: LP Phase1问题
- `WINDOW_SIZE_PROBLEM.md`: 窗口大小问题
- `PD_RATIO_VERIFICATION.md`: PD比率验证
- `OFFLOAD_FIX_SUMMARY.md`: Offload修复总结
- `fix_offload_double_counting.md`: 修复重复计数

## 开发工作流

1. **运行实验**: 使用 `experiments/` 中的脚本
2. **查看结果**: 结果保存在 `result/` 目录
3. **运行测试**: 使用 `pytest tests/`
4. **查阅文档**: 参考 `docs/` 中的相关文档

## 数据文件

大型trace文件：
- `AzureLLMInferenceTrace_code_1week.csv`
- `AzureLLMInferenceTrace_conv_1week.csv`
- `azure_mixed_24h.csv`
- `azure_mixed_24h_25pct.csv`
- 其他采样和处理后的trace文件

## 清理说明

以下临时文件已被清理：
- `debug_*.py`: 调试脚本
- `analyze_*.py`: 分析脚本
- `verify_*.py`: 验证脚本
- `quick_*.py`: 快速测试脚本
- 临时提取和处理脚本
- 过时的测试文件
