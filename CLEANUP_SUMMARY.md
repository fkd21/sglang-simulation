# 项目整理总结

**整理日期**: 2026-04-03

## 整理目标

将项目中的临时文件、测试文件和文档进行系统化组织，提高项目结构的清晰度和可维护性。

## 主要改动

### 1. 创建新目录结构

- **tests/**: 集中存放所有测试文件
- **docs/**: 集中存放所有文档和分析报告

### 2. 测试文件整理

#### 移动到 tests/ 目录的文件：
- `test_baseline.py` → `tests/test_baseline.py`
- `test_multiprocess.py` → `tests/test_multiprocess.py`
- `test_single_config.py` → `tests/test_single_config.py`
- `test_offload_pd_ratio_integration.py` → `tests/test_offload_pd_ratio_integration.py`
- `test_streaming.py` → `tests/test_streaming.py`
- `test_streaming_fix.py` → `tests/test_streaming_fix.py`
- `test_streaming_quick.py` → `tests/test_streaming_quick.py`

#### 删除的过时测试文件：
- `test_bug_detection.py`
- `test_cache_behavior.py`
- `test_drop_tracking.py`
- `test_ema_manual.py`
- `test_iteration_logging.py`
- `test_memory_fixes.py`
- `test_optimizations.py`
- `test_parallel_lp.py`
- `test_parallel_simple.py`
- `test_periodic_plots.py`
- `test_larger_window.py`

### 3. 文档文件整理

#### 移动到 docs/ 目录的文档：

**指南和参考文档：**
- `EMA_SMOOTHING_GUIDE.md`
- `ITERATION_LOGGING.md`
- `OPTIMIZATION_FEATURES.md`
- `PERFORMANCE_ANALYSIS.md`
- `README_VISUALIZATION.md`
- `codereviewer.md`
- `simulation_rule.md`

**问题分析和修复文档：**
- `DECODE_PROTECTION_ANALYSIS.md`
- `INFEASIBLE_PROBLEM.md`
- `LP_PHASE1_PROBLEM.md`
- `OFFLOAD_FIX_SUMMARY.md`
- `OFFLOAD_PROTECTION_BUG.md`
- `PD_RATIO_VERIFICATION.md`
- `SWITCHING_ANALYSIS_FINAL.md`
- `WINDOW_SIZE_PROBLEM.md`
- `fix_offload_double_counting.md`

### 4. 临时脚本清理

#### 删除的调试脚本：
- `debug_alpha_v2.py`
- `debug_lp_solver.py`
- `debug_offload.py`
- `debug_switching.py`
- `deep_dive_switching.py`

#### 删除的分析脚本：
- `analyze_alpha_v2_conditions.py`
- `analyze_lp_solver_phase1.py`
- `analyze_offload_behavior.py`
- `analyze_offload_logic_bug.py`
- `analyze_offload_problem.py`
- `analyze_phase1_queue_wait.py`
- `analyze_prefill_imbalance.py`
- `analyze_role_transitions.py`
- `analyze_switching_comparison.py`
- `analyze_switching_context.py`
- `analyze_window_size.py`

#### 删除的验证脚本：
- `verify_alpha_v5_prediction.py`
- `verify_budget_scaling.py`
- `verify_double_counting.py`
- `verify_fix.py`
- `verify_offload_fix.py`
- `verify_pd_ratio_after_switch.py`

#### 删除的快速测试脚本：
- `quick_alpha_v2_test.py`
- `quick_lp_debug.py`
- `quick_test_fix.py`
- `quick_window_test.py`

#### 删除的数据处理脚本：
- `extract_1h_trace.py`
- `extract_24h_trace.py`
- `extract_and_merge_monday.py`
- `extract_and_merge_monday_optimized.py`
- `run_extract_merge_monday.sh`
- `generate_plots.py`
- `profile_simulation.py`

#### 删除的其他临时文件：
- `check_queue_ordering.py`

### 5. 保留在根目录的核心文件

- `config.py`: 项目配置
- `main.py`: 主入口
- `policy_alpha.py`: Alpha策略主文件
- `policy_v1.py`: 策略v1版本
- `PROJECT_STRUCTURE.md`: 项目结构说明（新增）
- `CLEANUP_SUMMARY.md`: 本文件（新增）

## 整理效果

### 清理前：
- 根目录有 60+ 个 Python 和 Markdown 文件混杂
- 测试文件、文档、临时脚本分散在根目录
- 难以区分核心代码和临时文件

### 清理后：
- 根目录仅保留 4 个核心 Python 文件
- 测试文件统一在 `tests/` 目录（26个测试文件）
- 文档统一在 `docs/` 目录（16个文档）
- 删除了 50+ 个临时和过时文件
- 项目结构清晰，易于维护

## 未来建议

1. **测试文件管理**：
   - 在 `tests/` 目录中使用 pytest
   - 考虑按功能模块创建子目录（如 `tests/mechanisms/`, `tests/core/`）

2. **文档管理**：
   - 在 `docs/` 中创建 README.md 索引
   - 可以按类型分类（如 `docs/guides/`, `docs/analysis/`）

3. **开发流程**：
   - 临时调试脚本统一放在 `scripts/debug/` 目录
   - 数据处理脚本放在 `scripts/data/` 目录
   - 完成调试后及时清理或归档

4. **版本控制**：
   - 定期清理不再需要的临时文件
   - 重要的分析结果应归档到 `docs/` 而不是保留脚本

## 统计数据

- **删除文件**: 50+ 个
- **移动文件**: 23 个
- **新增目录**: 2 个
- **新增文档**: 2 个
- **保留核心文件**: 4 个

整理完成后，项目更加整洁、结构化，便于团队协作和长期维护。
