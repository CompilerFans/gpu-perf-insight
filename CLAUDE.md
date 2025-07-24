# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`perf_count` 是一个GPU性能测试对比数据统计图表生成工具，用于分析不同并行计算算法、不同数据量下多个GPU的吞吐量性能对比。支持多CSV文件输入和交互式Plotly图表。

## Development Commands

```bash
# 安装依赖
pip install -r requirements.txt

# 创建示例数据集
python perf_analyzer_plotly.py --create-sample

# 单GPU性能分析
python perf_analyzer_plotly.py \
  --csv-files rtx_4090_performance.csv \
  --labels "RTX 4090" \
  --output perf_single.html

# 多GPU性能对比
python perf_analyzer_plotly.py \
  --csv-files rtx_4090_performance.csv a100_performance.csv h100_performance.csv \
  --labels "RTX 4090" "A100" "H100" \
  --output gpu_comparison.html \
  --width 1400 --height 900

# 快速运行demo构建脚本
bash build.sh
```

## Project Architecture

### Core Components

- `perf_analyzer_plotly.py`: 主要的性能分析工具 (Plotly版本)
  - 支持命令行参数输入多个CSV文件
  - 使用Plotly生成交互式折线图
  - 每个CSV文件可设置独立的标签(如GPU名称)

- `perf_analyzer.py`: 旧版matplotlib分析工具
- `demo.py`: matplotlib版本演示脚本  
- `font_fix.py`: 中文字体检测和修复工具(仅matplotlib需要)
- `requirements.txt`: Python依赖包列表
- `README.md`: 英文使用说明文档

### Key Features

1. **多文件对比**: 支持同时加载多个CSV文件，每个文件设置独立标签
2. **交互式图表**: 使用Plotly生成可缩放、可悬停的交互式折线图
3. **并行算法支持**: 专门针对sort、reduce、scan、matmul、conv2d等并行计算算法
4. **命令行界面**: 完整的命令行参数支持，便于批处理和自动化
5. **报告生成**: 自动生成性能统计分析报告

### Data Structure

CSV文件数据格式(无需GPU列):
- `algorithm`: 算法名称 (sort, reduce, scan, matmul, conv2d等)
- `data_size`: 数据量规模，以MB为单位的数值 (如1, 16, 1024)
- `throughput`: 吞吐量，单位为GB/s

GPU标签通过命令行参数 `--labels` 指定，不在CSV中存储。

### Code Architecture

核心类 `PerformanceAnalyzer`:
- `load_dataset()`: 加载CSV文件并验证数据格式
- `create_comparison_line_chart()`: 生成交互式Plotly折线图
- `calculate_percentage_differences()`: 计算GPU间性能差异百分比
- `generate_summary_report()`: 生成性能统计报告

支持功能:
- 自动数据验证和清理
- Roofline性能模型计算
- 多种图表输出格式 (HTML, PNG, SVG, PDF)
- 百分比性能差异分析