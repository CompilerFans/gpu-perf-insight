#!/usr/bin/env python3
"""
GPU性能测试对比数据统计图表生成工具 (Plotly版本)
支持多个CSV文件输入，对比不同GPU的并行计算算法性能
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import argparse
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PerformanceAnalyzer:
    def __init__(self):
        self.datasets = {}  # {label: dataframe}
    
    def _get_rgb_from_name(self, color_name):
        """将颜色名称转换为RGB值"""
        color_map = {
            'red': '220, 38, 38',           # 亮红
            'green': '34, 197, 94',         # 翠绿  
            'blue': '59, 130, 246',         # 亮蓝
            'orange': '249, 115, 22',       # 亮橙
            'purple': '168, 85, 247',       # 亮紫
            'brown': '161, 98, 7',          # 棕色
            'pink': '236, 72, 153',         # 亮粉
            'gray': '107, 114, 128',        # 灰色
            'olive': '132, 204, 22',        # 橄榄绿
            'cyan': '6, 182, 212',          # 青色
            'indigo': '99, 102, 241',       # 靛蓝
            'emerald': '16, 185, 129',      # 翡翠绿
            'rose': '244, 63, 94',          # 玫瑰红
            'amber': '245, 158, 11',        # 琥珀色
            'teal': '20, 184, 166'          # 蓝绿色
        }
        return color_map.get(color_name, '0, 0, 0')  # 默认黑色
        
    def load_dataset(self, csv_path: str, label: str):
        """
        加载CSV数据集并指定标签(如GPU名称)
        
        CSV格式要求:
        - algorithm: 算法名称 (如sort, reduce, scan等)
        - data_size: 数据量规模 (以MB为单位的数值，如1, 16, 1024)
        - throughput: 吞吐量 (GB/s)
        """
        try:
            df = pd.read_csv(csv_path)
            
            # 验证必需的列
            required_columns = ['algorithm', 'data_size', 'throughput']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV文件 {csv_path} 缺少必需的列: {missing_columns}")
            
            # 确保data_size是数值类型
            df['data_size'] = pd.to_numeric(df['data_size'], errors='coerce')
            
            # 添加标签列
            df['gpu_label'] = label
            self.datasets[label] = df
            
            print(f"成功加载数据集: {label} ({len(df)} 条记录)")
            return df
            
        except Exception as e:
            print(f"加载数据集失败 {csv_path}: {e}")
            return None
    
    def get_combined_dataframe(self) -> pd.DataFrame:
        """合并所有数据集"""
        if not self.datasets:
            return pd.DataFrame()
        
        combined_df = pd.concat(self.datasets.values(), ignore_index=True)
        return combined_df
    
    
    def calculate_percentage_differences(self, extreme_threshold=300):
        """
        计算相对于第一个GPU的性能百分比差异
        
        Args:
            extreme_threshold: 极端数据阈值，超过此值的差异将被标记为极端数据
        
        Returns:
            tuple: (normal_data_df, extreme_data_list, anomaly_data_list) - 正常数据、极端数据和异常数据
        """
        df = self.get_combined_dataframe()
        if df.empty or len(df['gpu_label'].unique()) <= 1:
            return None, [], []
        
        gpu_labels = df['gpu_label'].unique()
        if len(gpu_labels) < 2:
            return None, [], []
        
        # 以第一个GPU为基准
        baseline_gpu = gpu_labels[0]
        comparison_gpus = gpu_labels[1:]
        
        percentage_data = []
        extreme_data = []
        anomaly_data = []
        
        # 获取所有算法和数据大小的组合
        algorithms = df['algorithm'].unique()
        data_sizes = df['data_size'].unique()
        
        for algorithm in algorithms:
            for data_size in data_sizes:
                # 获取基准性能
                baseline_mask = (df['gpu_label'] == baseline_gpu) & \
                               (df['algorithm'] == algorithm) & \
                               (df['data_size'] == data_size)
                baseline_data = df[baseline_mask]
                
                if baseline_data.empty:
                    continue
                
                baseline_throughput = baseline_data['throughput'].iloc[0]
                
                # 计算其他GPU的百分比差异
                for gpu in comparison_gpus:
                    comparison_mask = (df['gpu_label'] == gpu) & \
                                     (df['algorithm'] == algorithm) & \
                                     (df['data_size'] == data_size)
                    comparison_data = df[comparison_mask]
                    
                    if comparison_data.empty:
                        continue
                    
                    comparison_throughput = comparison_data['throughput'].iloc[0]
                    
                    # 检查异常数据
                    is_anomaly = False
                    anomaly_reason = ""
                    
                    # 检查基准数据异常
                    if (baseline_throughput <= 0 or 
                        np.isnan(baseline_throughput) or 
                        np.isinf(baseline_throughput)):
                        is_anomaly = True
                        anomaly_reason = f"异常基准值: {baseline_throughput}"
                    
                    # 检查对比数据异常
                    elif (comparison_throughput <= 0 or 
                          np.isnan(comparison_throughput) or 
                          np.isinf(comparison_throughput)):
                        is_anomaly = True
                        anomaly_reason = f"异常对比值: {comparison_throughput}"
                    
                    if is_anomaly:
                        # 记录异常数据
                        anomaly_data.append({
                            'algorithm': algorithm,
                            'data_size': data_size,
                            'gpu_label': gpu,
                            'baseline_throughput': baseline_throughput,
                            'comparison_throughput': comparison_throughput,
                            'reason': anomaly_reason
                        })
                        continue
                    
                    # 计算百分比比值，100%表示性能相等
                    percentage_diff = (comparison_throughput / baseline_throughput) * 100
                    
                    # 检查计算结果是否异常
                    if np.isnan(percentage_diff) or np.isinf(percentage_diff):
                        anomaly_data.append({
                            'algorithm': algorithm,
                            'data_size': data_size,
                            'gpu_label': gpu,
                            'baseline_throughput': baseline_throughput,
                            'comparison_throughput': comparison_throughput,
                            'reason': f"计算结果异常: {percentage_diff}"
                        })
                        continue
                    
                    data_point = {
                        'algorithm': algorithm,
                        'data_size': data_size,
                        'gpu_label': gpu,
                        'percentage_diff': percentage_diff,
                        'baseline_throughput': baseline_throughput,
                        'comparison_throughput': comparison_throughput
                    }
                    
                    # 检查是否为极端数据 (比值超过300%或低于33.3%)
                    if percentage_diff > extreme_threshold or percentage_diff < (100 * 100 / extreme_threshold):
                        extreme_data.append(data_point)
                    else:
                        percentage_data.append(data_point)
        
        return pd.DataFrame(percentage_data), extreme_data, anomaly_data

    def _add_bandwidth_reference_lines(self, fig, df, bandwidth_limits: list, has_multiple_gpus: bool):
        """添加带宽上限参考线到第一个图表"""
        # 为不同的带宽线设置颜色
        bandwidth_colors = ['red', 'orange', 'purple', 'brown', 'green']
        
        for i, bandwidth_limit in enumerate(bandwidth_limits):
            color = bandwidth_colors[i % len(bandwidth_colors)]
            
            # 添加水平参考线
            if has_multiple_gpus:
                fig.add_hline(
                    y=bandwidth_limit,
                    line_dash="dashdot",
                    line_color=color,
                    line_width=3,
                    row=1, col=1
                )
            else:
                fig.add_hline(
                    y=bandwidth_limit,
                    line_dash="dashdot",
                    line_color=color,
                    line_width=3
                )
            

    def create_comparison_line_chart(self, save_path: str = None, width: int = 1200, height: int = 800, reference_lines: list = None, bandwidth_limits: list = None, static_html: bool = False):
        """
        创建多GPU性能对比折线图和百分比差异图
        包含原始性能图和相对于第一个GPU的百分比差异图
        
        Args:
            reference_lines: 参考线列表，默认[100.0]。单个值表示全局参考线，多个值表示每个对比组的参考线
            bandwidth_limits: 带宽上限参考线列表，如[1600, 2000]表示1600GB/s和2000GB/s带宽上限
            static_html: 是否生成静态HTML（包含PNG图片）而非交互式Plotly，避免网页卡顿
        """
        df = self.get_combined_dataframe()
        
        if df.empty:
            print("没有数据可以绘制")
            return None
        
        # 处理参考线默认值
        if reference_lines is None:
            reference_lines = [100.0]
        
        # 获取GPU标签数量
        gpu_labels = df['gpu_label'].unique()
        has_multiple_gpus = len(gpu_labels) > 1
        
        # 如果有多个GPU，创建子图；否则只创建单个图表
        if has_multiple_gpus:
            # 根据GPU数量决定子图布局
            comparison_gpus = gpu_labels[1:]
            num_comparisons = len(comparison_gpus)
            
            # 动态布局：根据对比组数量创建子图
            # 第一行：原始性能对比
            # 第二行：百分比差异对比
            # 第三行开始：每个对比组一个直方图子图
            
            total_rows = 2 + num_comparisons  # 前两行 + 每个对比组的直方图
            comparison_cols = min(num_comparisons, 3)  # 最多3列直方图
            comparison_rows = (num_comparisons + comparison_cols - 1) // comparison_cols  # 计算需要的行数
            
            # 创建子图标题
            subplot_titles = [
                '原始性能对比',
                f'相对于 {gpu_labels[0]} 的性能百分比差异'
            ]
            
            # 为每个对比组添加直方图标题
            for i, gpu in enumerate(comparison_gpus):
                subplot_titles.append(f'{gpu} vs {gpu_labels[0]} - 差异分布直方图')
            
            # 计算布局规格
            if num_comparisons == 1:
                # 单对比：3行1列
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.12,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
                    row_heights=[0.35, 0.35, 0.3]
                )
                total_height = int(height * 2.2)
            elif num_comparisons == 2:
                # 双对比：3行2列（第三行放2个直方图）
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1,
                    specs=[[{"secondary_y": False, "colspan": 2}, None], 
                           [{"secondary_y": False, "colspan": 2}, None],
                           [{"secondary_y": False}, {"secondary_y": False}]],
                    row_heights=[0.35, 0.35, 0.3]
                )
                total_height = int(height * 2.2)
                width = int(width * 1.3)
            else:
                # 多对比：采用更灵活的布局
                cols = min(num_comparisons, 3)
                histogram_rows = (num_comparisons + cols - 1) // cols
                total_rows = 2 + histogram_rows
                
                # 构建specs
                specs = []
                # 前两行跨所有列
                specs.append([{"secondary_y": False, "colspan": cols}] + [None] * (cols - 1))
                specs.append([{"secondary_y": False, "colspan": cols}] + [None] * (cols - 1))
                
                # 直方图行
                for row in range(histogram_rows):
                    row_specs = []
                    for col in range(cols):
                        idx = row * cols + col
                        if idx < num_comparisons:
                            row_specs.append({"secondary_y": False})
                        else:
                            row_specs.append(None)
                    specs.append(row_specs)
                
                # 计算行高
                row_heights = [0.25, 0.25] + [0.5 / histogram_rows] * histogram_rows
                
                fig = make_subplots(
                    rows=total_rows, cols=cols,
                    subplot_titles=subplot_titles,
                    vertical_spacing=0.08,
                    horizontal_spacing=0.08,
                    specs=specs,
                    row_heights=row_heights
                )
                total_height = int(height * (1.5 + 0.5 * histogram_rows))
                width = int(width * 1.4)
        else:
            # 只有一个GPU时，创建单个图表
            fig = go.Figure()
            total_height = height
        
        # 获取唯一的算法和GPU标签
        algorithms = df['algorithm'].unique()
        
        # 颜色和线型配置
        colors = px.colors.qualitative.Set1
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        
        # 创建自定义hover text显示数据大小标签的函数
        def mb_to_label(mb_value):
            if mb_value < 1:
                simplified = f"{int(mb_value * 1024)}KB"
            elif mb_value < 1024:
                simplified = f"{mb_value:.1f}MB" if mb_value != int(mb_value) else f"{int(mb_value)}MB"
            else:
                simplified = f"{mb_value/1024:.1f}GB" if (mb_value/1024) != int(mb_value/1024) else f"{int(mb_value/1024)}GB"
            return f"{mb_value:.1f}MB ({simplified})"
        
        # 添加原始性能数据到第一个子图（或唯一的图表）
        for i, gpu in enumerate(gpu_labels):
            gpu_data = df[df['gpu_label'] == gpu]
            
            for j, algorithm in enumerate(algorithms):
                algo_data = gpu_data[gpu_data['algorithm'] == algorithm]
                
                if algo_data.empty:
                    continue
                
                # 按数据量排序
                algo_data = algo_data.sort_values('data_size')
                
                # 创建轨迹
                trace_name = f"{gpu} - {algorithm}"
                
                hover_labels = [mb_to_label(x) for x in algo_data['data_size']]
                
                scatter_trace = go.Scatter(
                    x=algo_data['data_size'],
                    y=algo_data['throughput'],
                    mode='lines+markers',
                    name=trace_name,
                    line=dict(
                        color=colors[j % len(colors)],
                        dash=line_styles[i % len(line_styles)],
                        width=2
                    ),
                    marker=dict(
                        size=8,
                        symbol=['circle', 'square', 'diamond', 'triangle-up'][i % 4]
                    ),
                    customdata=hover_labels,
                    hovertemplate=(
                        f"<b>{trace_name}</b><br>"
                        "Data Size: %{customdata}<br>"
                        "Throughput: %{y:.2f} GB/s<br>"
                        "<extra></extra>"
                    )
                )
                
                if has_multiple_gpus:
                    fig.add_trace(scatter_trace, row=1, col=1)
                else:
                    fig.add_trace(scatter_trace)
        
        # 添加带宽上限参考线到原始性能图（第一个图）
        if bandwidth_limits:
            self._add_bandwidth_reference_lines(fig, df, bandwidth_limits, has_multiple_gpus)
        
        # 如果有多个GPU，添加百分比差异图到第二个子图
        if has_multiple_gpus:
            percentage_df, extreme_data, anomaly_data = self.calculate_percentage_differences()
            
            if percentage_df is not None and not percentage_df.empty:
                baseline_gpu = gpu_labels[0]
                comparison_gpus = gpu_labels[1:]
                
                colors_comparison = px.colors.qualitative.Set2
                
                for i, gpu in enumerate(comparison_gpus):
                    gpu_data = percentage_df[percentage_df['gpu_label'] == gpu]
                    
                    for j, algorithm in enumerate(algorithms):
                        algo_data = gpu_data[gpu_data['algorithm'] == algorithm]
                        
                        if algo_data.empty:
                            continue
                        
                        # 按数据量排序
                        algo_data = algo_data.sort_values('data_size')
                        
                        # 创建百分比差异轨迹
                        trace_name = f"{gpu} vs {baseline_gpu} - {algorithm}"
                        
                        hover_labels = [mb_to_label(x) for x in algo_data['data_size']]
                        
                        percentage_trace = go.Scatter(
                            x=algo_data['data_size'],
                            y=algo_data['percentage_diff'],
                            mode='lines+markers',
                            name=f"{gpu} vs {baseline_gpu} - {algorithm}",
                            line=dict(
                                color=colors_comparison[j % len(colors_comparison)],
                                dash=line_styles[i % len(line_styles)],
                                width=2
                            ),
                            marker=dict(
                                size=8,
                                symbol=['circle', 'square', 'diamond', 'triangle-up'][i % 4]
                            ),
                            customdata=hover_labels,
                            hovertemplate=(
                                f"<b>{gpu} vs {baseline_gpu} - {algorithm}</b><br>"
                                "Data Size: %{customdata}<br>"
                                "Performance Ratio: %{y:.1f}%<br>"
                                "Baseline: %{customdata[0]:.2f} GB/s<br>"
                                "Comparison: %{customdata[1]:.2f} GB/s<br>"
                                "<extra></extra>"
                            )
                        )
                        
                        # 更新hover数据以包含吞吐量信息
                        hover_data = []
                        for _, row in algo_data.iterrows():
                            hover_data.append([
                                mb_to_label(row['data_size']),
                                row['baseline_throughput'],
                                row['comparison_throughput']
                            ])
                        
                        percentage_trace.customdata = hover_data
                        
                        fig.add_trace(percentage_trace, row=2, col=1)
        
        # 更新布局
        if has_multiple_gpus:
            # 多GPU时的布局设置
            fig.update_layout(
                title=dict(
                    text="GPU Performance Comparison - Original and Percentage Differences",
                    x=0.5,
                    font=dict(size=20)
                ),
                width=width,
                height=total_height,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white',
                # 设置直方图柱间距和组间距
                bargap=0.2,      # 同组内柱间距
                bargroupgap=0.1  # 不同组间距
            )
            
            # 获取实际数据规模并生成刻度标签
            actual_data_sizes = sorted(df['data_size'].unique())
            
            # 生成刻度标签函数
            def generate_tick_label(mb_value):
                if mb_value < 1:
                    return f"{int(mb_value * 1024)}KB"
                elif mb_value < 1024:
                    if mb_value == int(mb_value):
                        return f"{int(mb_value)}MB"
                    else:
                        return f"{mb_value:.1f}MB"
                else:
                    gb_value = mb_value / 1024
                    if gb_value == int(gb_value):
                        return f"{int(gb_value)}GB"
                    else:
                        return f"{gb_value:.1f}GB"
            
            # 选择合适的刻度点 - 显示所有重要的数据点以便用户查看详细数据
            # 由于我们有增强的数据采样，直接显示所有数据点作为刻度
            # 这样用户可以看到所有的1.2MB, 1.6MB, 1.8MB等详细数据点
            
            if len(actual_data_sizes) <= 30:
                # 数据点不太多，显示所有数据点
                selected_tickvals = actual_data_sizes
                selected_ticktext = [generate_tick_label(val) for val in selected_tickvals]
            else:
                # 数据点很多时，智能选择显示
                tick_indices = []
                
                # 1. 添加初期密集采样的重要点（1-10MB范围）
                for i, size in enumerate(actual_data_sizes):
                    if size <= 10:  # 1-10MB范围显示更多刻度
                        tick_indices.append(i)
                    elif size <= 100 and i % 2 == 0:  # 10-100MB范围隔点显示
                        tick_indices.append(i)
                    elif size <= 1024 and i % 4 == 0:  # 100MB-1GB范围稀疏显示
                        tick_indices.append(i)
                    elif i % 6 == 0:  # 1GB以上更稀疏显示
                        tick_indices.append(i)
                
                # 2. 确保包含重要的里程碑点
                milestone_values = [1, 1.2, 1.6, 1.8, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
                for milestone in milestone_values:
                    if milestone in actual_data_sizes:
                        milestone_idx = actual_data_sizes.index(milestone)
                        if milestone_idx not in tick_indices:
                            tick_indices.append(milestone_idx)
                
                # 3. 总是包含首尾点
                if 0 not in tick_indices:
                    tick_indices.append(0)
                if len(actual_data_sizes) - 1 not in tick_indices:
                    tick_indices.append(len(actual_data_sizes) - 1)
                
                tick_indices = sorted(list(set(tick_indices)))
                selected_tickvals = [actual_data_sizes[i] for i in tick_indices]
                selected_ticktext = [generate_tick_label(val) for val in selected_tickvals]
            
            # 更新第一个子图的轴标签
            fig.update_xaxes(
                title=dict(text="Data Size (MB)", font=dict(size=14)),
                type="log",
                tickmode='array',
                tickvals=selected_tickvals,
                ticktext=selected_ticktext,
                row=1, col=1
            )
            fig.update_yaxes(
                title=dict(text="Throughput (GB/s)", font=dict(size=14)),
                type="log",
                row=1, col=1
            )
            
            # 更新第二个子图的轴标签
            fig.update_xaxes(
                title=dict(text="Data Size (MB)", font=dict(size=14)),
                type="log",
                tickmode='array',
                tickvals=selected_tickvals,
                ticktext=selected_ticktext,
                row=2, col=1
            )
            fig.update_yaxes(
                title=dict(text="Performance Ratio (%)", font=dict(size=14)),
                row=2, col=1
            )
            
            # 为每个直方图子图添加网格
            for i, gpu in enumerate(comparison_gpus):
                if num_comparisons == 1:
                    hist_row, hist_col = 3, 1
                elif num_comparisons == 2:
                    hist_row, hist_col = 3, i + 1
                else:
                    cols = min(num_comparisons, 3)
                    hist_row = 3 + i // cols
                    hist_col = (i % cols) + 1
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=hist_row, col=hist_col)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=hist_row, col=hist_col)
            
            # 添加基准线(100%)到百分比图
            fig.add_hline(y=100, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
            
            # 添加用户指定的参考线到百分比图
            if reference_lines:
                for ref_line in reference_lines:
                    if ref_line != 100:  # 避免重复添加100%线
                        fig.add_hline(
                            y=ref_line, 
                            line_dash="solid", 
                            line_color="blue", 
                            line_width=2,
                            annotation_text=f"参考线: {ref_line}%",
                            annotation_position="right",
                            row=2, col=1
                        )
            
            # 为每个对比组创建独立的直方图子图
            if percentage_df is not None and not percentage_df.empty:
                # 创建自定义hover文本显示详细信息
                def mb_to_label(mb_value):
                    if mb_value < 1:
                        return f"{int(mb_value * 1024)}KB"
                    elif mb_value < 1024:
                        return f"{int(mb_value)}MB"
                    else:
                        return f"{int(mb_value / 1024)}GB"
                
                comparison_gpus = gpu_labels[1:]
                
                # 为每个对比组创建独立的直方图和散点图
                for i, gpu in enumerate(comparison_gpus):
                    gpu_data = percentage_df[percentage_df['gpu_label'] == gpu]
                    
                    if gpu_data.empty:
                        continue
                    
                    # 确定子图位置
                    if num_comparisons == 1:
                        hist_row, hist_col = 3, 1
                    elif num_comparisons == 2:
                        hist_row, hist_col = 3, i + 1
                    else:
                        # 多对比布局
                        cols = min(num_comparisons, 3)
                        hist_row = 3 + i // cols
                        hist_col = (i % cols) + 1
                    
                    # 1. 添加大直方图显示整体分布（背景）
                    histogram_trace = go.Histogram(
                        x=gpu_data['percentage_diff'],
                        nbinsx=20,
                        name=f'{gpu} 分布',
                        opacity=0.3,  # 降低透明度作为背景
                        marker=dict(
                            color='rgba(30, 144, 255, 0.3)',
                            line=dict(color='rgba(30, 144, 255, 0.8)', width=1)
                        ),
                        showlegend=True,
                        hovertemplate=(
                            f"<b>{gpu} vs {gpu_labels[0]} 分布</b><br>"
                            "性能比值范围: %{x}<br>"
                            "数据点数量: %{y}<br>"
                            "<extra></extra>"
                        )
                    )
                    fig.add_trace(histogram_trace, row=hist_row, col=hist_col)
                    
                    # 2. 计算直方图的柱体分布，用于散点的垂直定位
                    
                    # 计算直方图的bins和counts
                    percentage_values = gpu_data['percentage_diff'].values
                    hist_counts, bin_edges = np.histogram(percentage_values, bins=20)
                    bin_width = bin_edges[1] - bin_edges[0]
                    
                    # 按算法分组，将相同算法的数据点放在接近位置
                    # 1. 首先将数据按bin分组
                    bins_data = {}
                    for _, row in gpu_data.iterrows():
                        x_value = row['percentage_diff']
                        bin_index = min(int((x_value - bin_edges[0]) / bin_width), len(hist_counts) - 1)
                        bin_index = max(0, bin_index)
                        
                        if bin_index not in bins_data:
                            bins_data[bin_index] = []
                        bins_data[bin_index].append(row)
                    
                    # 2. 为每个bin内的数据按算法分组并分配Y坐标
                    scatter_points = []
                    for bin_index, bin_rows in bins_data.items():
                        bin_height = hist_counts[bin_index]
                        if bin_height == 0:
                            bin_height = 1  # 最小高度
                        
                        # 按算法分组
                        algo_groups = {}
                        for row in bin_rows:
                            algo = row['algorithm']
                            if algo not in algo_groups:
                                algo_groups[algo] = []
                            algo_groups[algo].append(row)
                        
                        # 为每个算法分配Y轴层级
                        algorithms = sorted(algo_groups.keys())  # 按字母序排序保持一致性
                        num_algorithms = len(algorithms)
                        
                        for algo_idx, algorithm in enumerate(algorithms):
                            algo_data = algo_groups[algorithm]
                            
                            # 计算该算法在bin中的Y轴范围
                            y_layer_start = (bin_height / num_algorithms) * algo_idx
                            y_layer_end = (bin_height / num_algorithms) * (algo_idx + 1)
                            
                            # 在算法层内按数据量排序
                            algo_data.sort(key=lambda x: x['data_size'])
                            
                            # 为算法内的每个数据点分配Y坐标
                            for data_idx, row in enumerate(algo_data):
                                x_value = row['percentage_diff']
                                
                                # 在算法层内均匀分布
                                if len(algo_data) > 1:
                                    y_ratio = data_idx / (len(algo_data) - 1)
                                else:
                                    y_ratio = 0.5  # 单个数据点居中
                                
                                # 在算法层范围内分布，留出10%边界
                                layer_height = y_layer_end - y_layer_start
                                y_value = y_layer_start + layer_height * (0.1 + y_ratio * 0.8)
                                
                                # 计算当前bin的X轴边界
                                bin_left = bin_edges[bin_index]
                                bin_right = bin_edges[bin_index + 1] if bin_index + 1 < len(bin_edges) else bin_edges[-1]
                                
                                # 确保X轴抖动不超出bin边界
                                max_x_jitter = min(
                                    x_value - bin_left,      # 距左边界的距离
                                    bin_right - x_value,     # 距右边界的距离
                                    bin_width * 0.3          # 最大抖动范围
                                ) * 0.8  # 留出20%的安全边界
                                
                                # Y轴抖动确保在算法层内
                                y_jitter = np.random.uniform(-layer_height * 0.03, layer_height * 0.03)
                                x_jitter = np.random.uniform(-max_x_jitter, max_x_jitter)
                                
                                final_x = x_value + x_jitter
                                final_y = y_value + y_jitter
                                
                                # 双重检查：确保散点在正确的边界内
                                final_x = max(bin_left + bin_width * 0.05, min(bin_right - bin_width * 0.05, final_x))
                                final_y = max(y_layer_start + layer_height * 0.02, min(y_layer_end - layer_height * 0.02, final_y))
                                
                                scatter_points.append({
                                    'x': final_x,
                                    'y': final_y,  # 不再强制最小值，确保在直方图内
                                    'algorithm': row['algorithm'],
                                    'data_size': mb_to_label(row['data_size']),
                                    'baseline_throughput': row['baseline_throughput'],
                                    'comparison_throughput': row['comparison_throughput'],
                                    'percentage_diff': row['percentage_diff']  # 保持原始值用于hover
                                })
                    
                    if scatter_points:
                        scatter_df = pd.DataFrame(scatter_points)
                        
                        # 根据算法设置不同的颜色和形状
                        algorithms = scatter_df['algorithm'].unique()
                        algorithm_colors = {}
                        algorithm_symbols = {}
                        
                        # 为每个算法分配颜色和符号 - 使用更丰富的调色板
                        available_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan', 'indigo', 'emerald', 'rose', 'amber', 'teal', 'gray']
                        available_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'hexagon', 'pentagon', 'x']
                        
                        for j, algo in enumerate(algorithms):
                            algorithm_colors[algo] = available_colors[j % len(available_colors)]
                            algorithm_symbols[algo] = available_symbols[j % len(available_symbols)]
                        
                        # 统一处理所有算法的小方块，确保同一bin内不同算法用不同颜色堆叠
                        # 1. 按bin分组所有数据点
                        bin_groups = {}
                        for _, row in scatter_df.iterrows():
                            x_value = row['x']
                            bin_index = min(int((x_value - bin_edges[0]) / bin_width), len(hist_counts) - 1)
                            bin_index = max(0, bin_index)
                            
                            if bin_index not in bin_groups:
                                bin_groups[bin_index] = []
                            bin_groups[bin_index].append(row)
                        
                        # 2. 为每个bin创建堆叠的小方块，不同算法用不同颜色
                        bar_x = []
                        bar_y = []
                        bar_base = []
                        bar_customdata = []
                        bar_colors_list = []
                        
                        for bin_index, bin_data_points in bin_groups.items():
                            bin_height = hist_counts[bin_index]
                            if bin_height == 0 or len(bin_data_points) == 0:
                                continue
                            
                            # 每个小方块的高度 = bin总高度 / 该bin内的数据点数量
                            small_block_height = bin_height / len(bin_data_points)
                            
                            # bin的X轴中心位置
                            bin_center = bin_edges[bin_index] + bin_width / 2
                            
                            # 按算法分组后堆叠（确保颜色一致性）
                            bin_data_points = sorted(bin_data_points, key=lambda x: x['algorithm'])
                            
                            # 从底部开始堆叠
                            for block_idx, data_point in enumerate(bin_data_points):
                                bar_x.append(bin_center)
                                bar_y.append(small_block_height)
                                bar_base.append(block_idx * small_block_height)  # 堆叠位置
                                
                                # 保存hover数据
                                bar_customdata.append([
                                    data_point['algorithm'],
                                    data_point['data_size'],
                                    data_point['baseline_throughput'],
                                    data_point['comparison_throughput'],
                                    data_point['percentage_diff']
                                ])
                                
                                # 颜色（基于算法）- 确保相同算法使用相同颜色
                                base_color = algorithm_colors[data_point['algorithm']]
                                alpha = 0.9
                                bar_colors_list.append(f'rgba({self._get_rgb_from_name(base_color)}, {alpha})')
                        
                        # 3. 创建统一的小方块trace
                        if bar_x:  # 只有当有数据时才添加trace
                            bar_trace = go.Bar(
                                x=bar_x,
                                y=bar_y,
                                base=bar_base,
                                width=bin_width * 0.8,  # 小方块宽度与bin宽度相同
                                name=f'数据点 ({gpu})',
                                marker=dict(
                                    color=bar_colors_list,
                                    line=dict(color='white', width=1),
                                    opacity=0.9
                                ),
                                customdata=bar_customdata,
                                hovertemplate=(
                                    f"<b>{gpu} vs {gpu_labels[0]}</b><br>"
                                    "算法: %{customdata[0]}<br>"
                                    "数据量: %{customdata[1]}<br>"
                                    "性能比值: %{customdata[4]:.1f}%<br>"
                                    f"{gpu_labels[0]}: %{{customdata[2]:.2f}} GB/s<br>"
                                    f"{gpu}: %{{customdata[3]:.2f}} GB/s<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=True
                            )
                            fig.add_trace(bar_trace, row=hist_row, col=hist_col)
                    
                    # 3. 首先添加用户指定的参考线
                    if reference_lines:
                        # 获取当前对比组的参考线
                        if len(reference_lines) == 1:
                            current_ref_line = reference_lines[0]
                        else:
                            current_ref_line = reference_lines[i] if i < len(reference_lines) else reference_lines[-1]
                        
                        # 添加用户指定的参考线
                        fig.add_vline(
                            x=current_ref_line, 
                            line_dash="solid", 
                            line_color="blue", 
                            line_width=3,
                            annotation_text=f"参考线: {current_ref_line}%",
                            annotation_position="top",
                            annotation=dict(
                                font=dict(size=12, color="blue"),
                                bgcolor="rgba(255,255,255,0.9)",
                                bordercolor="blue",
                                borderwidth=2
                            ),
                            row=hist_row, col=hist_col
                        )
                    
                    # 4. 添加统计参考线，智能避免标注重叠
                    mean_diff = gpu_data['percentage_diff'].mean()
                    median_diff = gpu_data['percentage_diff'].median()
                    
                    # 计算数据范围用于智能定位标注
                    data_min = gpu_data['percentage_diff'].min()
                    data_max = gpu_data['percentage_diff'].max()
                    data_range = data_max - data_min
                    
                    # 智能选择标注位置，避免重叠和与标题冲突  
                    # 如果均值和中位数很接近，错开显示位置
                    mean_median_gap = abs(mean_diff - median_diff)
                    
                    if mean_median_gap < data_range * 0.1:  # 差距小于数据范围的10%
                        # 差距较小时，一个在上一个在下，并稍微偏移
                        mean_position = "top right" if mean_diff > median_diff else "top left"
                        median_position = "bottom left" if mean_diff > median_diff else "bottom right"
                    else:
                        # 差距较大时，都可以在上方，但左右分开
                        mean_position = "top left"
                        median_position = "top right"
                    
                    # 添加均值线
                    fig.add_vline(
                        x=mean_diff, 
                        line_dash="dash", 
                        line_color="red", 
                        annotation_text=f"均值: {mean_diff:.1f}%", 
                        annotation_position=mean_position,
                        annotation=dict(
                            font=dict(size=10, color="red"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="red",
                            borderwidth=1
                        ),
                        row=hist_row, col=hist_col
                    )
                    
                    # 添加中位数线
                    fig.add_vline(
                        x=median_diff, 
                        line_dash="dot", 
                        line_color="green", 
                        annotation_text=f"中位数: {median_diff:.1f}%", 
                        annotation_position=median_position,
                        annotation=dict(
                            font=dict(size=10, color="green"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="green",  
                            borderwidth=1
                        ),
                        row=hist_row, col=hist_col
                    )
                    
                    # 添加基准线(100%)
                    fig.add_vline(
                        x=100, 
                        line_dash="solid", 
                        line_color="black", 
                        line_width=2,
                        annotation_text="基准线(100%)",
                        annotation_position="bottom",
                        annotation=dict(
                            font=dict(size=9, color="black"),
                            bgcolor="rgba(255,255,255,0.8)",
                            bordercolor="black",
                            borderwidth=1
                        ),
                        row=hist_row, col=hist_col
                    )
                    
                    # 更新直方图子图的轴标签
                    fig.update_xaxes(
                        title_text="性能比值百分比 (%)", 
                        row=hist_row, col=hist_col
                    )
                    fig.update_yaxes(
                        title_text="频次 / 数据点", 
                        row=hist_row, col=hist_col
                    )
                
                # 性能统计报告
                negative_data = percentage_df[percentage_df['percentage_diff'] < 100]
                if not negative_data.empty:
                    negative_count = len(negative_data)
                    total_count = len(percentage_df)
                    negative_ratio = (negative_count / total_count) * 100
                    
                    print(f"⚠️  性能低于基准: {negative_count} 个数据点 ({negative_ratio:.1f}%)")
                    
                    # 按算法分组显示负值统计
                    neg_by_algo = negative_data.groupby('algorithm').size().sort_values(ascending=False)
                    for algo, count in neg_by_algo.head(3).items():
                        print(f"  {algo}: {count} 个")
            
            # 添加网格到前两个子图
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
            
        else:
            # 单GPU时的布局设置
            fig.update_layout(
                title=dict(
                    text="GPU Performance - Single GPU Analysis",
                    x=0.5,
                    font=dict(size=20)
                ),
                xaxis=dict(
                    title=dict(text="Data Size (MB)", font=dict(size=14)),
                    type="log",
                    tickmode='array',
                    tickvals=[0.004, 0.016, 0.064, 0.256, 1, 4, 16, 64, 256, 1024, 2048, 4096],
                    ticktext=['4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB', '1GB', '2GB', '4GB']
                ),
                yaxis=dict(
                    title=dict(text="Throughput (GB/s)", font=dict(size=14)),
                    type="log"
                ),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=11)
                ),
                width=width,
                height=total_height,
                hovermode='closest',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # 添加网格
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        # 保存图表
        if save_path:
            if save_path.endswith('.html'):
                if static_html:
                    # 生成静态HTML（包含PNG图片）
                    self._write_static_html_with_statistics(fig, save_path, reference_lines, width, total_height)
                    print(f"静态HTML图表已保存: {save_path}")
                else:
                    # 生成包含统计表格的交互式HTML
                    self._write_html_with_statistics(fig, save_path, reference_lines)
                    print(f"交互式图表已保存: {save_path}")
            else:
                fig.write_image(save_path, width=width, height=total_height)
                print(f"静态图表已保存: {save_path}")
        
        # 不自动显示图表，避免输出HTML内容
        return fig

    def generate_statistics_tables(self, reference_lines: list = None) -> str:
        """生成HTML统计分析数据表"""
        if not self.datasets:
            return "<p>无数据可显示</p>"
        
        # 处理参考线默认值
        if reference_lines is None:
            reference_lines = [100.0]
        
        html_content = []
        
        # 获取合并数据
        df = self.get_combined_dataframe()
        if df.empty:
            return "<p>无数据可显示</p>"
        
        gpu_labels = list(self.datasets.keys())
        algorithms = df['algorithm'].unique()
        
        # 1. GPU性能对比总表
        html_content.append('<div class="statistics-section">')
        html_content.append('<h3>📊 GPU性能对比总览</h3>')
        html_content.append('<table class="stats-table">')
        html_content.append('<thead><tr><th>GPU</th><th>最高吞吐量 (GB/s)</th><th>数据点数量</th><th>覆盖算法</th></tr></thead>')
        html_content.append('<tbody>')
        
        for gpu in gpu_labels:
            gpu_data = df[df['gpu_label'] == gpu]
            if not gpu_data.empty:
                max_throughput = gpu_data['throughput'].max()
                max_row = gpu_data[gpu_data['throughput'] == max_throughput].iloc[0]
                data_count = len(gpu_data)
                
                # 列出该GPU测试的所有算法
                algorithms_tested = ', '.join(sorted(gpu_data['algorithm'].unique()))
                
                html_content.append(f'<tr>')
                html_content.append(f'<td><strong>{gpu}</strong></td>')
                html_content.append(f'<td>{max_throughput:.2f}<br><small>({max_row["algorithm"]}, {max_row["data_size"]:.0f}MB)</small></td>')
                html_content.append(f'<td>{data_count}</td>')
                html_content.append(f'<td><small>{algorithms_tested}</small></td>')
                html_content.append(f'</tr>')
        
        html_content.append('</tbody></table>')
        html_content.append('</div>')
        
        # 2. 算法性能范围分析表
        html_content.append('<div class="statistics-section">')
        html_content.append('<h3>🔬 算法性能范围分析</h3>')
        html_content.append('<table class="stats-table">')
        html_content.append('<thead><tr><th>算法</th>')
        
        for gpu in gpu_labels:
            html_content.append(f'<th>{gpu}<br><small>最高(GB/s)</small></th>')
        
        html_content.append('<th>总体最佳</th></tr></thead>')
        html_content.append('<tbody>')
        
        for algo in sorted(algorithms):
            html_content.append('<tr>')
            html_content.append(f'<td><strong>{algo}</strong></td>')
            
            algo_data = df[df['algorithm'] == algo]
            gpu_performances = []
            
            for gpu in gpu_labels:
                gpu_algo_data = algo_data[algo_data['gpu_label'] == gpu]
                if not gpu_algo_data.empty:
                    max_perf = gpu_algo_data['throughput'].max()
                    gpu_performances.append((gpu, max_perf))
                    html_content.append(f'<td>{max_perf:.2f}</td>')
                else:
                    gpu_performances.append((gpu, 0))
                    html_content.append(f'<td>-</td>')
            
            # 最佳GPU
            if gpu_performances:
                best_gpu = max(gpu_performances, key=lambda x: x[1])
                if best_gpu[1] > 0:
                    html_content.append(f'<td><span class="best-gpu">{best_gpu[0]}</span><br><small>{best_gpu[1]:.2f}</small></td>')
                else:
                    html_content.append(f'<td>-</td>')
            else:
                html_content.append(f'<td>-</td>')
                
            html_content.append('</tr>')
        
        html_content.append('</tbody></table>')
        html_content.append('</div>')
        
        # 3. 性能差异分析表（如果有多个GPU）
        if len(gpu_labels) > 1:
            percentage_df, extreme_data, anomaly_data = self.calculate_percentage_differences()
            if percentage_df is not None and not percentage_df.empty:
                html_content.append('<div class="statistics-section">')
                html_content.append('<h3>📈 性能差异分析</h3>')
                
                baseline_gpu = gpu_labels[0]
                comparison_gpus = gpu_labels[1:]
                
                for i, gpu in enumerate(comparison_gpus):
                    gpu_diff_data = percentage_df[percentage_df['gpu_label'] == gpu]
                    if gpu_diff_data.empty:
                        continue
                    
                    # 获取当前对比组的参考线
                    if len(reference_lines) == 1:
                        # 单个参考线，对所有对比组使用
                        current_ref_line = reference_lines[0]
                    else:
                        # 多个参考线，每个对比组使用对应的参考线
                        current_ref_line = reference_lines[i] if i < len(reference_lines) else reference_lines[-1]
                        
                    html_content.append(f'<h4>{gpu} vs {baseline_gpu} (参考线: {current_ref_line}%)</h4>')
                    html_content.append('<table class="stats-table">')
                    html_content.append('<thead><tr><th>统计指标</th><th>数值</th><th>说明</th></tr></thead>')
                    html_content.append('<tbody>')
                    
                    # 基本统计
                    mean_diff = gpu_diff_data['percentage_diff'].mean()
                    median_diff = gpu_diff_data['percentage_diff'].median()
                    std_diff = gpu_diff_data['percentage_diff'].std()
                    min_diff = gpu_diff_data['percentage_diff'].min()
                    max_diff = gpu_diff_data['percentage_diff'].max()
                    
                    # 基于参考线的性能分析
                    above_ref_count = len(gpu_diff_data[gpu_diff_data['percentage_diff'] > current_ref_line])
                    below_ref_count = len(gpu_diff_data[gpu_diff_data['percentage_diff'] < current_ref_line])
                    total_count = len(gpu_diff_data)
                    
                    # 性能优势统计 (现在100%为基准)
                    positive_count = len(gpu_diff_data[gpu_diff_data['percentage_diff'] > 100])
                    negative_count = len(gpu_diff_data[gpu_diff_data['percentage_diff'] < 100])
                    
                    stats_data = [
                        ('平均比值', f'{mean_diff:.1f}%', '大于100%表示性能提升，小于100%表示性能下降'),
                        ('中位数比值', f'{median_diff:.1f}%', '50%的测试点性能比值在此值以下'),
                        ('标准差', f'{std_diff:.1f}%', '性能比值的离散程度'),
                        ('最高比值', f'{max_diff:.1f}%', '单个测试点的最高性能比值'),
                        ('最低比值', f'{min_diff:.1f}%', '单个测试点的最低性能比值'),
                        ('', '', ''),  # 分隔线
                        ('高于参考线比例', f'{above_ref_count}/{total_count} ({100*above_ref_count/total_count:.1f}%)', f'性能比值>{current_ref_line}%的测试点比例'),
                        ('低于参考线比例', f'{below_ref_count}/{total_count} ({100*below_ref_count/total_count:.1f}%)', f'性能比值<{current_ref_line}%的测试点比例'),
                        ('', '', ''),  # 分隔线
                        ('性能优于基准比例', f'{positive_count}/{total_count} ({100*positive_count/total_count:.1f}%)', '性能比值>100%的测试点比例'),
                        ('性能低于基准比例', f'{negative_count}/{total_count} ({100*negative_count/total_count:.1f}%)', '性能比值<100%的测试点比例')
                    ]
                    
                    for stat_name, stat_value, stat_desc in stats_data:
                        color_class = ""
                        # 为空行提供分隔样式
                        if stat_name == "" and stat_value == "":
                            html_content.append(f'<tr style="height: 8px;"><td colspan="3" style="border: none; background: #f8f9fa;"></td></tr>')
                            continue
                            
                        if "优于" in stat_name and positive_count > negative_count:
                            color_class = ' class="positive"'
                        elif "低于" in stat_name and negative_count > positive_count:
                            color_class = ' class="negative"'
                        elif "高于参考线" in stat_name and above_ref_count > below_ref_count:
                            color_class = ' class="positive"'
                        elif "低于参考线" in stat_name and below_ref_count > above_ref_count:
                            color_class = ' class="negative"'
                            
                        html_content.append(f'<tr>')
                        html_content.append(f'<td><strong>{stat_name}</strong></td>')
                        html_content.append(f'<td{color_class}><strong>{stat_value}</strong></td>')
                        html_content.append(f'<td><small>{stat_desc}</small></td>')
                        html_content.append(f'</tr>')
                    
                    html_content.append('</tbody></table><br>')
                
                html_content.append('</div>')
        
        # 4. 数据规模性能分析表
        html_content.append('<div class="statistics-section">')
        html_content.append('<h3>📏 数据规模性能分析</h3>')
        html_content.append('<table class="stats-table">')
        html_content.append('<thead><tr><th>数据规模</th>')
        
        for gpu in gpu_labels:
            html_content.append(f'<th>{gpu}<br><small>最高 (GB/s)</small></th>')
        
        html_content.append('<th>最佳表现</th></tr></thead>')
        html_content.append('<tbody>')
        
        data_sizes = sorted(df['data_size'].unique())
        for size in data_sizes:
            size_data = df[df['data_size'] == size]
            
            # 格式化数据规模显示
            if size < 1:
                size_label = f"{int(size * 1024)}KB"
            elif size < 1024:
                size_label = f"{int(size)}MB"
            else:
                size_label = f"{int(size / 1024)}GB"
            
            html_content.append('<tr>')
            html_content.append(f'<td><strong>{size_label}</strong></td>')
            
            size_performances = []
            
            for gpu in gpu_labels:
                gpu_size_data = size_data[size_data['gpu_label'] == gpu]
                if not gpu_size_data.empty:
                    max_perf = gpu_size_data['throughput'].max()
                    size_performances.append((gpu, max_perf))
                    html_content.append(f'<td>{max_perf:.2f}</td>')
                else:
                    size_performances.append((gpu, 0))
                    html_content.append(f'<td>-</td>')
            
            # 最佳表现
            if size_performances:
                best_gpu_size = max(size_performances, key=lambda x: x[1])
                if best_gpu_size[1] > 0:
                    html_content.append(f'<td><span class="best-gpu">{best_gpu_size[0]}</span><br><small>{best_gpu_size[1]:.2f}</small></td>')
                else:
                    html_content.append(f'<td>-</td>')
            else:
                html_content.append(f'<td>-</td>')
                
            html_content.append('</tr>')
        
        html_content.append('</tbody></table>')
        html_content.append('</div>')
        
        # 5. 极端和异常数据报告（如果有多个GPU）
        if len(gpu_labels) > 1:
            percentage_df, extreme_data, anomaly_data = self.calculate_percentage_differences()
            
            # 极端数据展示
            if extreme_data:
                html_content.append('<div class="statistics-section">')
                html_content.append('<h3>⚠️ 极端性能比值数据 (>300% 或 <33%)</h3>')
                html_content.append(f'<p><small>共发现 <strong>{len(extreme_data)}</strong> 个极端性能比值数据点，已从常规对比分析中排除</small></p>')
                
                # 按GPU分组显示极端数据
                extreme_by_gpu = {}
                for item in extreme_data:
                    gpu = item['gpu_label']
                    if gpu not in extreme_by_gpu:
                        extreme_by_gpu[gpu] = []
                    extreme_by_gpu[gpu].append(item)
                
                for gpu, gpu_extreme_data in extreme_by_gpu.items():
                    html_content.append(f'<h4>{gpu} vs {gpu_labels[0]} - 极端案例 ({len(gpu_extreme_data)} 个)</h4>')
                    html_content.append('<table class="stats-table">')
                    html_content.append('<thead><tr><th>算法</th><th>数据规模</th><th>性能比值</th><th>基准值</th><th>对比值</th></tr></thead>')
                    html_content.append('<tbody>')
                    
                    # 按差异程度排序，只显示前10个
                    gpu_extreme_data.sort(key=lambda x: abs(x['percentage_diff'] - 100), reverse=True)
                    for item in gpu_extreme_data[:10]:
                        data_size_label = f"{int(item['data_size'] * 1024)}KB" if item['data_size'] < 1 else \
                                        f"{int(item['data_size'])}MB" if item['data_size'] < 1024 else \
                                        f"{int(item['data_size'] / 1024)}GB"
                        
                        diff_class = 'positive' if item['percentage_diff'] > 100 else 'negative'
                        html_content.append('<tr>')
                        html_content.append(f'<td>{item["algorithm"]}</td>')
                        html_content.append(f'<td>{data_size_label}</td>')
                        html_content.append(f'<td class="{diff_class}"><strong>{item["percentage_diff"]:.1f}%</strong></td>')
                        html_content.append(f'<td>{item["baseline_throughput"]:.2f} GB/s</td>')
                        html_content.append(f'<td>{item["comparison_throughput"]:.2f} GB/s</td>')
                        html_content.append('</tr>')
                    
                    if len(gpu_extreme_data) > 10:
                        html_content.append(f'<tr><td colspan="5"><small><em>... 还有 {len(gpu_extreme_data) - 10} 个极端案例</em></small></td></tr>')
                    
                    html_content.append('</tbody></table><br>')
                
                html_content.append('</div>')
            
            # 异常数据展示
            if anomaly_data:
                html_content.append('<div class="statistics-section">')
                html_content.append('<h3>🚫 异常数据 (无效值)</h3>')
                html_content.append(f'<p><small>共发现 <strong>{len(anomaly_data)}</strong> 个包含无效值的数据点，已从所有分析中排除</small></p>')
                
                # 按GPU分组显示异常数据
                anomaly_by_gpu = {}
                for item in anomaly_data:
                    gpu = item['gpu_label']
                    if gpu not in anomaly_by_gpu:
                        anomaly_by_gpu[gpu] = []
                    anomaly_by_gpu[gpu].append(item)
                
                for gpu, gpu_anomaly_data in anomaly_by_gpu.items():
                    html_content.append(f'<h4>{gpu} vs {gpu_labels[0]} - 异常案例 ({len(gpu_anomaly_data)} 个)</h4>')
                    html_content.append('<table class="stats-table">')
                    html_content.append('<thead><tr><th>算法</th><th>数据规模</th><th>异常原因</th><th>基准值</th><th>对比值</th></tr></thead>')
                    html_content.append('<tbody>')
                    
                    # 只显示前15个异常案例
                    for item in gpu_anomaly_data[:15]:
                        data_size_label = f"{int(item['data_size'] * 1024)}KB" if item['data_size'] < 1 else \
                                        f"{int(item['data_size'])}MB" if item['data_size'] < 1024 else \
                                        f"{int(item['data_size'] / 1024)}GB"
                        
                        html_content.append('<tr>')
                        html_content.append(f'<td>{item["algorithm"]}</td>')
                        html_content.append(f'<td>{data_size_label}</td>')
                        html_content.append(f'<td class="negative"><small>{item["reason"]}</small></td>')
                        html_content.append(f'<td>{item["baseline_throughput"]:.2f} GB/s</td>')
                        html_content.append(f'<td>{item["comparison_throughput"]:.2f} GB/s</td>')
                        html_content.append('</tr>')
                    
                    if len(gpu_anomaly_data) > 15:
                        html_content.append(f'<tr><td colspan="5"><small><em>... 还有 {len(gpu_anomaly_data) - 15} 个异常案例</em></small></td></tr>')
                    
                    html_content.append('</tbody></table><br>')
                
                html_content.append('</div>')
        
        return '\n'.join(html_content)

    def _get_html_styles(self) -> str:
        """获取表格样式CSS"""
        return """
        <style>
        .statistics-container {
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .statistics-section {
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .statistics-section h3 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
            font-size: 1.4em;
        }
        
        .statistics-section h4 {
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-left: 4px solid #e74c3c;
            padding-left: 12px;
        }
        
        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }
        
        .stats-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #ddd;
            font-size: 0.85em;
        }
        
        .stats-table td {
            padding: 10px 8px;
            text-align: center;
            border: 1px solid #ddd;
            background-color: #fff;
        }
        
        .stats-table tbody tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .stats-table tbody tr:hover {
            background-color: #e8f4f8;
            transition: background-color 0.2s;
        }
        
        .best-gpu {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 0.8em;
            display: inline-block;
        }
        
        .positive {
            color: #28a745;
            font-weight: bold;
        }
        
        .negative {
            color: #dc3545;
            font-weight: bold;
        }
        
        .stats-table td small {
            color: #6c757d;
            font-size: 0.8em;
            display: block;
            margin-top: 2px;
        }
        
        .stats-table th small {
            font-weight: normal;
            font-size: 0.75em;
            opacity: 0.9;
        }
        
        /* 响应式设计 */
        @media (max-width: 768px) {
            .statistics-container {
                margin: 10px;
                padding: 10px;
            }
            
            .stats-table {
                font-size: 0.8em;
            }
            
            .stats-table th,
            .stats-table td {
                padding: 6px 4px;
            }
        }
        </style>
        """

    def _write_html_with_statistics(self, fig, save_path: str, reference_lines: list = None):
        """将图表和统计表格写入HTML文件"""
        import tempfile
        import os
        
        # 先将图表保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            fig.write_html(tmp_file.name)
            
            # 读取Plotly生成的HTML内容
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                plotly_html = f.read()
        
        # 删除临时文件
        os.unlink(tmp_file.name)
        
        # 生成统计表格
        statistics_html = self.generate_statistics_tables(reference_lines)
        
        # 获取样式
        styles = self._get_html_styles()
        
        # 在Plotly HTML中插入统计表格和样式
        # 找到</head>标签，在之前插入样式
        if '</head>' in plotly_html:
            plotly_html = plotly_html.replace('</head>', f'{styles}</head>')
        
        # 找到</body>标签，在之前插入统计表格
        statistics_section = f'''
        <div class="statistics-container">
            <h2 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">
                📈 GPU性能统计分析报告
            </h2>
            {statistics_html}
        </div>
        '''
        
        if '</body>' in plotly_html:
            plotly_html = plotly_html.replace('</body>', f'{statistics_section}</body>')
        
        # 写入最终的HTML文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(plotly_html)

    def _write_static_html_with_statistics(self, fig, save_path: str, reference_lines: list = None, width: int = 1200, height: int = 800):
        """将图表作为PNG图片和统计表格写入静态HTML文件"""
        import base64
        
        try:
            # 生成PNG图片到内存
            img_bytes = fig.to_image(format="png", width=width, height=height)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            print(f"⚠️  无法生成PNG图片: {e}")
            print("🔄 使用matplotlib生成静态图片...")
            try:
                img_base64 = self._generate_matplotlib_image(width, height)
            except Exception as e2:
                print(f"⚠️  matplotlib也无法生成图片: {e2}")
                print("🔄 回退到交互式HTML模式...")
                self._write_html_with_statistics(fig, save_path, reference_lines)
                return
        
        # 生成统计表格
        statistics_html = self.generate_statistics_tables(reference_lines)
        
        # 获取样式
        styles = self._get_html_styles()
        
        # 创建完整的静态HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GPU性能对比分析</title>
    {styles}
    <style>
        .chart-container {{
            text-align: center;
            margin: 20px auto;
            max-width: {width}px;
        }}
        .chart-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="chart-container">
        <img src="data:image/png;base64,{img_base64}" alt="GPU性能对比图表" class="chart-image">
    </div>
    <div class="statistics-container">
        <h2 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">
            📈 GPU性能统计分析报告
        </h2>
        {statistics_html}
    </div>
</body>
</html>"""
        
        # 写入HTML文件
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _generate_matplotlib_image(self, width: int = 1200, height: int = 800) -> str:
        """使用matplotlib生成静态图片作为备选方案"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import base64
        import io
        
        df = self.get_combined_dataframe()
        if df.empty:
            raise ValueError("没有数据可以绘制")
        
        # 设置图片大小
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # 为每个GPU绘制数据
        gpu_labels = df['gpu_label'].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, gpu in enumerate(gpu_labels):
            gpu_data = df[df['gpu_label'] == gpu]
            color = colors[i % len(colors)]
            
            # 按算法分组绘制
            for algorithm in gpu_data['algorithm'].unique():
                alg_data = gpu_data[gpu_data['algorithm'] == algorithm]
                if not alg_data.empty:
                    ax.plot(alg_data['data_size'], alg_data['throughput'], 
                           'o-', color=color, alpha=0.7, markersize=3,
                           label=f'{gpu}-{algorithm}' if i == 0 else "")
        
        # 设置标签和标题
        ax.set_xlabel('数据量 (MB)', fontsize=12)
        ax.set_ylabel('吞吐量 (GB/s)', fontsize=12)
        ax.set_title('GPU性能对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # 添加图例
        if len(gpu_labels) > 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存到内存
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        plt.close(fig)
        return img_base64

    def generate_summary_report(self, reference_lines: list = None) -> str:
        """生成性能分析总结报告"""
        df = self.get_combined_dataframe()
        
        if df.empty:
            return "没有数据可以分析"
        
        # 处理参考线默认值
        if reference_lines is None:
            reference_lines = [100.0]
        
        report = []
        report.append("=" * 50)
        report.append("Performance Analysis Report")
        report.append("=" * 50)
        report.append(f"Total samples: {len(df)}")
        report.append(f"Algorithms tested: {', '.join(df['algorithm'].unique())}")
        report.append(f"GPU labels: {', '.join(df['gpu_label'].unique())}")
        report.append(f"Data sizes: {', '.join(map(str, df['data_size'].unique()))}")
        report.append("")
        
        # 最佳性能统计
        best_perf = df.loc[df['throughput'].idxmax()]
        report.append("Best Performance:")
        report.append(f"  Algorithm: {best_perf['algorithm']}")
        report.append(f"  GPU: {best_perf['gpu_label']}")
        report.append(f"  Data Size: {best_perf['data_size']}")
        report.append(f"  Throughput: {best_perf['throughput']:.2f} GB/s")
        report.append("")
        
        # 各GPU最佳性能
        report.append("Best Performance by GPU:")
        for gpu in df['gpu_label'].unique():
            gpu_data = df[df['gpu_label'] == gpu]
            max_perf = gpu_data['throughput'].max()
            max_row = gpu_data[gpu_data['throughput'] == max_perf].iloc[0]
            report.append(f"  {gpu}: {max_perf:.2f} GB/s ({max_row['algorithm']}, {max_row['data_size']:.0f}MB)")
        report.append("")
        
        # 性能百分比差异分析（如果有多个GPU）
        gpu_labels = df['gpu_label'].unique()
        if len(gpu_labels) > 1:
            report.append("Performance Percentage Ratios:")
            report.append(f"  Baseline GPU: {gpu_labels[0]}")
            report.append("")
            
            percentage_df, extreme_data, anomaly_data = self.calculate_percentage_differences()
            if percentage_df is not None:
                comparison_gpus = gpu_labels[1:]
                for i, gpu in enumerate(comparison_gpus):
                    gpu_data = percentage_df[percentage_df['gpu_label'] == gpu]
                    if not gpu_data.empty:
                        # 获取当前对比组的参考线
                        if len(reference_lines) == 1:
                            current_ref_line = reference_lines[0]
                        else:
                            current_ref_line = reference_lines[i] if i < len(reference_lines) else reference_lines[-1]
                        
                        max_diff = gpu_data['percentage_diff'].max()
                        min_diff = gpu_data['percentage_diff'].min()
                        median_diff = gpu_data['percentage_diff'].median()
                        
                        # 基于参考线的统计
                        above_ref_count = len(gpu_data[gpu_data['percentage_diff'] > current_ref_line])
                        below_ref_count = len(gpu_data[gpu_data['percentage_diff'] < current_ref_line])
                        total_count = len(gpu_data)
                        
                        report.append(f"  {gpu} vs {gpu_labels[0]} (Reference: {current_ref_line}%):")
                        report.append(f"    Max Ratio: {max_diff:.1f}%")
                        report.append(f"    Min Ratio: {min_diff:.1f}%")
                        report.append(f"    Median Ratio: {median_diff:.1f}%")
                        report.append(f"    Above Reference: {above_ref_count}/{total_count} ({100*above_ref_count/total_count:.1f}%)")
                        report.append(f"    Below Reference: {below_ref_count}/{total_count} ({100*below_ref_count/total_count:.1f}%)")
                        report.append("")
            
            # 添加极端数据报告
            if extreme_data:
                report.append("Extreme Performance Ratios (>300% or <33%):")
                report.append(f"  Total extreme data points: {len(extreme_data)}")
                report.append("")
                
                # 按GPU分组显示极端数据
                extreme_by_gpu = {}
                for item in extreme_data:
                    gpu = item['gpu_label']
                    if gpu not in extreme_by_gpu:
                        extreme_by_gpu[gpu] = []
                    extreme_by_gpu[gpu].append(item)
                
                for gpu, gpu_extreme_data in extreme_by_gpu.items():
                    report.append(f"  {gpu} vs {gpu_labels[0]} - Extreme Cases:")
                    # 按差异程度排序
                    gpu_extreme_data.sort(key=lambda x: abs(x['percentage_diff'] - 100), reverse=True)
                    
                    for item in gpu_extreme_data[:10]:  # 只显示前10个最极端的
                        data_size_label = f"{int(item['data_size'] * 1024)}KB" if item['data_size'] < 1 else \
                                        f"{int(item['data_size'])}MB" if item['data_size'] < 1024 else \
                                        f"{int(item['data_size'] / 1024)}GB"
                        report.append(f"    {item['algorithm']} ({data_size_label}): {item['percentage_diff']:.1f}% "
                                    f"({item['baseline_throughput']:.2f} → {item['comparison_throughput']:.2f} GB/s)")
                    
                    if len(gpu_extreme_data) > 10:
                        report.append(f"    ... and {len(gpu_extreme_data) - 10} more extreme cases")
                    report.append("")
            
            # 添加异常数据报告
            if anomaly_data:
                report.append("Anomalous Data (invalid values):")
                report.append(f"  Total anomalous data points: {len(anomaly_data)}")
                report.append("")
                
                # 按GPU分组显示异常数据
                anomaly_by_gpu = {}
                for item in anomaly_data:
                    gpu = item['gpu_label']
                    if gpu not in anomaly_by_gpu:
                        anomaly_by_gpu[gpu] = []
                    anomaly_by_gpu[gpu].append(item)
                
                for gpu, gpu_anomaly_data in anomaly_by_gpu.items():
                    report.append(f"  {gpu} vs {gpu_labels[0]} - Anomalous Cases:")
                    
                    for item in gpu_anomaly_data[:15]:  # 只显示前15个异常案例
                        data_size_label = f"{int(item['data_size'] * 1024)}KB" if item['data_size'] < 1 else \
                                        f"{int(item['data_size'])}MB" if item['data_size'] < 1024 else \
                                        f"{int(item['data_size'] / 1024)}GB"
                        report.append(f"    {item['algorithm']} ({data_size_label}): {item['reason']} "
                                    f"({item['baseline_throughput']:.2f} vs {item['comparison_throughput']:.2f} GB/s)")
                    
                    if len(gpu_anomaly_data) > 15:
                        report.append(f"    ... and {len(gpu_anomaly_data) - 15} more anomalous cases")
                    report.append("")
        
        # 各算法最佳性能
        report.append("Best Performance by Algorithm:")
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            max_perf = algo_data['throughput'].max()
            max_row = algo_data[algo_data['throughput'] == max_perf].iloc[0]
            report.append(f"  {algo}: {max_perf:.2f} GB/s ({max_row['gpu_label']}, {max_row['data_size']:.0f}MB)")
        
        return "\n".join(report)


def create_comprehensive_datasets(scale_factor=1.0):
    """创建综合性能数据集，包含更多算法类别和数据规模
    
    Args:
        scale_factor: 数据量缩放系数，1.0=完整数据集，0.1=测试用小数据集
    """
    
    # CUB和Thrust相关的GPU并行计算算法
    algorithms = {
        # CUB核心算法
        'cub_reduce': 'CUB Reduce',
        'cub_scan': 'CUB Scan/Prefix Sum',
        'cub_sort': 'CUB Radix Sort',
        'cub_histogram': 'CUB Histogram',
        'cub_select': 'CUB Stream Compaction',
        
        # Thrust算法
        'thrust_reduce': 'Thrust Reduce',
        'thrust_scan': 'Thrust Scan',
        'thrust_sort': 'Thrust Sort',
        'thrust_transform': 'Thrust Transform',
        'thrust_copy_if': 'Thrust Copy If',
        
        # 组合算法
        'reduce_by_key': 'Reduce By Key',
        'unique': 'Unique/Adjacent Difference',
        'partition': 'Partition',
        'merge': 'Merge',
        'set_operations': 'Set Operations'
    }
    
    # 扩展的数据规模范围 (几KB到几GB) - 以MB为单位的数值，以2倍递增
    # 增强的数据采样：1k起始，4GB结束，初期密集采样
    base_sizes = [1.0, 1.2, 1.6, 1.8]  # 初期基础点：1MB, 1.2MB, 1.6MB, 1.8MB
    
    # 生成完整的数据规模序列
    data_sizes_mb = []
    
    # 添加初期基础点及其2倍扩展序列
    for base in base_sizes:
        current = base
        while current <= 4096:  # 扩展到4GB
            data_sizes_mb.append(current)
            current *= 2
    
    # 在1-4GB范围内增加更多采样密度
    additional_1gb_4gb = [
        1024, 1280, 1536, 1792,  # 1GB-1.75GB 范围
        2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840,  # 2GB-3.75GB 范围  
        4096  # 4GB
    ]
    
    # 合并并去重排序
    data_sizes_mb.extend(additional_1gb_4gb)
    data_sizes_mb = sorted(list(set(data_sizes_mb)))
    
    # 生成对应的显示标签
    data_size_labels = []
    for size_mb in data_sizes_mb:
        if size_mb < 1:
            data_size_labels.append(f'{int(size_mb * 1024)}KB')
        elif size_mb < 1024:
            if size_mb == int(size_mb):
                data_size_labels.append(f'{int(size_mb)}MB')
            else:
                data_size_labels.append(f'{size_mb:.1f}MB')
        else:
            gb_size = size_mb / 1024
            if gb_size == int(gb_size):
                data_size_labels.append(f'{int(gb_size)}GB')
            else:
                data_size_labels.append(f'{gb_size:.1f}GB')
    
    # 根据scale_factor调整数据规模
    if scale_factor < 1.0:
        # 测试模式：减少算法和数据规模
        algorithm_count = max(3, int(len(algorithms) * scale_factor))
        data_size_count = max(5, int(len(data_sizes_mb) * scale_factor))
        
        selected_algorithms = dict(list(algorithms.items())[:algorithm_count])
        selected_data_sizes = [data_sizes_mb[i] for i in range(0, len(data_sizes_mb), max(1, len(data_sizes_mb) // data_size_count))][:data_size_count]
        selected_labels = [data_size_labels[i] for i in range(0, len(data_size_labels), max(1, len(data_size_labels) // data_size_count))][:data_size_count]
        
        print(f"测试模式 (scale={scale_factor}): {len(selected_algorithms)} 种算法, {len(selected_data_sizes)} 个数据规模")
    else:
        # 完整模式
        selected_algorithms = algorithms
        selected_data_sizes = data_sizes_mb
        selected_labels = data_size_labels
    
    # GPU型号配置 (简化版，只保留必要的name)
    gpu_configs = {
        'RTX_4090': {'name': 'RTX_4090'},
        'A100': {'name': 'A100'},
        'H100': {'name': 'H100'},
        'X500': {'name': 'X500'},
        'X500_optimized': {'name': 'X500_optimized'},
        'X600': {'name': 'X600'}
    }
    
    np.random.seed(42)
    
    def calculate_roofline_performance(algorithm, data_size_mb, gpu_config):
        """简化的roofline模型：y=a*x 线性关系"""
        
        # 算法特性系数 - 调整以使A100在大数据量时达到约1700GB/s平台
        algorithm_coefficients = {
            # CUB核心算法
            'cub_reduce': 16.5, 'cub_scan': 15.0, 'cub_sort': 13.6, 'cub_histogram': 11.8, 'cub_select': 12.8,
            # Thrust算法  
            'thrust_reduce': 16.0, 'thrust_scan': 15.6, 'thrust_sort': 13.0, 'thrust_transform': 18.2, 'thrust_copy_if': 14.4,
            # 组合算法
            'reduce_by_key': 15.2, 'unique': 14.0, 'partition': 13.6, 'merge': 17.0, 'set_operations': 14.8
        }
        
        # GPU相对性能系数 (基于A100=1700GB/s平台调整)
        gpu_multipliers = {
            'RTX_4090': 0.61,       # ~1040GB/s (A100的61%)
            'A100': 1.0,            # ~1700GB/s 基准
            'H100': 1.35,           # ~2300GB/s (A100的135%)  
            'X500': 0.80,           # ~1360GB/s (A100的80%)
            'X500_optimized': 0.90, # ~1530GB/s (软件优化提升到90%)
            'X600': 0.50            # ~850GB/s (A100的50%)
        }
        
        # 获取算法系数和GPU系数
        a = algorithm_coefficients.get(algorithm, 15.0)  # 默认系数15.0
        gpu_factor = gpu_multipliers.get(gpu_config.get('name', 'A100'), 1.0)
        
        data_mb = data_size_mb
        
        if data_mb <= 64:  # <= 64MB: 纯线性关系 y = a*x
            effective_bandwidth = a * gpu_factor * data_mb
        else:  # > 64MB: 平台阶段
            # 平台值基于64MB时的线性值
            plateau_base = a * gpu_factor * 64
            effective_bandwidth = plateau_base * (0.95 + 0.1 * np.random.random())  # 95-105%随机变化
        
        # 添加小量随机噪声
        noise_factor = 1 + np.random.normal(0, 0.3)
        effective_bandwidth *= noise_factor
        
        return effective_bandwidth
    
    # 为每个GPU生成数据集
    datasets = {}
    
    for gpu_name, gpu_config in gpu_configs.items():
        print(f"生成 {gpu_name} 数据集...")
        gpu_data = []
        
        for algorithm in selected_algorithms.keys():
            for i, data_size_mb in enumerate(selected_data_sizes):
                throughput = calculate_roofline_performance(algorithm, data_size_mb, gpu_config)
                
                gpu_data.append({
                    'algorithm': algorithm,
                    'data_size_mb': data_size_mb,  # 数值形式
                    'data_size_label': selected_labels[i] if i < len(selected_labels) else f"{data_size_mb}MB",  # 显示标签
                    'throughput': max(throughput, 0.01)  # 确保非负值
                })
        
        # 保存数据集 (保存数值格式的data_size_mb用于绘图)
        df = pd.DataFrame(gpu_data)
        # 为了兼容性，CSV中只保存algorithm, data_size_mb, throughput
        csv_df = df[['algorithm', 'data_size_mb', 'throughput']].rename(columns={'data_size_mb': 'data_size'})
        filename = f"{gpu_name.lower()}_performance.csv"
        csv_df.to_csv(filename, index=False)
        datasets[gpu_name] = df
        
        print(f"  - {filename}: {len(gpu_data)} 条记录")
    
    print(f"\n共生成 {len(datasets)} 个GPU数据集，覆盖 {len(selected_algorithms)} 种算法类型")
    print(f"数据规模范围: {selected_labels[0]} - {selected_labels[-1]}")
    
    return datasets

def create_sample_datasets(scale_factor=1.0):
    """创建简单示例数据集（向后兼容）"""
    return create_comprehensive_datasets(scale_factor=scale_factor)


def main():
    parser = argparse.ArgumentParser(description='GPU Performance Comparison Tool')
    parser.add_argument('--csv-files', nargs='+',
                        help='CSV file paths')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for each CSV file (e.g., GPU names)')
    parser.add_argument('--output', '-o', default='performance_comparison.html',
                        help='Output file path (default: performance_comparison.html)')
    parser.add_argument('--width', type=int, default=1200,
                        help='Chart width (default: 1200)')
    parser.add_argument('--height', type=int, default=800,
                        help='Chart height (default: 800)')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample datasets and exit')
    parser.add_argument('--sample-scale', type=float, default=1.0,
                        help='Data generation scale factor (default: 1.0, use 0.2 for testing)')
    parser.add_argument('--reference-lines', nargs='+', type=float,
                        help='Reference lines for performance comparison (e.g., --reference-lines 80 90)')
    parser.add_argument('--bandwidth-limits', nargs='+', type=float,
                        help='Bandwidth upper limit reference lines in GB/s (e.g., --bandwidth-limits 1600 2000)')
    parser.add_argument('--static-html', action='store_true',
                        help='Generate static HTML with PNG image instead of interactive Plotly (faster loading)')
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_sample:
        create_sample_datasets(scale_factor=args.sample_scale)
        return
    
    # 验证必需参数（仅在没有创建示例数据时需要）
    if not args.create_sample and (not args.csv_files or not args.labels):
        print("错误: 需要指定 --csv-files 和 --labels 参数")
        print("使用 --help 查看完整帮助信息")
        print("或使用 --create-sample 创建示例数据")
        sys.exit(1)
    
    # 验证参数
    if len(args.csv_files) != len(args.labels):
        print("错误: CSV文件数量与标签数量不匹配")
        sys.exit(1)
    
    # 验证文件存在
    for csv_file in args.csv_files:
        if not Path(csv_file).exists():
            print(f"错误: 文件不存在 {csv_file}")
            sys.exit(1)
    
    # 创建分析器
    analyzer = PerformanceAnalyzer()
    
    # 加载数据集
    for csv_file, label in zip(args.csv_files, args.labels):
        analyzer.load_dataset(csv_file, label)
    
    if not analyzer.datasets:
        print("错误: 没有成功加载任何数据集")
        sys.exit(1)
    
    # 生成对比图表
    print("\n正在生成性能对比图表...")
    analyzer.create_comparison_line_chart(
        save_path=args.output,
        width=args.width,
        height=args.height,
        reference_lines=args.reference_lines,
        bandwidth_limits=args.bandwidth_limits,
        static_html=args.static_html
    )
    
    # 生成分析报告
    print("\n" + analyzer.generate_summary_report(args.reference_lines))


if __name__ == "__main__":
    main()
