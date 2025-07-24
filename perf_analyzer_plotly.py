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
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PerformanceAnalyzer:
    def __init__(self):
        self.datasets = {}  # {label: dataframe}
    
    def _get_rgb_from_name(self, color_name):
        """将颜色名称转换为RGB值"""
        color_map = {
            'red': '255, 0, 0',
            'green': '0, 128, 0', 
            'blue': '0, 0, 255',
            'orange': '255, 165, 0',
            'purple': '128, 0, 128',
            'brown': '139, 69, 19',
            'pink': '255, 192, 203',
            'gray': '128, 128, 128',
            'olive': '128, 128, 0',
            'cyan': '0, 255, 255'
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
    
    def create_comparison_line_chart(self, save_path: str = None, width: int = 1200, height: int = 800):
        """
        创建多GPU性能对比折线图
        每个算法一条线，不同GPU用不同颜色/样式区分
        """
        df = self.get_combined_dataframe()
        
        if df.empty:
            print("没有数据可以绘制")
            return None
        
        # 创建图表
        fig = go.Figure()
        
        # 获取唯一的算法和GPU标签
        algorithms = df['algorithm'].unique()
        gpu_labels = df['gpu_label'].unique()
        
        # 颜色和线型配置
        colors = px.colors.qualitative.Set1
        line_styles = ['solid', 'dash', 'dot', 'dashdot']
        
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
                
                # 创建自定义hover text显示数据大小标签
                def mb_to_label(mb_value):
                    if mb_value < 1:
                        return f"{int(mb_value * 1024)}KB"
                    elif mb_value < 1024:
                        return f"{int(mb_value)}MB"
                    else:
                        return f"{int(mb_value / 1024)}GB"
                
                hover_labels = [mb_to_label(x) for x in algo_data['data_size']]
                
                fig.add_trace(go.Scatter(
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
                ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text="GPU Performance Comparison - Parallel Computing Algorithms",
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title=dict(text="Data Size (MB)", font=dict(size=14)),
                tickfont=dict(size=12),
                type="log",  # 使用对数坐标
                tickmode='array',
                tickvals=[0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                ticktext=['4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB', '2GB', '4GB']
            ),
            yaxis=dict(
                title=dict(text="Throughput (GB/s)", font=dict(size=14)),
                tickfont=dict(size=12),
                type="log"  # 使用对数坐标
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
            height=height,
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
                fig.write_html(save_path)
                print(f"交互式图表已保存: {save_path}")
            else:
                fig.write_image(save_path, width=width, height=height)
                print(f"静态图表已保存: {save_path}")
        
        # 不自动显示图表，避免输出HTML内容
        return fig
    
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
                    import numpy as np
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
                    
                    # 计算百分比差异
                    percentage_diff = ((comparison_throughput - baseline_throughput) / baseline_throughput) * 100
                    
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
                    
                    # 检查是否为极端数据
                    if abs(percentage_diff) > extreme_threshold:
                        extreme_data.append(data_point)
                    else:
                        percentage_data.append(data_point)
        
        return pd.DataFrame(percentage_data), extreme_data, anomaly_data

    def create_comparison_line_chart(self, save_path: str = None, width: int = 1200, height: int = 800):
        """
        创建多GPU性能对比折线图和百分比差异图
        包含原始性能图和相对于第一个GPU的百分比差异图
        """
        df = self.get_combined_dataframe()
        
        if df.empty:
            print("没有数据可以绘制")
            return None
        
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
                return f"{int(mb_value * 1024)}KB"
            elif mb_value < 1024:
                return f"{int(mb_value)}MB"
            else:
                return f"{int(mb_value / 1024)}GB"
        
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
                                "Performance Difference: %{y:+.1f}%<br>"
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
            
            # 更新第一个子图的轴标签
            fig.update_xaxes(
                title=dict(text="Data Size (MB)", font=dict(size=14)),
                type="log",
                tickmode='array',
                tickvals=[0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                ticktext=['4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB', '2GB', '4GB'],
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
                tickvals=[0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                ticktext=['4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB', '2GB', '4GB'],
                row=2, col=1
            )
            fig.update_yaxes(
                title=dict(text="Performance Difference (%)", font=dict(size=14)),
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
            
            # 添加零线到百分比图
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
            
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
                    
                    # 1. 添加直方图显示整体分布
                    histogram_trace = go.Histogram(
                        x=gpu_data['percentage_diff'],
                        nbinsx=20,
                        name=f'{gpu} 分布',
                        opacity=0.7,
                        marker=dict(
                            color='rgba(30, 144, 255, 0.7)',
                            line=dict(color='rgba(30, 144, 255, 1)', width=1)
                        ),
                        showlegend=True,
                        hovertemplate=(
                            f"<b>{gpu} vs {gpu_labels[0]} 分布</b><br>"
                            "性能差异范围: %{x}<br>"
                            "数据点数量: %{y}<br>"
                            "<extra></extra>"
                        )
                    )
                    fig.add_trace(histogram_trace, row=hist_row, col=hist_col)
                    
                    # 2. 计算直方图的柱体分布，用于散点的垂直定位
                    import numpy as np
                    
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
                                    'y': max(0.1, final_y),
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
                        
                        # 为每个算法分配颜色和符号
                        available_colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                        available_symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'triangle-down', 'star', 'hexagon', 'pentagon', 'x']
                        
                        for j, algo in enumerate(algorithms):
                            algorithm_colors[algo] = available_colors[j % len(available_colors)]
                            algorithm_symbols[algo] = available_symbols[j % len(available_symbols)]
                        
                        # 为每个算法创建一个散点trace
                        for algo in algorithms:
                            algo_data = scatter_df[scatter_df['algorithm'] == algo]
                            
                            # 根据性能差异调整透明度
                            colors_with_alpha = []
                            for diff in algo_data['x']:
                                base_color = algorithm_colors[algo]
                                alpha = 0.9 if abs(diff) > 10 else 0.6  # 差异大的点更显著
                                colors_with_alpha.append(f'rgba({self._get_rgb_from_name(base_color)}, {alpha})')
                            
                            scatter_trace = go.Scatter(
                                x=algo_data['x'],
                                y=algo_data['y'],
                                mode='markers',
                                name=f'{algo} ({gpu})',
                                marker=dict(
                                    size=10,
                                    color=colors_with_alpha,
                                    symbol=algorithm_symbols[algo],
                                    line=dict(color='white', width=1)
                                ),
                                customdata=list(zip(
                                    algo_data['algorithm'],
                                    algo_data['data_size'],
                                    algo_data['baseline_throughput'],
                                    algo_data['comparison_throughput'],
                                    algo_data['percentage_diff']  # 使用原始的性能差异值
                                )),
                                hovertemplate=(
                                    f"<b>{gpu} vs {gpu_labels[0]}</b><br>"
                                    "算法: %{customdata[0]}<br>"
                                    "数据量: %{customdata[1]}<br>"
                                    "性能差异: %{customdata[4]:+.1f}%<br>"
                                    f"{gpu_labels[0]}: %{{customdata[2]:.2f}} GB/s<br>"
                                    f"{gpu}: %{{customdata[3]:.2f}} GB/s<br>"
                                    "<extra></extra>"
                                ),
                                showlegend=True
                            )
                            fig.add_trace(scatter_trace, row=hist_row, col=hist_col)
                    
                    # 3. 添加统计参考线，智能避免标注重叠
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
                    
                    # 添加零线（基准线）
                    fig.add_vline(
                        x=0, 
                        line_dash="solid", 
                        line_color="black", 
                        line_width=2,
                        annotation_text="基准线",
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
                        title_text="性能差异百分比 (%)", 
                        row=hist_row, col=hist_col
                    )
                    fig.update_yaxes(
                        title_text="频次 / 数据点", 
                        row=hist_row, col=hist_col
                    )
                
                # 性能统计报告
                negative_data = percentage_df[percentage_df['percentage_diff'] < 0]
                if not negative_data.empty:
                    negative_count = len(negative_data)
                    total_count = len(percentage_df)
                    negative_ratio = (negative_count / total_count) * 100
                    
                    print(f"⚠️  性能落后: {negative_count} 个数据点 ({negative_ratio:.1f}%)")
                    
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
                # 生成包含统计表格的完整HTML
                self._write_html_with_statistics(fig, save_path)
                print(f"交互式图表已保存: {save_path}")
            else:
                fig.write_image(save_path, width=width, height=total_height)
                print(f"静态图表已保存: {save_path}")
        
        # 不自动显示图表，避免输出HTML内容
        return fig

    def generate_statistics_tables(self) -> str:
        """生成HTML统计分析数据表"""
        if not self.datasets:
            return "<p>无数据可显示</p>"
        
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
        html_content.append('<thead><tr><th>GPU</th><th>平均吞吐量 (GB/s)</th><th>最高吞吐量 (GB/s)</th><th>数据点数量</th><th>主要优势算法</th></tr></thead>')
        html_content.append('<tbody>')
        
        for gpu in gpu_labels:
            gpu_data = df[df['gpu_label'] == gpu]
            if not gpu_data.empty:
                avg_throughput = gpu_data['throughput'].mean()
                max_throughput = gpu_data['throughput'].max()
                max_row = gpu_data[gpu_data['throughput'] == max_throughput].iloc[0]
                data_count = len(gpu_data)
                
                # 找出该GPU表现最好的算法
                algo_avg = gpu_data.groupby('algorithm')['throughput'].mean().sort_values(ascending=False)
                top_algorithms = ', '.join(algo_avg.head(3).index.tolist())
                
                html_content.append(f'<tr>')
                html_content.append(f'<td><strong>{gpu}</strong></td>')
                html_content.append(f'<td>{avg_throughput:.2f}</td>')
                html_content.append(f'<td>{max_throughput:.2f}<br><small>({max_row["algorithm"]}, {max_row["data_size"]:.0f}MB)</small></td>')
                html_content.append(f'<td>{data_count}</td>')
                html_content.append(f'<td><small>{top_algorithms}</small></td>')
                html_content.append(f'</tr>')
        
        html_content.append('</tbody></table>')
        html_content.append('</div>')
        
        # 2. 算法性能分析表
        html_content.append('<div class="statistics-section">')
        html_content.append('<h3>🔬 算法性能分析</h3>')
        html_content.append('<table class="stats-table">')
        html_content.append('<thead><tr><th>算法</th>')
        
        for gpu in gpu_labels:
            html_content.append(f'<th>{gpu}<br><small>(GB/s)</small></th>')
        
        html_content.append('<th>算法平均</th><th>最佳GPU</th></tr></thead>')
        html_content.append('<tbody>')
        
        for algo in sorted(algorithms):
            html_content.append('<tr>')
            html_content.append(f'<td><strong>{algo}</strong></td>')
            
            algo_data = df[df['algorithm'] == algo]
            gpu_performances = []
            
            for gpu in gpu_labels:
                gpu_algo_data = algo_data[algo_data['gpu_label'] == gpu]
                if not gpu_algo_data.empty:
                    avg_perf = gpu_algo_data['throughput'].mean()
                    gpu_performances.append((gpu, avg_perf))
                    html_content.append(f'<td>{avg_perf:.2f}</td>')
                else:
                    gpu_performances.append((gpu, 0))
                    html_content.append(f'<td>-</td>')
            
            # 算法整体平均
            algo_avg = algo_data['throughput'].mean()
            html_content.append(f'<td><strong>{algo_avg:.2f}</strong></td>')
            
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
                
                for gpu in comparison_gpus:
                    gpu_diff_data = percentage_df[percentage_df['gpu_label'] == gpu]
                    if gpu_diff_data.empty:
                        continue
                        
                    html_content.append(f'<h4>{gpu} vs {baseline_gpu}</h4>')
                    html_content.append('<table class="stats-table">')
                    html_content.append('<thead><tr><th>统计指标</th><th>数值</th><th>说明</th></tr></thead>')
                    html_content.append('<tbody>')
                    
                    # 基本统计
                    mean_diff = gpu_diff_data['percentage_diff'].mean()
                    median_diff = gpu_diff_data['percentage_diff'].median()
                    std_diff = gpu_diff_data['percentage_diff'].std()
                    min_diff = gpu_diff_data['percentage_diff'].min()
                    max_diff = gpu_diff_data['percentage_diff'].max()
                    
                    # 性能优势统计
                    positive_count = len(gpu_diff_data[gpu_diff_data['percentage_diff'] > 0])
                    negative_count = len(gpu_diff_data[gpu_diff_data['percentage_diff'] < 0])
                    total_count = len(gpu_diff_data)
                    
                    stats_data = [
                        ('平均差异', f'{mean_diff:+.1f}%', '正值表示性能提升，负值表示性能下降'),
                        ('中位数差异', f'{median_diff:+.1f}%', '50%的测试点性能差异在此值以下'),
                        ('标准差', f'{std_diff:.1f}%', '性能差异的离散程度'),
                        ('最大提升', f'{max_diff:+.1f}%', '单个测试点的最大性能提升'),
                        ('最大下降', f'{min_diff:+.1f}%', '单个测试点的最大性能下降'),
                        ('性能提升比例', f'{positive_count}/{total_count} ({100*positive_count/total_count:.1f}%)', '表现更好的测试点比例'),
                        ('性能下降比例', f'{negative_count}/{total_count} ({100*negative_count/total_count:.1f}%)', '表现较差的测试点比例')
                    ]
                    
                    for stat_name, stat_value, stat_desc in stats_data:
                        color_class = ""
                        if "提升" in stat_name and "+" in stat_value:
                            color_class = ' class="positive"'
                        elif "下降" in stat_name and "-" in stat_value:
                            color_class = ' class="negative"'
                            
                        html_content.append(f'<tr>')
                        html_content.append(f'<td><strong>{stat_name}</strong></td>')
                        html_content.append(f'<td{color_class}><strong>{stat_value}</strong></td>')
                        html_content.append(f'<td><small>{stat_desc}</small></td>')
                        html_content.append(f'</tr>')
                    
                    html_content.append('</tbody></table><br>')
                
                html_content.append('</div>')
        
        # 4. 数据规模分析表
        html_content.append('<div class="statistics-section">')
        html_content.append('<h3>📏 数据规模性能分析</h3>')
        html_content.append('<table class="stats-table">')
        html_content.append('<thead><tr><th>数据规模</th>')
        
        for gpu in gpu_labels:
            html_content.append(f'<th>{gpu}<br><small>平均 (GB/s)</small></th>')
        
        html_content.append('<th>规模平均</th><th>最佳表现</th></tr></thead>')
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
                    avg_perf = gpu_size_data['throughput'].mean()
                    size_performances.append((gpu, avg_perf))
                    html_content.append(f'<td>{avg_perf:.2f}</td>')
                else:
                    size_performances.append((gpu, 0))
                    html_content.append(f'<td>-</td>')
            
            # 规模平均
            size_avg = size_data['throughput'].mean()
            html_content.append(f'<td><strong>{size_avg:.2f}</strong></td>')
            
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
                html_content.append('<h3>⚠️ 极端性能差异数据 (>300%)</h3>')
                html_content.append(f'<p><small>共发现 <strong>{len(extreme_data)}</strong> 个极端性能差异数据点，已从常规对比分析中排除</small></p>')
                
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
                    html_content.append('<thead><tr><th>算法</th><th>数据规模</th><th>性能差异</th><th>基准值</th><th>对比值</th></tr></thead>')
                    html_content.append('<tbody>')
                    
                    # 按差异程度排序，只显示前10个
                    gpu_extreme_data.sort(key=lambda x: abs(x['percentage_diff']), reverse=True)
                    for item in gpu_extreme_data[:10]:
                        data_size_label = f"{int(item['data_size'] * 1024)}KB" if item['data_size'] < 1 else \
                                        f"{int(item['data_size'])}MB" if item['data_size'] < 1024 else \
                                        f"{int(item['data_size'] / 1024)}GB"
                        
                        diff_class = 'positive' if item['percentage_diff'] > 0 else 'negative'
                        html_content.append('<tr>')
                        html_content.append(f'<td>{item["algorithm"]}</td>')
                        html_content.append(f'<td>{data_size_label}</td>')
                        html_content.append(f'<td class="{diff_class}"><strong>{item["percentage_diff"]:+.1f}%</strong></td>')
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

    def _write_html_with_statistics(self, fig, save_path: str):
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
        statistics_html = self.generate_statistics_tables()
        
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

    def generate_summary_report(self) -> str:
        """生成性能分析总结报告"""
        df = self.get_combined_dataframe()
        
        if df.empty:
            return "没有数据可以分析"
        
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
        
        # 各GPU平均性能
        report.append("Average Performance by GPU:")
        for gpu in df['gpu_label'].unique():
            avg_perf = df[df['gpu_label'] == gpu]['throughput'].mean()
            report.append(f"  {gpu}: {avg_perf:.2f} GB/s")
        report.append("")
        
        # 性能百分比差异分析（如果有多个GPU）
        gpu_labels = df['gpu_label'].unique()
        if len(gpu_labels) > 1:
            report.append("Performance Percentage Differences:")
            report.append(f"  Baseline GPU: {gpu_labels[0]}")
            report.append("")
            
            percentage_df, extreme_data, anomaly_data = self.calculate_percentage_differences()
            if percentage_df is not None:
                for gpu in gpu_labels[1:]:
                    gpu_data = percentage_df[percentage_df['gpu_label'] == gpu]
                    if not gpu_data.empty:
                        avg_diff = gpu_data['percentage_diff'].mean()
                        max_diff = gpu_data['percentage_diff'].max()
                        min_diff = gpu_data['percentage_diff'].min()
                        report.append(f"  {gpu} vs {gpu_labels[0]}:")
                        report.append(f"    Average: {avg_diff:+.1f}%")
                        report.append(f"    Max: {max_diff:+.1f}%")
                        report.append(f"    Min: {min_diff:+.1f}%")
                        report.append("")
            
            # 添加极端数据报告
            if extreme_data:
                report.append("Extreme Performance Differences (>300%):")
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
                    gpu_extreme_data.sort(key=lambda x: abs(x['percentage_diff']), reverse=True)
                    
                    for item in gpu_extreme_data[:10]:  # 只显示前10个最极端的
                        data_size_label = f"{int(item['data_size'] * 1024)}KB" if item['data_size'] < 1 else \
                                        f"{int(item['data_size'])}MB" if item['data_size'] < 1024 else \
                                        f"{int(item['data_size'] / 1024)}GB"
                        report.append(f"    {item['algorithm']} ({data_size_label}): {item['percentage_diff']:+.1f}% "
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
        
        # 各算法平均性能
        report.append("Average Performance by Algorithm:")
        for algo in df['algorithm'].unique():
            avg_perf = df[df['algorithm'] == algo]['throughput'].mean()
            report.append(f"  {algo}: {avg_perf:.2f} GB/s")
        
        return "\n".join(report)


def create_comprehensive_datasets():
    """创建综合性能数据集，包含更多算法类别和数据规模"""
    
    # 扩展的并行计算算法类别
    algorithms = {
        # 基础并行算法
        'sort': 'Parallel Sorting',
        'reduce': 'Parallel Reduction', 
        'scan': 'Prefix Sum/Scan',
        'histogram': 'Histogram Computing',
        'compact': 'Stream Compaction',
        
        # 线性代数算法
        'matmul': 'Matrix Multiplication',
        'gemv': 'Matrix-Vector Multiply',
        'gemm': 'General Matrix Multiply',
        'spmv': 'Sparse Matrix-Vector',
        'trsv': 'Triangular Solve',
        
        # 图像/信号处理算法
        'conv2d': '2D Convolution',
        'conv3d': '3D Convolution',
        'fft': 'Fast Fourier Transform',
        'gaussian_blur': 'Gaussian Blur',
        'bilateral_filter': 'Bilateral Filter',
        
        # 机器学习算法
        'dnn_training': 'DNN Training',
        'dnn_inference': 'DNN Inference',
        'cnn_forward': 'CNN Forward Pass',
        'cnn_backward': 'CNN Backward Pass',
        'transformer_attn': 'Transformer Attention',
        
        # 科学计算算法
        'stencil_2d': '2D Stencil Computation',
        'stencil_3d': '3D Stencil Computation',
        'molecular_dynamics': 'Molecular Dynamics',
        'monte_carlo': 'Monte Carlo Simulation',
        'n_body': 'N-Body Simulation'
    }
    
    # 扩展的数据规模范围 (几KB到几GB) - 以MB为单位的数值，以2倍递增
    data_sizes_mb = [
        0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096
    ]
    
    # 对应的显示标签  
    data_size_labels = [
        '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', 
        '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', 
        '256MB', '512MB', '1GB', '2GB', '4GB'
    ]
    
    # GPU型号及其性能特征 (只测试3个GPU)
    gpu_configs = {
        'RTX_4090': {
            'name': 'RTX_4090',
            'memory_bandwidth': 1008,  # GB/s
            'compute_units': 128,
            'base_clock': 2230,  # MHz
            'memory_size': 24,   # GB
        },
        'A100': {
            'name': 'A100',
            'memory_bandwidth': 1935,
            'compute_units': 108,
            'base_clock': 1410,
            'memory_size': 80,
        },
        'H100': {
            'name': 'H100',
            'memory_bandwidth': 3350,
            'compute_units': 132,
            'base_clock': 1980,
            'memory_size': 80,
        }
    }
    
    np.random.seed(42)
    
    def mb_to_bytes(size_mb):
        """将MB为单位的数值转换为字节数"""
        return size_mb * 1024 * 1024
    
    def calculate_roofline_performance(algorithm, data_size_mb, gpu_config):
        """简化的roofline模型：y=a*x 线性关系"""
        
        # 算法特性系数 - 不同算法有不同的线性系数，可能互有胜负
        algorithm_coefficients = {
            # 基础并行算法
            'sort': 6.8, 'reduce': 8.2, 'scan': 7.5, 'histogram': 5.9, 'compact': 6.4,
            # 线性代数算法  
            'matmul': 9.1, 'gemv': 7.8, 'gemm': 9.5, 'spmv': 6.2, 'trsv': 7.3,
            # 图像处理算法
            'conv2d': 8.7, 'conv3d': 9.3, 'fft': 7.1, 'gaussian_blur': 8.0, 'bilateral_filter': 8.4,
            # 机器学习算法
            'dnn_training': 9.8, 'dnn_inference': 8.6, 'cnn_forward': 9.2, 'cnn_backward': 9.6, 'transformer_attn': 8.9,
            # 科学计算算法
            'stencil_2d': 7.6, 'stencil_3d': 8.1, 'molecular_dynamics': 8.5, 'monte_carlo': 8.8, 'n_body': 8.3
        }
        
        # GPU差异系数 - 不同GPU差距不大，大约20%范围内
        gpu_multipliers = {
            'RTX_4090': 1.0,    # 基准
            'A100': 1.08,       # 稍高8%
            'H100': 1.04        # 稍高4%
        }
        
        # 获取算法系数和GPU系数
        a = algorithm_coefficients.get(algorithm, 7.5)  # 默认系数7.5
        gpu_factor = gpu_multipliers.get(gpu_config.get('name', 'RTX_4090'), 1.0)
        
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
        
        for algorithm in algorithms.keys():
            for i, data_size_mb in enumerate(data_sizes_mb):
                # 检查GPU内存限制
                data_bytes = mb_to_bytes(data_size_mb)
                if data_bytes > gpu_config['memory_size'] * 1024**3:
                    continue  # 跳过超出GPU内存的数据大小
                
                throughput = calculate_roofline_performance(algorithm, data_size_mb, gpu_config)
                
                gpu_data.append({
                    'algorithm': algorithm,
                    'data_size_mb': data_size_mb,  # 数值形式
                    'data_size_label': data_size_labels[i],  # 显示标签
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
    
    print(f"\n共生成 {len(datasets)} 个GPU数据集，覆盖 {len(algorithms)} 种算法类型")
    print(f"数据规模范围: {data_size_labels[0]} - {data_size_labels[-1]}")
    
    return datasets

def create_sample_datasets():
    """创建简单示例数据集（向后兼容）"""
    return create_comprehensive_datasets()


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
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_sample:
        create_sample_datasets()
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
        height=args.height
    )
    
    # 生成分析报告
    print("\n" + analyzer.generate_summary_report())


if __name__ == "__main__":
    main()
