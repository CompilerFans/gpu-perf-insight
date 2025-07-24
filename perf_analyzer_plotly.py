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
                tickvals=[0.004, 0.016, 0.064, 0.256, 1, 4, 16, 64, 256, 1024, 2048, 4096],
                ticktext=['4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB', '1GB', '2GB', '4GB']
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
    
    def calculate_percentage_differences(self) -> Optional[pd.DataFrame]:
        """计算相对于第一个GPU的性能百分比差异"""
        df = self.get_combined_dataframe()
        if df.empty or len(df['gpu_label'].unique()) <= 1:
            return None
        
        gpu_labels = df['gpu_label'].unique()
        if len(gpu_labels) < 2:
            return None
        
        # 以第一个GPU为基准
        baseline_gpu = gpu_labels[0]
        comparison_gpus = gpu_labels[1:]
        
        percentage_data = []
        
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
                    
                    if baseline_throughput > 0:
                        percentage_diff = ((comparison_throughput - baseline_throughput) / baseline_throughput) * 100
                    else:
                        percentage_diff = 0
                    
                    percentage_data.append({
                        'algorithm': algorithm,
                        'data_size': data_size,
                        'gpu_label': gpu,
                        'percentage_diff': percentage_diff,
                        'baseline_throughput': baseline_throughput,
                        'comparison_throughput': comparison_throughput
                    })
        
        return pd.DataFrame(percentage_data)

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
            
            if num_comparisons <= 2:
                # 3个子图布局：原始图、百分比图、散点图显示每个具体数据点
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('原始性能对比', f'相对于 {gpu_labels[0]} 的性能百分比差异', '性能差异分布与数据来源分析 - 直方图+散点图'),
                    vertical_spacing=0.1,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
                    row_heights=[0.35, 0.35, 0.3]
                )
                total_height = int(height * 2)
            else:
                # 4个子图布局：原始图、百分比图、每个GPU的独立散点图
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('原始性能对比', f'相对于 {gpu_labels[0]} 的性能百分比差异', 
                                  f'{comparison_gpus[0]} vs {gpu_labels[0]} 分布+数据点', f'{comparison_gpus[1]} vs {gpu_labels[0]} 分布+数据点'),
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1,
                    specs=[[{"secondary_y": False}, {"secondary_y": False}], 
                           [{"secondary_y": False}, {"secondary_y": False}]],
                    row_heights=[0.5, 0.5],
                    column_widths=[0.5, 0.5]
                )
                total_height = height
                width = int(width * 1.5)
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
            percentage_df = self.calculate_percentage_differences()
            
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
                paper_bgcolor='white'
            )
            
            # 更新第一个子图的轴标签
            fig.update_xaxes(
                title=dict(text="Data Size (MB)", font=dict(size=14)),
                type="log",
                tickmode='array',
                tickvals=[0.004, 0.016, 0.064, 0.256, 1, 4, 16, 64, 256, 1024, 2048, 4096],
                ticktext=['4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB', '1GB', '2GB', '4GB'],
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
                tickvals=[0.004, 0.016, 0.064, 0.256, 1, 4, 16, 64, 256, 1024, 2048, 4096],
                ticktext=['4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', '64MB', '256MB', '1GB', '2GB', '4GB'],
                row=2, col=1
            )
            fig.update_yaxes(
                title=dict(text="Performance Difference (%)", font=dict(size=14)),
                row=2, col=1
            )
            
            if num_comparisons <= 2:
                # 3个子图布局的轴标签
                fig.update_xaxes(title=dict(text="Data Size (MB)", font=dict(size=14)), row=1, col=1)
                fig.update_yaxes(title=dict(text="Throughput (GB/s)", font=dict(size=14)), row=1, col=1)
                fig.update_xaxes(title=dict(text="Data Size (MB)", font=dict(size=14)), row=2, col=1)
                fig.update_yaxes(title=dict(text="Performance Difference (%)", font=dict(size=14)), row=2, col=1)
                fig.update_xaxes(title=dict(text="性能差异百分比", font=dict(size=14)), row=3, col=1)
                fig.update_yaxes(title=dict(text="", font=dict(size=14)), row=3, col=1, showticklabels=False)
            else:
                # 4个子图布局的轴标签
                fig.update_xaxes(title=dict(text="Data Size (MB)", font=dict(size=14)), row=1, col=1)
                fig.update_yaxes(title=dict(text="Throughput (GB/s)", font=dict(size=14)), row=1, col=1)
                fig.update_xaxes(title=dict(text="Data Size (MB)", font=dict(size=14)), row=1, col=2)
                fig.update_yaxes(title=dict(text="Performance Difference (%)", font=dict(size=14)), row=1, col=2)
                fig.update_xaxes(title=dict(text="性能差异百分比", font=dict(size=14)), row=2, col=1)
                fig.update_yaxes(title=dict(text="", font=dict(size=14)), row=2, col=1, showticklabels=False)
                fig.update_xaxes(title=dict(text="性能差异百分比", font=dict(size=14)), row=2, col=2)
                fig.update_yaxes(title=dict(text="", font=dict(size=14)), row=2, col=2, showticklabels=False)
            
            # 添加零线到百分比图
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=2, col=1)
            
            # 添加散点图显示每个具体数据点的算法来源
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
                
                if num_comparisons <= 2:
                    # 创建散点图显示每个具体数据点的算法来源
                    for i, gpu in enumerate(comparison_gpus):
                        gpu_data = percentage_df[percentage_df['gpu_label'] == gpu]
                        
                        if not gpu_data.empty:
                            # 为每个数据点创建散点图
                            scatter_points = []
                            for _, row in gpu_data.iterrows():
                                scatter_points.append({
                                    'x': row['percentage_diff'],
                                    'y': 1,  # 固定y值，所有点在同一水平线上
                                    'algorithm': row['algorithm'],
                                    'data_size': mb_to_label(row['data_size']),
                                    'baseline_throughput': row['baseline_throughput'],
                                    'comparison_throughput': row['comparison_throughput'],
                                    'percentage_diff': row['percentage_diff']
                                })
                            
                            if scatter_points:
                                scatter_df = pd.DataFrame(scatter_points)
                                
                                # 根据性能差异设置颜色：红色表示负值，绿色表示正值
                                colors_list = []
                                for diff in scatter_df['x']:
                                    if diff < 0:
                                        colors_list.append('rgba(255, 0, 0, 0.8)')  # 红色
                                    else:
                                        colors_list.append('rgba(0, 128, 0, 0.8)')  # 绿色
                                
                                # 创建散点图
                                scatter_trace = go.Scatter(
                                    x=scatter_df['x'],
                                    y=scatter_df['y'],
                                    mode='markers',
                                    name=f'{gpu} vs {gpu_labels[0]}',
                                    marker=dict(
                                        size=12,
                                        color=colors_list,
                                        line=dict(color='white', width=1)
                                    ),
                                    customdata=list(zip(
                                        scatter_df['algorithm'],
                                        scatter_df['data_size'],
                                        scatter_df['baseline_throughput'],
                                        scatter_df['comparison_throughput'],
                                        scatter_df['percentage_diff']
                                    )),
                                    hovertemplate=(
                                        f"<b>{gpu} vs {gpu_labels[0]}</b><br>"
                                        "算法: %{customdata[0]}<br>"
                                        "数据量: %{customdata[1]}<br>"
                                        "性能差异: %{customdata[4]:+.1f}%<br>"
                                        f"{gpu_labels[0]}: %{{customdata[2]:.2f}} GB/s<br>"
                                        f"{gpu}: %{{customdata[3]:.2f}} GB/s<br>"
                                        "<extra></extra>"
                                    )
                                )
                                
                                fig.add_trace(scatter_trace, row=3, col=1)
                    
                    # 添加垂直参考线
                    all_differences = percentage_df['percentage_diff']
                    mean_diff = all_differences.mean()
                    median_diff = all_differences.median()
                    
                    # 添加统计参考线
                    fig.add_vline(x=mean_diff, line_dash="dash", line_color="red", 
                                annotation_text=f"均值: {mean_diff:.0f}%", 
                                annotation_position="top left", row=3, col=1)
                    fig.add_vline(x=median_diff, line_dash="dot", line_color="green", 
                                annotation_text=f"中位数: {median_diff:.0f}%", 
                                annotation_position="bottom right", row=3, col=1)
                    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, row=3, col=1)
                    
                else:
                    # 为每个GPU创建独立的散点图
                    for i, gpu in enumerate(comparison_gpus):
                        gpu_data = percentage_df[percentage_df['gpu_label'] == gpu]
                        
                        if not gpu_data.empty:
                            row = 2  # 第2行
                            col = 1 if i < 2 else 2  # 第1或2列
                            
                            # 为每个数据点创建散点图
                            scatter_points = []
                            for _, row in gpu_data.iterrows():
                                scatter_points.append({
                                    'x': row['percentage_diff'],
                                    'y': 1,  # 固定y值，所有点在同一水平线上
                                    'algorithm': row['algorithm'],
                                    'data_size': mb_to_label(row['data_size']),
                                    'baseline_throughput': row['baseline_throughput'],
                                    'comparison_throughput': row['comparison_throughput'],
                                    'percentage_diff': row['percentage_diff']
                                })
                            
                            if scatter_points:
                                scatter_df = pd.DataFrame(scatter_points)
                                
                                # 根据性能差异设置颜色：红色表示负值，绿色表示正值
                                colors_list = []
                                for diff in scatter_df['x']:
                                    if diff < 0:
                                        colors_list.append('rgba(255, 0, 0, 0.8)')  # 红色
                                    else:
                                        colors_list.append('rgba(0, 128, 0, 0.8)')  # 绿色
                                
                                # 创建散点图
                                scatter_trace = go.Scatter(
                                    x=scatter_df['x'],
                                    y=scatter_df['y'],
                                    mode='markers',
                                    name=f'{gpu} vs {gpu_labels[0]}',
                                    marker=dict(
                                        size=12,
                                        color=colors_list,
                                        line=dict(color='white', width=1)
                                    ),
                                    customdata=list(zip(
                                        scatter_df['algorithm'],
                                        scatter_df['data_size'],
                                        scatter_df['baseline_throughput'],
                                        scatter_df['comparison_throughput'],
                                        scatter_df['percentage_diff']
                                    )),
                                    hovertemplate=(
                                        f"<b>{gpu} vs {gpu_labels[0]}</b><br>"
                                        "算法: %{customdata[0]}<br>"
                                        "数据量: %{customdata[1]}<br>"
                                        "性能差异: %{customdata[4]:+.1f}%<br>"
                                        f"{gpu_labels[0]}: %{{customdata[2]:.2f}} GB/s<br>"
                                        f"{gpu}: %{{customdata[3]:.2f}} GB/s<br>"
                                        "<extra></extra>"
                                    )
                                )
                                
                                fig.add_trace(scatter_trace, row=row, col=col)
                                
                                # 添加垂直参考线
                                gpu_diff = gpu_data['percentage_diff']
                                mean_gpu_diff = gpu_diff.mean()
                                median_gpu_diff = gpu_diff.median()
                                
                                fig.add_vline(x=mean_gpu_diff, line_dash="dash", line_color="red", 
                                            annotation_text=f"均值: {mean_gpu_diff:.0f}%", 
                                            annotation_position="top left", row=row, col=col)
                                fig.add_vline(x=median_gpu_diff, line_dash="dot", line_color="green", 
                                            annotation_text=f"中位数: {median_gpu_diff:.0f}%", 
                                            annotation_position="bottom right", row=row, col=col)
                                fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1, row=row, col=col)
                
                # 负值统计（简化显示）
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
            
            # 添加网格
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=1, col=1)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=2, col=1)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=3, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=3, col=1)
            
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
                fig.write_html(save_path)
                print(f"交互式图表已保存: {save_path}")
            else:
                fig.write_image(save_path, width=width, height=total_height)
                print(f"静态图表已保存: {save_path}")
        
        # 不自动显示图表，避免输出HTML内容
        return fig

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
            
            percentage_df = self.calculate_percentage_differences()
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
    
    # 扩展的数据规模范围 (几KB到几GB) - 以MB为单位的数值
    data_sizes_mb = [
        0.004, 0.016, 0.064, 0.256, 1, 4, 16, 
        64, 256, 1024, 2048, 4096
    ]
    
    # 对应的显示标签
    data_size_labels = [
        '4KB', '16KB', '64KB', '256KB', '1MB', '4MB', '16MB', 
        '64MB', '256MB', '1GB', '2GB', '4GB'
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
