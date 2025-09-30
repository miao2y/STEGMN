# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_experiments(logs_dir="logs"):
    """
    统计logs文件夹下所有实验的best_train, best_val, best_test, best_epoch(根据val)
    区分md17_logs和mdanalysis_logs
    新的数据结构包含 loss、r2、mae 三个指标，每个指标都有 train、val、test 版本
    
    Args:
        logs_dir (str): logs文件夹路径
    
    Returns:
        pd.DataFrame: 包含所有实验结果的汇总表格
    """
    
    results = []
    
    # 检查是否存在md17_logs和mdanalysis_logs子目录
    md17_logs_path = os.path.join(logs_dir, "md17_logs")
    mdanalysis_logs_path = os.path.join(logs_dir, "mdanalysis_logs")
    
    # 处理md17_logs
    if os.path.exists(md17_logs_path):
        print("正在分析md17_logs...")
        results.extend(analyze_logs_directory(md17_logs_path, "md17"))
    
    # 处理mdanalysis_logs
    if os.path.exists(mdanalysis_logs_path):
        print("正在分析mdanalysis_logs...")
        results.extend(analyze_logs_directory(mdanalysis_logs_path, "mdanalysis"))
    
    # 创建DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # 按模型和数据集排序
        df = df.sort_values(['model', 'dataset'])
        
        # 格式化数值列
        numeric_columns = [
            'best_train_loss', 'best_val_loss', 'best_test_loss',
            'best_train_r2', 'best_val_r2', 'best_test_r2',
            'best_train_mae', 'best_val_mae', 'best_test_mae'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: "{:.6f}".format(x) if x is not None else "N/A")
        
        return df
    else:
        return pd.DataFrame()

def analyze_logs_directory(logs_path, dataset_type):
    """
    分析指定日志目录下的实验
    
    Args:
        logs_path (str): 日志目录路径
        dataset_type (str): 数据集类型 ('md17' 或 'mdanalysis')
    
    Returns:
        list: 实验结果列表
    """
    results = []
    
    # 遍历目录下的所有子目录
    for dir_name in os.listdir(logs_path):
        dir_path = os.path.join(logs_path, dir_name)
        
        # 检查是否是目录且不是隐藏目录
        if os.path.isdir(dir_path) and not dir_name.startswith('.'):
            loss_json_path = os.path.join(dir_path, 'loss.json')
            
            # 检查是否存在loss.json文件
            if os.path.exists(loss_json_path):
                try:
                    # 解析实验名称
                    parts = dir_name.split('-')
                    if len(parts) >= 2:
                        model_name = parts[0]
                        dataset_name = '-'.join(parts[1:])  # 处理数据集名可能包含多个'-'的情况
                    else:
                        model_name = dir_name
                        dataset_name = "unknown"
                    
                    # 读取loss.json文件
                    with open(loss_json_path, 'r') as f:
                        data = json.load(f)
                    
                    # 提取数据
                    epochs = data.get('epochs', [])
                    train_losses = data.get('train loss', [])
                    val_losses = data.get('val loss', [])
                    test_losses = data.get('test loss', [])
                    train_r2 = data.get('train r2', [])
                    val_r2 = data.get('val r2', [])
                    test_r2 = data.get('test r2', [])
                    train_mae = data.get('train mae', [])
                    val_mae = data.get('val mae', [])
                    test_mae = data.get('test mae', [])
                    
                    if epochs and val_losses:
                        # 找到最佳验证损失的epoch
                        best_val_idx = np.argmin(val_losses)
                        best_epoch = epochs[best_val_idx]
                        best_val_loss = val_losses[best_val_idx]
                        
                        # 获取对应的train和test损失
                        best_train_loss = train_losses[best_val_idx] if best_val_idx < len(train_losses) else None
                        best_test_loss = test_losses[best_val_idx] if best_val_idx < len(test_losses) else None
                        
                        # 获取对应的r2值
                        best_train_r2 = train_r2[best_val_idx] if best_val_idx < len(train_r2) else None
                        best_val_r2 = val_r2[best_val_idx] if best_val_idx < len(val_r2) else None
                        best_test_r2 = test_r2[best_val_idx] if best_val_idx < len(test_r2) else None
                        
                        # 获取对应的mae值
                        best_train_mae = train_mae[best_val_idx] if best_val_idx < len(train_mae) else None
                        best_val_mae = val_mae[best_val_idx] if best_val_idx < len(val_mae) else None
                        best_test_mae = test_mae[best_val_idx] if best_val_idx < len(test_mae) else None
                        
                        # 添加到结果列表
                        results.append({
                            'model': model_name,
                            'dataset': dataset_name,
                            'dataset_type': dataset_type,
                            'best_train_loss': best_train_loss,
                            'best_val_loss': best_val_loss,
                            'best_test_loss': best_test_loss,
                            'best_train_r2': best_train_r2,
                            'best_val_r2': best_val_r2,
                            'best_test_r2': best_test_r2,
                            'best_train_mae': best_train_mae,
                            'best_val_mae': best_val_mae,
                            'best_test_mae': best_test_mae,
                            'best_epoch': best_epoch,
                            'experiment_path': dir_path
                        })
                        
                except Exception as e:
                    print("处理实验 {} 时出错: {}".format(dir_name, e))
                    continue
    
    return results

def print_summary_table(df):
    """
    打印格式化的汇总表格
    
    Args:
        df (pd.DataFrame): 实验结果DataFrame
    """
    if df.empty:
        print("未找到任何实验结果")
        return
    
    print("=" * 150)
    print("实验汇总表格")
    print("=" * 150)
    
    # 设置pandas显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 15)
    
    # 打印表格
    print(df.to_string(index=False))
    print("=" * 150)
    
    # 打印统计信息
    print("\n总计实验数量: {}".format(len(df)))
    print("模型数量: {}".format(df['model'].nunique()))
    print("数据集数量: {}".format(df['dataset'].nunique()))
    
    # 按数据集类型统计
    if 'dataset_type' in df.columns:
        print("\n按数据集类型统计:")
        dataset_type_stats = df.groupby('dataset_type').size().sort_values(ascending=False)
        for dataset_type, count in dataset_type_stats.items():
            print("  {}: {} 个实验".format(dataset_type, count))
    
    # 按模型统计
    print("\n按模型统计:")
    model_stats = df.groupby('model').size().sort_values(ascending=False)
    for model, count in model_stats.items():
        print("  {}: {} 个实验".format(model, count))
    
    # 按数据集统计
    print("\n按数据集统计:")
    dataset_stats = df.groupby('dataset').size().sort_values(ascending=False)
    for dataset, count in dataset_stats.items():
        print("  {}: {} 个实验".format(dataset, count))

def generate_detailed_analysis(df):
    """
    生成详细的分析报告
    
    Args:
        df (pd.DataFrame): 实验结果DataFrame
    """
    if df.empty:
        return
    
    # 创建详细分析报告
    report = []
    report.append("=" * 80)
    report.append("详细实验分析报告")
    report.append("=" * 80)
    
    # 按模型分析
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        report.append("\n模型: {}".format(model))
        report.append("-" * 40)
        
        # 计算平均性能
        numeric_cols = [col for col in df.columns if col.startswith('best_') and col != 'best_epoch']
        for col in numeric_cols:
            if col in model_df.columns:
                values = pd.to_numeric(model_df[col], errors='coerce')
                mean_val = values.mean()
                std_val = values.std()
                if not pd.isna(mean_val):
                    report.append("{}: {:.6f} ± {:.6f}".format(col, mean_val, std_val))
    
    # 按数据集分析
    report.append("\n\n数据集性能分析")
    report.append("=" * 40)
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        report.append("\n数据集: {}".format(dataset))
        report.append("-" * 30)
        
        # 找到该数据集上表现最好的模型
        if 'best_val_loss' in dataset_df.columns:
            best_model_idx = dataset_df['best_val_loss'].astype(float).idxmin()
            best_model = dataset_df.loc[best_model_idx, 'model']
            best_loss = dataset_df.loc[best_model_idx, 'best_val_loss']
            report.append("最佳模型: {} (val_loss: {})".format(best_model, best_loss))
    
    # 保存详细报告
    with open('result/detailed_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("详细分析报告已保存到 result/detailed_analysis_report.txt")

def main():
    """
    主函数
    """
    # 分析实验
    print("正在分析实验数据...")
    df = analyze_experiments()
    
    if not df.empty:
        # 打印汇总表格
        print_summary_table(df)
        
        # 保存到CSV文件到result目录
        output_file = "result/experiment_summary.csv"
        df.to_csv(output_file, index=False)
        print("\n结果已保存到 {}".format(output_file))
        
        # 分别保存md17和mdanalysis的结果
        if 'dataset_type' in df.columns:
            md17_df = df[df['dataset_type'] == 'md17']
            if not md17_df.empty:
                md17_df.to_csv("result/experiment_summary_md17.csv", index=False)
                print("MD17实验结果已保存到 result/experiment_summary_md17.csv")
            
            mdanalysis_df = df[df['dataset_type'] == 'mdanalysis']
            if not mdanalysis_df.empty:
                mdanalysis_df.to_csv("result/experiment_summary_mdanalysis.csv", index=False)
                print("MDAnalysis实验结果已保存到 result/experiment_summary_mdanalysis.csv")
        
        # 生成详细分析报告
        generate_detailed_analysis(df)
        
        # 生成性能对比表格
        generate_performance_comparison(df)
        
    else:
        print("未找到任何实验结果")

def generate_performance_comparison(df):
    """
    生成性能对比表格
    
    Args:
        df (pd.DataFrame): 实验结果DataFrame
    """
    if df.empty:
        return
    
    # 创建性能对比表格
    comparison_data = []
    
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        
        # 找到最佳模型（基于验证损失）
        if 'best_val_loss' in dataset_df.columns:
            best_idx = dataset_df['best_val_loss'].astype(float).idxmin()
            best_model = dataset_df.loc[best_idx, 'model']
            
            # 收集所有模型在该数据集上的性能
            for _, row in dataset_df.iterrows():
                comparison_data.append({
                    'dataset': dataset,
                    'model': row['model'],
                    'val_loss': row['best_val_loss'],
                    'test_loss': row['best_test_loss'],
                    'val_r2': row['best_val_r2'],
                    'test_r2': row['best_test_r2'],
                    'val_mae': row['best_val_mae'],
                    'test_mae': row['best_test_mae'],
                    'is_best': row['model'] == best_model
                })
    
    # 创建对比DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存对比表格
    comparison_df.to_csv("result/performance_comparison.csv", index=False)
    print("性能对比表格已保存到 result/performance_comparison.csv")

if __name__ == "__main__":
    main()
