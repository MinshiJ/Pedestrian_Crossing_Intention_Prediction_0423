import matplotlib.pyplot as plt
import os
import numpy as np
from utils import get_path

def save_training_plots(history, path_params, model_name=''):
    """
    保存训练历史的所有图像
    
    Args:
        history: Keras训练历史对象
        path_params: 路径参数字典，包含保存文件夹等信息
        model_name: 模型名称，用于区分不同模型的绘图策略
    """
    # 获取所有可用的指标
    available_metrics = list(history.history.keys())
    print(f"Available metrics: {available_metrics}")
    
    # 根据模型类型绘制不同的图像
    if model_name == 'Transformer_depth':
        # _plot_transformer_depth_model(history, path_params, available_metrics)
        pass
    else:
        _plot_standard_model(history, path_params, available_metrics)
    
    # 保存所有指标的详细图（每个指标单独一张图）
    _plot_individual_metrics(history, path_params, available_metrics)
    
    print(f"Training plots saved to model directory")

def _plot_transformer_depth_model(history, path_params, available_metrics):
    """绘制Transformer_depth模型的多任务图表"""
    # Transformer_depth模型的多任务绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training History - Transformer_depth Model', fontsize=16)
    
    # 意图预测损失
    if 'intention_loss' in available_metrics:
        axes[0, 0].plot(history.history['intention_loss'], label='Train Loss (Intention)', color='blue')
        if 'val_intention_loss' in available_metrics:
            axes[0, 0].plot(history.history['val_intention_loss'], label='Val Loss (Intention)', color='red')
        axes[0, 0].set_title('Intention Prediction Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # 意图预测准确率
    if 'intention_accuracy' in available_metrics:
        axes[0, 1].plot(history.history['intention_accuracy'], label='Train Accuracy (Intention)', color='blue')
        if 'val_intention_accuracy' in available_metrics:
            axes[0, 1].plot(history.history['val_intention_accuracy'], label='Val Accuracy (Intention)', color='red')
        axes[0, 1].set_title('Intention Prediction Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 轨迹预测损失
    if 'etraj_loss' in available_metrics:
        axes[1, 0].plot(history.history['etraj_loss'], label='Train Loss (Trajectory)', color='green')
        if 'val_etraj_loss' in available_metrics:
            axes[1, 0].plot(history.history['val_etraj_loss'], label='Val Loss (Trajectory)', color='orange')
        axes[1, 0].set_title('Trajectory Prediction Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (MSE)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 轨迹预测MAE
    if 'etraj_mae' in available_metrics:
        axes[1, 1].plot(history.history['etraj_mae'], label='Train MAE (Trajectory)', color='green')
        if 'val_etraj_mae' in available_metrics:
            axes[1, 1].plot(history.history['val_etraj_mae'], label='Val MAE (Trajectory)', color='orange')
        axes[1, 1].set_title('Trajectory Prediction MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path, _ = get_path(**path_params, file_name='transformer_depth_training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存Precision和Recall图
    if 'intention_precision' in available_metrics or 'intention_recall' in available_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Precision and Recall - Intention Prediction', fontsize=14)
        
        if 'intention_precision' in available_metrics:
            axes[0].plot(history.history['intention_precision'], label='Train Precision', color='purple')
            if 'val_intention_precision' in available_metrics:
                axes[0].plot(history.history['val_intention_precision'], label='Val Precision', color='pink')
            axes[0].set_title('Precision')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Precision')
            axes[0].legend()
            axes[0].grid(True)
        
        if 'intention_recall' in available_metrics:
            axes[1].plot(history.history['intention_recall'], label='Train Recall', color='brown')
            if 'val_intention_recall' in available_metrics:
                axes[1].plot(history.history['val_intention_recall'], label='Val Recall', color='yellow')
            axes[1].set_title('Recall')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Recall')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plot_path, _ = get_path(**path_params, file_name='transformer_depth_precision_recall.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def _plot_standard_model(history, path_params, available_metrics):
    """绘制标准模型的训练历史图表"""
    # 计算需要的子图数量
    metrics_to_plot = []
    val_metrics = []
    
    for metric in available_metrics:
        if not metric.startswith('val_'):
            metrics_to_plot.append(metric)
            if f'val_{metric}' in available_metrics:
                val_metrics.append(f'val_{metric}')
    
    num_metrics = len(metrics_to_plot)
    if num_metrics == 0:
        print("No metrics found to plot")
        return
    
    # 计算子图布局
    cols = min(2, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Training History', fontsize=16)
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i] if num_metrics > 1 else axes[0]
        
        # 绘制训练指标
        ax.plot(history.history[metric], label=f'Train {metric.title()}', linewidth=2)
        
        # 绘制验证指标（如果存在）
        val_metric = f'val_{metric}'
        if val_metric in available_metrics:
            ax.plot(history.history[val_metric], label=f'Val {metric.title()}', linewidth=2)
        
        ax.set_title(f'{metric.title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plot_path, _ = get_path(**path_params, file_name='training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def _plot_individual_metrics(history, path_params, available_metrics):
    """为每个指标保存单独的详细图表"""
    for metric in available_metrics:
        if not metric.startswith('val_'):
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[metric], label=f'Train {metric.title()}', 
                    linewidth=2, marker='o', markersize=4)
            
            val_metric = f'val_{metric}'
            if val_metric in available_metrics:
                plt.plot(history.history[val_metric], label=f'Val {metric.title()}', 
                        linewidth=2, marker='s', markersize=4)
            
            plt.title(f'{metric.title()} Over Epochs', fontsize=14)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel(metric.title(), fontsize=12)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # 保存单个指标图
            safe_metric_name = metric.replace('/', '_').replace(' ', '_')
            plot_path, _ = get_path(**path_params, file_name=f'{safe_metric_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()