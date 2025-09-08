#!/usr/bin/env python3
"""
Custom callbacks for training models
"""

import os
import json
import yaml
import pickle
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime


class EpochSaveCallback(tf.keras.callbacks.Callback):
    """
    自定义回调函数：保存每个epoch的模型
    """
    
    def __init__(self, save_dir, save_format='h5', save_weights_only=False):
        """
        Args:
            save_dir: 保存目录
            save_format: 保存格式 'h5' 或 'tf'
            save_weights_only: 是否只保存权重
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_format = save_format
        self.save_weights_only = save_weights_only
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'epochs'), exist_ok=True)
        
        # 训练历史记录
        self.epoch_history = []
        
    def on_train_begin(self, logs=None):
        """训练开始时调用"""
        print(f"\n🚀 Training started!")
        print(f"📁 Models will be saved to: {self.save_dir}")
        
        # 复制 action_predict.py 到模型目录（在训练开始时）
        try:
            # 获取项目根目录（假设callback文件在项目根目录）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            source_file = os.path.join(script_dir, "action_predict.py")
            target_file = os.path.join(self.save_dir, "action_predict.py")
            
            if os.path.exists(source_file):
                if not os.path.exists(target_file):
                    shutil.copy2(source_file, target_file)
                    print(f"📋 已复制 action_predict.py 到模型目录")
                else:
                    print(f"📁 action_predict.py 已存在于模型目录中")
            else:
                print(f"⚠️  未找到源文件: {source_file}")
        except Exception as e:
            print(f"❌ 复制 action_predict.py 失败: {str(e)}")
        
    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时调用"""
        logs = logs or {}
        
        # 保存当前epoch信息
        epoch_info = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        self.epoch_history.append(epoch_info)
        
        # 保存每个epoch的模型
        epoch_filename = f"epoch_{epoch+1:03d}_loss_{logs.get('val_loss', 0):.4f}_acc_{logs.get('val_accuracy', 0):.4f}"
        if self.save_format == 'h5':
            epoch_filepath = os.path.join(self.save_dir, 'epochs', f"{epoch_filename}.h5")
            if self.save_weights_only:
                self.model.save_weights(epoch_filepath)
            else:
                self.model.save(epoch_filepath)
        else:
            epoch_filepath = os.path.join(self.save_dir, 'epochs', epoch_filename)
            self.model.save(epoch_filepath, save_format='tf')
        
        # 保存训练历史
        self._save_training_history()
        
    def _save_training_history(self):
        """保存训练历史"""
        # 保存为JSON格式
        # json_path = os.path.join(self.save_dir, 'training_history.json')
        # with open(json_path, 'w') as f:
        #     json.dump(self.epoch_history, f, indent=2)
        
        # 保存为YAML格式
        yaml_path = os.path.join(self.save_dir, 'training_history.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(self.epoch_history, f, default_flow_style=False)
    
    def on_train_end(self, logs=None):
        """训练结束时调用"""
        print(f"\n🎯 Training completed!")
        print(f"📁 All epoch models saved in: {os.path.join(self.save_dir, 'epochs')}")


class DetailedLoggingCallback(tf.keras.callbacks.Callback):
    """详细的日志记录回调"""
    
    def __init__(self, log_dir, log_frequency=1):
        """
        Args:
            log_dir: 日志保存目录
            log_frequency: 记录频率（每n个epoch记录一次）
        """
        super().__init__()
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化日志文件
        self.log_file = os.path.join(log_dir, 'detailed_training.log')
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("=" * 80 + "\n")
    
    def on_epoch_end(self, epoch, logs=None):
        """记录详细的epoch信息"""
        if (epoch + 1) % self.log_frequency == 0:
            logs = logs or {}
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            with open(self.log_file, 'a') as f:
                f.write(f"\nEpoch {epoch + 1} - {timestamp}\n")
                f.write("-" * 40 + "\n")
                for key, value in logs.items():
                    f.write(f"{key:15s}: {value:.6f}\n")
                f.write("-" * 40 + "\n")
    
    def on_train_end(self, logs=None):
        """训练结束时记录"""
        with open(self.log_file, 'a') as f:
            f.write(f"\nTraining completed at {datetime.now()}\n")
            f.write("=" * 80 + "\n")


class MetricsVisualizationCallback(tf.keras.callbacks.Callback):
    """训练指标可视化回调（可选，需要matplotlib）"""
    
    def __init__(self, save_dir, plot_frequency=5):
        """
        Args:
            save_dir: 图片保存目录
            plot_frequency: 绘图频率（每n个epoch绘制一次）
        """
        super().__init__()
        self.save_dir = save_dir
        self.plot_frequency = plot_frequency
        self.metrics_history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 检查是否有matplotlib
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plot_available = True
        except ImportError:
            print("Warning: matplotlib not available, skipping plots")
            self.plot_available = False
    
    def on_epoch_end(self, epoch, logs=None):
        """记录指标并绘图"""
        if not self.plot_available:
            return
            
        logs = logs or {}
        
        # 记录指标
        for key in self.metrics_history:
            if key in logs:
                self.metrics_history[key].append(logs[key])
        
        # 定期绘图
        if (epoch + 1) % self.plot_frequency == 0:
            self._plot_metrics(epoch + 1)
    
    def _plot_metrics(self, epoch):
        """绘制训练指标"""
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失
        ax1.plot(self.metrics_history['loss'], label='Training Loss', color='blue')
        if self.metrics_history['val_loss']:
            ax1.plot(self.metrics_history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率
        ax2.plot(self.metrics_history['accuracy'], label='Training Accuracy', color='blue')
        if self.metrics_history['val_accuracy']:
            ax2.plot(self.metrics_history['val_accuracy'], label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 保存图片
        plot_path = os.path.join(self.save_dir, f'training_metrics_epoch_{epoch:03d}.png')
        self.plt.tight_layout()
        self.plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        self.plt.close()
        
        # print(f"📈 Metrics plot saved: {os.path.basename(plot_path)}")
