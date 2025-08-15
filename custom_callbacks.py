#!/usr/bin/env python3
"""
Custom callbacks for training models
"""

import os
import json
import yaml
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime


class EpochSaveCallback(tf.keras.callbacks.Callback):
    """
    自定义回调函数：保存每个epoch的模型和训练信息
    """
    
    def __init__(self, save_dir, save_best_k=5, monitor='val_loss', mode='min', 
                 save_weights_only=False, save_format='h5'):
        """
        Args:
            save_dir: 保存目录
            save_best_k: 保存最好的k个模型
            monitor: 监控的指标
            mode: 'min' 或 'max'
            save_weights_only: 是否只保存权重
            save_format: 保存格式 'h5' 或 'tf'
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_best_k = save_best_k
        self.monitor = monitor
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.save_format = save_format
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'epochs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'best_models'), exist_ok=True)
        
        # 跟踪最佳模型
        self.best_models = []  # [(score, epoch, filepath), ...]
        
        # 训练历史记录
        self.epoch_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        """每个epoch结束时调用"""
        logs = logs or {}
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            print(f"Warning: {self.monitor} not found in logs")
            return
        
        # 保存当前epoch信息
        epoch_info = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            **logs
        }
        self.epoch_history.append(epoch_info)
        
        # 1. 保存每个epoch的模型
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
        
        # print(f"💾 Saved epoch model: {os.path.basename(epoch_filepath)}")
        
        # 2. 管理最佳模型
        self._manage_best_models(current_score, epoch + 1, epoch_filepath)
        
        # 3. 保存训练历史
        self._save_training_history()
        
        # 4. 打印epoch摘要
        self._print_epoch_summary(epoch + 1, logs)
    
    def _manage_best_models(self, current_score, epoch, epoch_filepath):
        """管理最佳模型保存"""
        # 判断是否比现有的最佳模型好
        is_better = self._is_better(current_score)
        
        if len(self.best_models) < self.save_best_k:
            # 如果还没有达到保存数量限制，直接添加
            best_filename = f"best_{len(self.best_models)+1:02d}_epoch_{epoch:03d}_score_{current_score:.4f}"
            if self.save_format == 'h5':
                best_filepath = os.path.join(self.save_dir, 'best_models', f"{best_filename}.h5")
                if self.save_weights_only:
                    self.model.save_weights(best_filepath)
                else:
                    self.model.save(best_filepath)
            else:
                best_filepath = os.path.join(self.save_dir, 'best_models', best_filename)
                self.model.save(best_filepath, save_format='tf')
            
            self.best_models.append((current_score, epoch, best_filepath))
            # print(f"🌟 Added to best models ({len(self.best_models)}/{self.save_best_k}): score={current_score:.4f}")
            
        elif is_better:
            # 找到最差的模型并替换
            worst_idx = self._get_worst_model_idx()
            worst_score, worst_epoch, worst_path = self.best_models[worst_idx]
            
            # 删除最差的模型文件
            try:
                if os.path.isdir(worst_path):
                    import shutil
                    shutil.rmtree(worst_path)
                else:
                    os.remove(worst_path)
                print(f"🗑️  Removed worse model: epoch_{worst_epoch}, score={worst_score:.4f}")
            except OSError as e:
                print(f"Warning: Could not remove {worst_path}: {e}")
            
            # 保存新的最佳模型
            best_filename = f"best_{worst_idx+1:02d}_epoch_{epoch:03d}_score_{current_score:.4f}"
            if self.save_format == 'h5':
                best_filepath = os.path.join(self.save_dir, 'best_models', f"{best_filename}.h5")
                if self.save_weights_only:
                    self.model.save_weights(best_filepath)
                else:
                    self.model.save(best_filepath)
            else:
                best_filepath = os.path.join(self.save_dir, 'best_models', best_filename)
                self.model.save(best_filepath, save_format='tf')
            
            self.best_models[worst_idx] = (current_score, epoch, best_filepath)
            print(f"🌟 Updated best models: score={current_score:.4f} (replaced score={worst_score:.4f})")
    
    def _is_better(self, current_score):
        """判断当前分数是否更好"""
        if not self.best_models:
            return True
        
        worst_score = self._get_worst_score()
        if self.mode == 'min':
            return current_score < worst_score
        else:
            return current_score > worst_score
    
    def _get_worst_model_idx(self):
        """获取最差模型的索引"""
        if self.mode == 'min':
            return max(range(len(self.best_models)), key=lambda i: self.best_models[i][0])
        else:
            return min(range(len(self.best_models)), key=lambda i: self.best_models[i][0])
    
    def _get_worst_score(self):
        """获取最差分数"""
        worst_idx = self._get_worst_model_idx()
        return self.best_models[worst_idx][0]
    
    def _save_training_history(self):
        """保存训练历史"""
        # 保存为JSON格式（便于阅读）
        json_path = os.path.join(self.save_dir, 'training_history.json')
        with open(json_path, 'w') as f:
            json.dump(self.epoch_history, f, indent=2)
        
        # 保存为YAML格式
        yaml_path = os.path.join(self.save_dir, 'training_history.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(self.epoch_history, f, default_flow_style=False)
        
        # 保存为pickle格式（便于程序读取）
        pkl_path = os.path.join(self.save_dir, 'training_history.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.epoch_history, f)
    
    def _print_epoch_summary(self, epoch, logs):
        """打印epoch摘要"""
        # print(f"\n📊 Epoch {epoch} Summary:")
        # print(f"   Loss: {logs.get('loss', 0):.4f} | Val Loss: {logs.get('val_loss', 0):.4f}")
        # print(f"   Acc:  {logs.get('accuracy', 0):.4f} | Val Acc:  {logs.get('val_accuracy', 0):.4f}")
        
        if self.best_models:
            best_scores = [score for score, _, _ in self.best_models]
            best_score = min(best_scores) if self.mode == 'min' else max(best_scores)
            # print(f"   Best {self.monitor}: {best_score:.4f}")
    
    def on_train_end(self, logs=None):
        """训练结束时调用"""
        print(f"\n🎯 Training completed!")
        print(f"📁 Models saved in: {self.save_dir}")
        print(f"🏆 Best {self.save_best_k} models saved in: {os.path.join(self.save_dir, 'best_models')}")
        
        # 打印最佳模型摘要
        if self.best_models:
            print(f"\n🌟 Best Models Summary:")
            sorted_models = sorted(self.best_models, key=lambda x: x[0], reverse=(self.mode == 'max'))
            for i, (score, epoch, filepath) in enumerate(sorted_models, 1):
                filename = os.path.basename(filepath)
                print(f"   {i}. Epoch {epoch}: {self.monitor}={score:.4f} -> {filename}")


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
