#!/usr/bin/env python3
"""
简化版训练和测试管道脚本
保持原有的训练输出格式，训练完成后自动运行测试
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

def find_latest_model_dir(base_path="data/models"):
    """找到最新创建的模型目录"""
    if not os.path.exists(base_path):
        return None
    
    model_dirs = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            try:
                dir_files = os.listdir(full_path)
                if any(f.endswith('.h5') for f in dir_files) or 'configs.yaml' in dir_files:
                    model_dirs.append(full_path)
            except (OSError, PermissionError):
                continue
    
    if not model_dirs:
        return None
    
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return model_dirs[0]

def main():
    parser = argparse.ArgumentParser(description='简化版训练和测试管道')
    parser.add_argument('-c', '--config', required=True, help='训练配置文件路径')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 训练和测试管道启动")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 运行训练
    print(f"\n🚀 开始训练...")
    train_cmd = [sys.executable, "train_test.py", "-c", args.config]
    
    start_time = time.time()
    train_result = subprocess.run(train_cmd)
    train_end_time = time.time()
    
    if train_result.returncode != 0:
        print(f"\n❌ 训练失败，退出码: {train_result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ 训练完成 (耗时: {(train_end_time - start_time) / 60:.1f} 分钟)")
    
    # 2. 查找模型目录
    print("🔍 查找最新模型目录...")
    model_dir = find_latest_model_dir()
    
    if not model_dir:
        print("❌ 未找到模型目录")
        sys.exit(1)
    
    print(f"📁 找到模型目录: {model_dir}")
    
    # 3. 运行测试
    print(f"\n🧪 开始测试模型...")
    test_cmd = [sys.executable, "compare_all_epochs.py", "-d", model_dir]
    
    test_start_time = time.time()
    test_result = subprocess.run(test_cmd)
    test_end_time = time.time()
    
    if test_result.returncode != 0:
        print(f"\n❌ 测试失败，退出码: {test_result.returncode}")
        sys.exit(1)
    
    print(f"\n✅ 测试完成 (耗时: {(test_end_time - test_start_time) / 60:.1f} 分钟)")
    
    # 4. 完成
    total_time = (test_end_time - start_time) / 60
    print("\n" + "=" * 80)
    print("🎉 训练和测试管道完成!")
    print("=" * 80)
    print(f"模型目录: {model_dir}")
    print(f"总耗时: {total_time:.1f} 分钟")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
