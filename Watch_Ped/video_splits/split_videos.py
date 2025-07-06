#!/usr/bin/env python3
"""
脚本用于将video_0001到video_0254分为训练集和测试集
"""

import os
import random
from sklearn.model_selection import train_test_split
import pickle

def create_video_split(start_video=1, end_video=254, train_ratio=0.8, random_seed=42):
    """
    创建视频数据分割
    
    Args:
        start_video: 起始视频编号
        end_video: 结束视频编号
        train_ratio: 训练集比例
        random_seed: 随机种子
    """
    # 生成视频ID列表
    video_ids = []
    for i in range(start_video, end_video + 1):
        video_id = f"video_{i:04d}"
        video_ids.append(video_id)
    
    print(f"总共有 {len(video_ids)} 个视频")
    print(f"视频范围: {video_ids[0]} 到 {video_ids[-1]}")
    
    # 设置随机种子确保结果可重现
    random.seed(random_seed)
    
    # 分割数据
    train_videos, test_videos = train_test_split(
        video_ids, 
        train_size=train_ratio, 
        random_state=random_seed,
        shuffle=True
    )
    
    print(f"\n训练集: {len(train_videos)} 个视频 ({len(train_videos)/len(video_ids)*100:.1f}%)")
    print(f"测试集: {len(test_videos)} 个视频 ({len(test_videos)/len(video_ids)*100:.1f}%)")
    
    return train_videos, test_videos

def save_split_files(train_videos, test_videos, output_dir="Watch_Ped/video_splits"):
    """
    保存分割结果到文件
    
    Args:
        train_videos: 训练集视频列表
        test_videos: 测试集视频列表
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_file = os.path.join(output_dir, "train.txt")
    with open(train_file, 'w') as f:
        for video in sorted(train_videos):
            f.write(f"{video}\n")
    
    # 保存测试集
    test_file = os.path.join(output_dir, "test.txt")
    with open(test_file, 'w') as f:
        for video in sorted(test_videos):
            f.write(f"{video}\n")
    
    print(f"\n分割结果已保存到:")
    print(f"- 训练集: {train_file}")
    print(f"- 测试集: {test_file}")

    return train_file, test_file


def main():
    """
    主函数
    """
    print("视频数据分割工具")
    print("="*50)
    
    # 创建视频分割
    train_videos, test_videos = create_video_split(
        start_video=1,
        end_video=254,
        train_ratio=0.8,  # 80% 训练，20% 测试
        random_seed=42
    )
    
    
    # 保存分割文件
    save_split_files(train_videos, test_videos)
    
    
    print(f"\n✅ 视频分割完成！")


if __name__ == "__main__":
    main()