#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def calculate_bbox_pixel_average(image_path, x1, y1, x2, y2):
    """
    计算指定图片中bbox区域的像素平均值
    
    Args:
        image_path (str): 图片文件路径
        x1, y1 (int): bbox左上角坐标
        x2, y2 (int): bbox右下角坐标
    
    Returns:
        dict: 包含像素统计信息的字典
    """
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片尺寸
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
    
    # 确保bbox坐标在图片范围内
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    
    # 确保坐标顺序正确 (x1, y1为左上角，x2, y2为右下角)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    
    # 提取bbox区域（不包含边界）
    bbox_region = image[y1:y2, x1:x2]
    
    # 检查bbox是否框住有效范围
    if bbox_region.size == 0:
        raise ValueError(f"bbox区域无效：({x1},{y1})->({x2},{y2})，区域大小为0")
    
    # 计算实际的bbox尺寸
    actual_width = x2 - x1
    actual_height = y2 - y1
    
    if actual_width <= 0 or actual_height <= 0:
        raise ValueError(f"bbox尺寸无效：宽度={actual_width}, 高度={actual_height}")

    # 计算bbox区域的像素统计信息
    if len(bbox_region.shape) == 3:
        # 彩色图片
        result = {
            'bbox_coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'bbox_size': {'width': actual_width, 'height': actual_height},
            'image_info': {
                'path': image_path,
                'dimensions': {'width': width, 'height': height, 'channels': channels}
            },
            'pixel_stats': {}
        }
        
        # 计算每个通道的统计信息（包含0值）
        for channel in range(channels):
            channel_data = bbox_region[:, :, channel]
            result['pixel_stats'][f'channel_{channel}'] = {
                'mean': float(np.mean(channel_data)),
                'median': float(np.median(channel_data)),
                'std': float(np.std(channel_data)),
                'min': float(np.min(channel_data)),
                'max': float(np.max(channel_data))
            }
        
        # 计算整体平均值 (所有通道的平均值，包含0值)
        overall_mean = np.mean(bbox_region)
        result['pixel_stats']['overall'] = {
            'mean': float(overall_mean),
            'median': float(np.median(bbox_region)),
            'std': float(np.std(bbox_region)),
            'min': float(np.min(bbox_region)),
            'max': float(np.max(bbox_region))
        }
        
        # 如果是BGR图片，添加RGB格式的信息
        if channels == 3:
            result['pixel_stats']['bgr_mean'] = {
                'blue': result['pixel_stats']['channel_0']['mean'],
                'green': result['pixel_stats']['channel_1']['mean'],
                'red': result['pixel_stats']['channel_2']['mean']
            }
    
    else:
        # 灰度图片或单通道图片
        result = {
            'bbox_coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'bbox_size': {'width': actual_width, 'height': actual_height},
            'image_info': {
                'path': image_path,
                'dimensions': {'width': width, 'height': height, 'channels': 1}
            },
            'pixel_stats': {
                'mean': float(np.mean(bbox_region)),
                'median': float(np.median(bbox_region)),
                'std': float(np.std(bbox_region)),
                'min': float(np.min(bbox_region)),
                'max': float(np.max(bbox_region))
            }
        }
    
    return result

def main():
    # 直接在代码中设置参数，避免命令行输入
    # 修改下面的参数来测试不同的图片和bbox
    image_path = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD/image_depth_gray/video_0001/00503.png"  # 图片路径
    bbox = [
        186,
            652,
            337,
            996
        ]  # bbox坐标 [x1, y1, x2, y2]
    output_file = None  # 输出JSON文件路径，设为None则不保存
    verbose = True  # 是否显示详细信息
    
    # 解析bbox坐标
    x1, y1, x2, y2 = bbox
    
    try:
        # 计算bbox区域的像素平均值
        result = calculate_bbox_pixel_average(image_path, x1, y1, x2, y2)
        
        # 输出结果
        if verbose:
            print(f"图片路径: {result['image_info']['path']}")
            print(f"图片尺寸: {result['image_info']['dimensions']['width']}x{result['image_info']['dimensions']['height']}")
            print(f"通道数: {result['image_info']['dimensions']['channels']}")
            print(f"Bbox坐标: ({result['bbox_coordinates']['x1']}, {result['bbox_coordinates']['y1']}) -> ({result['bbox_coordinates']['x2']}, {result['bbox_coordinates']['y2']})")
            print(f"Bbox尺寸: {result['bbox_size']['width']}x{result['bbox_size']['height']}")
            print()
        
        if result['image_info']['dimensions']['channels'] == 1:
            # 单通道图片
            stats = result['pixel_stats']
            print(f"像素平均值: {stats['mean']:.2f}")
            if verbose:
                print(f"像素中位数: {stats['median']:.2f}")
                print(f"像素标准差: {stats['std']:.2f}")
                print(f"像素最小值: {stats['min']:.2f}")
                print(f"像素最大值: {stats['max']:.2f}")
        else:
            # 多通道图片
            print(f"整体像素平均值: {result['pixel_stats']['overall']['mean']:.2f}")
            if verbose:
                print(f"整体像素中位数: {result['pixel_stats']['overall']['median']:.2f}")
                print(f"整体像素标准差: {result['pixel_stats']['overall']['std']:.2f}")
                print(f"整体像素最小值: {result['pixel_stats']['overall']['min']:.2f}")
                print(f"整体像素最大值: {result['pixel_stats']['overall']['max']:.2f}")
                print()
                
                # 显示每个通道的信息
                for channel in range(result['image_info']['dimensions']['channels']):
                    channel_stats = result['pixel_stats'][f'channel_{channel}']
                    print(f"通道 {channel} 平均值: {channel_stats['mean']:.2f}")
                
                # 如果是BGR图片，显示RGB信息
                if 'bgr_mean' in result['pixel_stats']:
                    bgr = result['pixel_stats']['bgr_mean']
                    print(f"BGR平均值: B={bgr['blue']:.2f}, G={bgr['green']:.2f}, R={bgr['red']:.2f}")
        
        # 保存到JSON文件
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())