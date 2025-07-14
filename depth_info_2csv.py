import pickle
import pandas as pd
import csv
from pathlib import Path

def extract_depth_info_to_csv(pkl_path, output_csv_path):
    """
    从PKL文件中提取depth_info数据并保存为CSV文件
    """
    print(f"正在读取PKL文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 准备数据列表
    depth_data = []
    
    # 遍历所有视频
    for video_id, video_data in data.items():
        if 'image_info' in video_data:
            # 遍历每个视频的图像信息
            for frame_id, frame_data in video_data['image_info'].items():
                if 'depth_info' in frame_data:
                    depth_data.append({
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'depth_info': frame_data['depth_info'],
                        'crossing': video_data.get('crossing', None),
                        'crossing_point': video_data.get('crossing_point', None),
                        'vehicle_speed': frame_data.get('vehicle_speed', None)
                    })
    
    # 创建DataFrame
    df = pd.DataFrame(depth_data)
    
    # 保存为CSV文件
    df.to_csv(output_csv_path, index=False)
    
    print(f"已提取 {len(depth_data)} 条depth_info记录")
    print(f"数据已保存到: {output_csv_path}")
    print(f"数据概览:")
    print(df.head())
    print(f"\n统计信息:")
    print(df.describe())
    
    return df

def extract_depth_info_to_excel(pkl_path, output_excel_path):
    """
    从PKL文件中提取depth_info数据并保存为Excel文件
    """
    print(f"正在读取PKL文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 准备数据字典，按视频分组
    video_sheets = {}
    
    # 遍历所有视频
    for video_id, video_data in data.items():
        if 'image_info' in video_data:
            depth_data = []
            # 遍历每个视频的图像信息
            for frame_id, frame_data in video_data['image_info'].items():
                if 'depth_info' in frame_data:
                    depth_data.append({
                        'frame_id': frame_id,
                        'depth_info': frame_data['depth_info'],
                        'vehicle_speed': frame_data.get('vehicle_speed', None),
                        'bbox_x1': frame_data.get('bbox_ped', {}).get('bbox_x1', None),
                        'bbox_y1': frame_data.get('bbox_ped', {}).get('bbox_y1', None),
                        'bbox_x2': frame_data.get('bbox_ped', {}).get('bbox_x2', None),
                        'bbox_y2': frame_data.get('bbox_ped', {}).get('bbox_y2', None),
                        'acc_x': frame_data.get('acc_info', {}).get('acc_x', None),
                        'acc_y': frame_data.get('acc_info', {}).get('acc_y', None),
                        'acc_z': frame_data.get('acc_info', {}).get('acc_z', None)
                    })
            
            if depth_data:
                video_sheets[video_id] = pd.DataFrame(depth_data)
    
    # 保存为Excel文件，每个视频一个sheet
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # 创建汇总sheet
        all_data = []
        for video_id, df in video_sheets.items():
            df_copy = df.copy()
            df_copy['video_id'] = video_id
            all_data.append(df_copy)
        
        if all_data:
            summary_df = pd.concat(all_data, ignore_index=True)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 为每个视频创建单独的sheet
            for video_id, df in video_sheets.items():
                # Excel sheet名称不能超过31字符
                sheet_name = video_id[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    total_records = sum(len(df) for df in video_sheets.values())
    print(f"已提取 {total_records} 条depth_info记录")
    print(f"数据已保存到: {output_excel_path}")
    print(f"包含 {len(video_sheets)} 个视频的数据")

if __name__ == "__main__":
    # 输入和输出文件路径
    pkl_path = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/Watch_Ped/Watch_Ped_cache.pkl"
    output_csv_path = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/depth_info_data.csv"
    output_excel_path = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/depth_info_data.xlsx"
    
    # 提取到CSV
    print("="*60)
    print("提取depth_info数据到CSV文件")
    print("="*60)
    df_csv = extract_depth_info_to_csv(pkl_path, output_csv_path)
    
    # 提取到Excel（包含更多信息）
    print("\n" + "="*60)
    print("提取depth_info数据到Excel文件")
    print("="*60)
    extract_depth_info_to_excel(pkl_path, output_excel_path)