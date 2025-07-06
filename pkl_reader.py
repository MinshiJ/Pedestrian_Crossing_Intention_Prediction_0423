import pickle
import json
from pprint import pprint

def visualize_pkl_content(pkl_path, max_depth=6, max_items=6):
    """
    简单的pkl内容查看器
    """
    print(f"正在读取PKL文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("="*60)
    print("PKL文件内容概览")
    print("="*60)
    
    # # 添加深度信息搜索功能
    # depth_keywords = ['depth', 'depth_gray', 'depth_rgb', 'image_depth', 'gray', 'rgb']
    # found_depth_keys = []
    
    # def search_depth_keys(obj, path=""):
    #     if isinstance(obj, dict):
    #         for k, v in obj.items():
    #             current_path = f"{path}.{k}" if path else k
    #             # 检查键名是否包含深度相关关键词
    #             if any(keyword in str(k).lower() for keyword in depth_keywords):
    #                 found_depth_keys.append(current_path)
    #             search_depth_keys(v, current_path)
    #     elif isinstance(obj, list) and obj:
    #         for i, item in enumerate(obj[:3]):  # 只检查前3个元素
    #             search_depth_keys(item, f"{path}[{i}]")
    
    def show_structure(obj, name="root", depth=0, max_depth=6):
        indent = "  " * depth
        
        if depth > max_depth:
            print(f"{indent}{name}: <深度限制，跳过显示>")
            return
            
        if isinstance(obj, dict):
            print(f"{indent}{name}: dict ({len(obj)} items)")
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_items:
                    print(f"{indent}  ... 还有 {len(obj) - max_items} 个项目")
                    break
                show_structure(v, f"[{k}]", depth+1, max_depth)
                
        elif isinstance(obj, list):
            print(f"{indent}{name}: list ({len(obj)} items)")
            for i, item in enumerate(obj[:max_items]):
                show_structure(item, f"[{i}]", depth+1, max_depth)
            if len(obj) > max_items:
                print(f"{indent}  ... 还有 {len(obj) - max_items} 个项目")
                
        elif isinstance(obj, (str, int, float, bool)):
            print(f"{indent}{name}: {type(obj).__name__} = {obj}")
        else:
            print(f"{indent}{name}: {type(obj).__name__}")
    
    # 先搜索深度相关键
    # search_depth_keys(data)
    
    show_structure(data)
    
    # # 显示搜索结果
    # print("\n" + "="*60)
    # print("深度信息搜索结果:")
    # print("="*60)
    # if found_depth_keys:
    #     print("找到以下可能包含深度信息的键:")
    #     for key in found_depth_keys:
    #         print(f"  - {key}")
    # else:
    #     print("未找到明显的深度相关键名")

# 使用示例 - 检查Watch_Ped缓存文件
pkl_path = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/Watch_Ped/Watch_Ped_cache.pkl"
visualize_pkl_content(pkl_path)