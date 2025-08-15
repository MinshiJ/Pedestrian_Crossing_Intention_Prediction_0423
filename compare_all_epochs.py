"""
专用测试脚本 - 只测试，不训练
用于测试文件夹中的模型文件
"""
import os
import yaml
import getopt
import sys
import glob
import pandas as pd
from datetime import datetime

import numpy as np
from tensorflow.keras import backend as K

from action_predict import action_prediction
from jaad_data import JAAD
from pie_data import PIE
from watch_ped_data import WATCH_PED

import tensorflow as tf

# 设置GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 数据集路径
path_jaad = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD"
path_pie = "/media/minshi/WD_2T/PIE/annotations"
path_watch_ped = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/Watch_Ped"


def load_config_from_path(path):
    """从路径加载配置文件
    参数:
    - path: 可以是模型目录或单个模型文件路径
    """
    if os.path.isfile(path):
        # 如果是文件，找到父目录
        model_dir = os.path.dirname(path)
        # 如果在epochs子目录中，需要向上一级找配置文件
        if os.path.basename(model_dir) == 'epochs':
            model_dir = os.path.dirname(model_dir)
    else:
        # 如果是目录
        model_dir = path
    
    config_path = os.path.join(model_dir, 'configs.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"配置文件未找到: {config_path}")


def get_model_files(path):
    """获取模型文件列表
    参数:
    - path: 可以是模型目录或单个模型文件路径
    """
    model_files = []
    
    # 如果是单个文件
    if os.path.isfile(path) and path.endswith('.h5'):
        return [path]
    
    # 如果是目录
    if os.path.isdir(path):
        # 查找 .h5 文件
        h5_files = glob.glob(os.path.join(path, "*.h5"))
        for f in h5_files:
            model_files.append(f)
        
        # 查找 epochs 子目录中的文件
        epochs_dir = os.path.join(path, "epochs")
        if os.path.exists(epochs_dir):
            epoch_files = glob.glob(os.path.join(epochs_dir, "*.h5"))
            model_files.extend(epoch_files)
    
    return sorted(model_files)


def test_single_model(model_file_path, configs, imdb, beh_seq_test):
    """测试单个模型文件"""
    print(f"\n🔍 测试模型: {os.path.basename(model_file_path)}")
    
    try:
        from tensorflow.keras.models import load_model
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
        
        # 获取模型类
        method_class = action_prediction(configs['model_opts']['model'])(
            dataset=configs['model_opts']['dataset'], 
            sample_type=configs['data_opts']['sample_type'], 
            **configs['net_opts']
        )
        
        # 直接加载模型文件
        test_model = load_model(model_file_path)
        
        # 准备测试数据
        test_data = method_class.get_data('test', beh_seq_test, {**configs['model_opts'], 'batch_size': 1})
        
        # 进行预测
        test_results = test_model.predict(test_data['data'][0], batch_size=1, verbose=1)
        
        # 计算指标
        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        
        return {
            'model_file': os.path.basename(model_file_path),
            'model_path': model_file_path,
            'accuracy': float(acc),
            'auc': float(auc),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return {
            'model_file': os.path.basename(model_file_path),
            'model_path': model_file_path,
            'accuracy': 0.0,
            'auc': 0.0,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'status': f'error: {str(e)}'
        }


def test_model_path(path):
    """测试模型路径中的模型
    参数:
    - path: 可以是模型目录或单个模型文件路径
    """
    if os.path.isfile(path):
        print(f"🚀 开始测试单个模型: {os.path.basename(path)}")
    else:
        print(f"🚀 开始测试模型目录: {path}")
    
    # 加载配置
    configs = load_config_from_path(path)
    print("✅ 配置文件加载成功")
    
    # 初始化数据集
    if configs['model_opts']['dataset'] == 'pie':
        imdb = PIE(data_path=path_pie)
    elif configs['model_opts']['dataset'] == 'jaad':
        imdb = JAAD(data_path=path_jaad)
    elif configs['model_opts']['dataset'] in ["watch_ped", "watch"]:
        imdb = WATCH_PED(data_path=path_watch_ped)
    else:
        raise ValueError(f"不支持的数据集: {configs['model_opts']['dataset']}")
    
    print("✅ 数据集初始化成功")
    
    # 生成测试数据（与训练时完全相同的预处理步骤）
    print("🔄 生成测试数据...")
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
    print("✅ 测试数据生成完成")
    
    # 获取所有模型文件
    model_files = get_model_files(path)
    print(f"📁 找到 {len(model_files)} 个模型文件")
    
    if not model_files:
        print("❌ 未找到模型文件")
        return
    
    # 测试结果列表
    results = []
    
    # 逐个测试模型
    for i, model_path in enumerate(model_files, 1):
        print(f"\n{'='*60}")
        if len(model_files) == 1:
            print(f"测试单个模型")
        else:
            print(f"进度: {i}/{len(model_files)}")
        
        result = test_single_model(model_path, configs, imdb, beh_seq_test)
        results.append(result)
        
        if result['status'] == 'success':
            print(f"✅ 准确率: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}, F1: {result['f1']:.4f}")
        else:
            print(f"❌ {result['status']}")
    
    # 确定保存目录
    if os.path.isfile(path):
        save_dir = os.path.dirname(path)
        if os.path.basename(save_dir) == 'epochs':
            save_dir = os.path.dirname(save_dir)
    else:
        save_dir = path
    
    # 保存结果
    save_results(save_dir, results, configs)
    
    # 显示汇总
    display_summary(results)


def save_results(model_dir, results, configs):
    """保存测试结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存为CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(model_dir, f'test_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 结果已保存到: {csv_path}")
    
    # 保存详细报告
    report_path = os.path.join(model_dir, f'test_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("模型测试报告 (Model Test Report)\n")
        f.write("="*80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型目录: {model_dir}\n")
        f.write(f"数据集: {configs['model_opts']['dataset']}\n")
        f.write(f"模型类型: {configs['model_opts']['model']}\n")
        f.write(f"样本类型: {configs['data_opts']['sample_type']}\n\n")
        
        f.write("测试结果汇总:\n")
        f.write("-"*80 + "\n")
        
        success_results = [r for r in results if r['status'] == 'success']
        if success_results:
            df_success = pd.DataFrame(success_results)
            f.write(f"成功测试模型数量: {len(success_results)}\n")
            f.write(f"平均准确率: {df_success['accuracy'].mean():.4f}\n")
            f.write(f"最高准确率: {df_success['accuracy'].max():.4f}\n")
            f.write(f"平均AUC: {df_success['auc'].mean():.4f}\n")
            f.write(f"最高AUC: {df_success['auc'].max():.4f}\n")
            f.write(f"平均F1: {df_success['f1'].mean():.4f}\n")
            f.write(f"最高F1: {df_success['f1'].max():.4f}\n\n")
            
            # 找出最佳模型
            best_acc_idx = df_success['accuracy'].idxmax()
            best_auc_idx = df_success['auc'].idxmax()
            best_f1_idx = df_success['f1'].idxmax()
            
            f.write("最佳模型:\n")
            f.write(f"最高准确率模型: {success_results[best_acc_idx]['model_file']} (Acc: {success_results[best_acc_idx]['accuracy']:.4f})\n")
            f.write(f"最高AUC模型: {success_results[best_auc_idx]['model_file']} (AUC: {success_results[best_auc_idx]['auc']:.4f})\n")
            f.write(f"最高F1模型: {success_results[best_f1_idx]['model_file']} (F1: {success_results[best_f1_idx]['f1']:.4f})\n\n")
        
        f.write("详细结果:\n")
        f.write("-"*80 + "\n")
        for result in results:
            f.write(f"模型: {result['model_file']}\n")
            if result['status'] == 'success':
                f.write(f"  准确率: {result['accuracy']:.4f}\n")
                f.write(f"  AUC: {result['auc']:.4f}\n")
                f.write(f"  F1: {result['f1']:.4f}\n")
                f.write(f"  精确率: {result['precision']:.4f}\n")
                f.write(f"  召回率: {result['recall']:.4f}\n")
            else:
                f.write(f"  状态: {result['status']}\n")
            f.write("\n")
    
    print(f"📝 报告已保存到: {report_path}")


def display_summary(results):
    """显示测试结果汇总"""
    success_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']
    
    print("\n" + "="*80)
    print("🎯 测试结果汇总")
    print("="*80)
    print(f"总模型数量: {len(results)}")
    print(f"成功测试: {len(success_results)}")
    print(f"失败测试: {len(failed_results)}")
    
    if success_results:
        df = pd.DataFrame(success_results)
        print(f"\n📊 性能统计:")
        print(f"平均准确率: {df['accuracy'].mean():.4f} (±{df['accuracy'].std():.4f})")
        print(f"平均AUC: {df['auc'].mean():.4f} (±{df['auc'].std():.4f})")
        print(f"平均F1: {df['f1'].mean():.4f} (±{df['f1'].std():.4f})")
        
        print(f"\n🏆 最佳模型:")
        best_acc = df.loc[df['accuracy'].idxmax()]
        best_auc = df.loc[df['auc'].idxmax()]
        best_f1 = df.loc[df['f1'].idxmax()]
        
        print(f"最高准确率: {best_acc['model_file']} (Acc: {best_acc['accuracy']:.4f})")
        print(f"最高AUC: {best_auc['model_file']} (AUC: {best_auc['auc']:.4f})")
        print(f"最高F1: {best_f1['model_file']} (F1: {best_f1['f1']:.4f})")


def usage():
    """显示帮助信息"""
    print('专用模型测试脚本 - 支持测试单个模型或整个目录')
    print('用法: python compare_all_epochs.py [选项]')
    print('选项:')
    print('-h, --help\t\t', '显示帮助信息')
    print('-d, --model_dir\t', '模型目录路径')
    print('-f, --model_file\t', '单个模型文件路径')
    print()
    print('示例:')
    print('# 测试整个目录')
    print('python compare_all_epochs.py -d /path/to/model/directory')
    print('# 测试单个模型')
    print('python compare_all_epochs.py -f /path/to/model.h5')
    print()


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hd:f:', ['help', 'model_dir=', 'model_file='])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)

    model_path = None

    for o, a in opts:
        if o in ["-h", "--help"]:
            usage()
            sys.exit(0)
        elif o in ['-d', '--model_dir']:
            model_path = a
        elif o in ['-f', '--model_file']:
            model_path = a

    if not model_path:
        print('\x1b[1;37;41m' + 'ERROR: 请提供模型目录路径(-d)或模型文件路径(-f)!' + '\x1b[0m')
        usage()
        sys.exit(2)

    if not os.path.exists(model_path):
        print(f'\x1b[1;37;41m' + f'ERROR: 路径不存在: {model_path}' + '\x1b[0m')
        sys.exit(2)

    test_model_path(model_path)
