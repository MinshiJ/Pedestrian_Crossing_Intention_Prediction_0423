"""
专用测试脚本 - 只测试，不训练
- 支持传入模型目录（含 epochs 子目录）或单个 .h5 模型文件
- 强制使用生成器作为测试输入
- 从生成器中读取 y_true（labels[0]）计算分类指标
"""

import os
import sys
import glob
import yaml
import getopt
import shutil
import pandas as pd
from datetime import datetime
import numpy as np
import tensorflow as tf

from action_predict import action_prediction  # 使用我们在 action_predict.py 里注册的模型类
from jaad_data import JAAD
from pie_data import PIE
from watch_ped_data import WATCH_PED

# ========= GPU 设置 =========
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ========= 数据集路径（按你的机器实际路径修改）=========
path_jaad = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/JAAD"
path_pie = "/media/minshi/WD_2T/PIE/annotations"
path_watch_ped = "/home/minshi/Pedestrian_Crossing_Intention_Prediction/Watch_Ped"


# ========= 工具函数 =========
def load_config_from_path(path):
    """
    从路径加载配置文件
    参数:
        path: 模型目录或单个模型文件路径
    """
    if os.path.isfile(path):
        model_dir = os.path.dirname(path)
        if os.path.basename(model_dir) == 'epochs':
            model_dir = os.path.dirname(model_dir)
    else:
        model_dir = path

    config_path = os.path.join(model_dir, 'configs.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_files(path):
    """
    获取模型文件列表
    参数:
        path: 模型目录或单个模型文件路径
    返回:
        排序后的 .h5 文件列表
    """
    if os.path.isfile(path) and path.endswith('.h5'):
        return [path]

    model_files = []
    if os.path.isdir(path):
        model_files.extend(glob.glob(os.path.join(path, "*.h5")))
        epochs_dir = os.path.join(path, "epochs")
        if os.path.exists(epochs_dir):
            model_files.extend(glob.glob(os.path.join(epochs_dir, "*.h5")))
    return sorted(model_files)


# ========= 核心测试函数（强制生成器 + 从生成器取标签）=========
def test_single_model(model_file_path, configs, imdb, beh_seq_test):
    """
    测试单个模型文件（强制使用生成器）
    """
    print(f"\n🔍 测试模型: {os.path.basename(model_file_path)}")

    try:
        from tensorflow.keras.models import load_model
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

        # 1) 拿到模型类实例（主要是为了拿 get_data 的逻辑）
        Method = action_prediction(configs['model_opts']['model'])
        method_class = Method(
            dataset=configs['model_opts']['dataset'],
            sample_type=configs['data_opts']['sample_type'],
            **configs['net_opts']
        )

        # 2) 直接加载模型（自定义层在 action_predict 里已用 @register_keras_serializable 注册，通常可直加载）
        test_model = load_model(model_file_path)

        # 3) 生成测试数据：强制 generator 路径
        #    与训练时一致：generator=True, to_fit=False, shuffle=False
        test_data_dict = method_class.get_data(
            'test',
            beh_seq_test,
            {**configs['model_opts'], 'batch_size': 1}
        )
        x_test = test_data_dict['data'][0]   # -> tf.keras.utils.Sequence
        gen = x_test
        # 安全地从生成器中取出标签（意图标签在 labels[0]）
        labels = getattr(gen, "labels", None)
        if not (isinstance(labels, list) and len(labels) > 0 and labels[0] is not None):
            raise ValueError("测试生成器未包含有效的意图标签（labels[0] 为空）")
        y_true = np.asarray(labels[0]).astype(np.int32).reshape(-1)

        # 4) 预测
        test_results = test_model.predict(x_test, batch_size=1, verbose=1)
        # 兼容双输出/单输出
        if isinstance(test_results, (list, tuple)) and len(test_results) >= 1:
            intention_results = test_results[0]
            etraj_pred = test_results[1] if len(test_results) > 1 else None
        else:
            intention_results = test_results
            etraj_pred = None

        # 5) 计算指标（注意 AUC 用概率，不要先 round）
        y_prob = np.asarray(intention_results).reshape(-1)
        y_pred = np.round(y_prob)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        return {
            'model_file': os.path.basename(model_file_path),
            'model_path': model_file_path,
            'accuracy': float(acc),
            'auc': float(auc) if not np.isnan(auc) else 0.0,
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
    """
    测试模型路径中的模型（目录或单个文件）
    """
    if os.path.isfile(path):
        print(f"🚀 开始测试单个模型: {os.path.basename(path)}")
    else:
        print(f"🚀 开始测试模型目录: {path}")

    # 保存目录：若是 epochs 里的文件，回到上级目录；否则用目录自身
    if os.path.isfile(path):
        save_dir = os.path.dirname(path)
        if os.path.basename(save_dir) == 'epochs':
            save_dir = os.path.dirname(save_dir)
    else:
        save_dir = path

    # 1) 加载配置
    configs = load_config_from_path(path)
    print("✅ 配置文件加载成功")

    # 2) 初始化数据集
    ds = configs['model_opts']['dataset']
    if ds == 'pie':
        imdb = PIE(data_path=path_pie)
    elif ds == 'jaad':
        imdb = JAAD(data_path=path_jaad)
    elif ds in ("watch_ped", "watch"):
        imdb = WATCH_PED(data_path=path_watch_ped)
    else:
        raise ValueError(f"不支持的数据集: {ds}")
    print("✅ 数据集初始化成功")

    # 3) 生成测试数据（与训练同样的接口）
    print("🔄 生成测试数据...")
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
    print("✅ 测试数据生成完成")

    # 4) 找出所有 .h5
    model_files = get_model_files(path)
    print(f"📁 找到 {len(model_files)} 个模型文件")
    if not model_files:
        print("❌ 未找到模型文件")
        return

    # 5) 逐个测试
    results = []
    for i, model_path in enumerate(model_files, 1):
        print("\n" + "=" * 60)
        print("测试单个模型" if len(model_files) == 1 else f"进度: {i}/{len(model_files)}")
        result = test_single_model(model_path, configs, imdb, beh_seq_test)
        results.append(result)
        if result['status'] == 'success':
            print(f"✅ 准确率: {result['accuracy']:.4f}, AUC: {result['auc']:.4f}, F1: {result['f1']:.4f}")
        else:
            print(f"❌ {result['status']}")

    # 6) 保存结果、清理、重命名、汇总
    save_results(save_dir, results, configs)
    cleanup_models(save_dir, results)
    final_dir = rename_model_directory(save_dir, results)
    display_summary(results)

    return final_dir


# ========= 结果持久化 & 整理 =========
def save_results(model_dir, results, configs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame(results)
    csv_path = os.path.join(model_dir, f'test_results_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📊 结果已保存到: {csv_path}")

    report_path = os.path.join(model_dir, f'test_report_{timestamp}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型测试报告 (Model Test Report)\n")
        f.write("=" * 80 + "\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型目录: {model_dir}\n")
        f.write(f"数据集: {configs['model_opts']['dataset']}\n")
        f.write(f"模型类型: {configs['model_opts']['model']}\n")
        f.write(f"样本类型: {configs['data_opts']['sample_type']}\n\n")

        f.write("测试结果汇总:\n")
        f.write("-" * 80 + "\n")
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

            best_acc_idx = df_success['accuracy'].idxmax()
            best_auc_idx = df_success['auc'].idxmax()
            best_f1_idx = df_success['f1'].idxmax()

            f.write("最佳模型:\n")
            f.write(f"最高准确率模型: {success_results[best_acc_idx]['model_file']} (Acc: {success_results[best_acc_idx]['accuracy']:.4f})\n")
            f.write(f"最高AUC模型: {success_results[best_auc_idx]['model_file']} (AUC: {success_results[best_auc_idx]['auc']:.4f})\n")
            f.write(f"最高F1模型: {success_results[best_f1_idx]['model_file']} (F1: {success_results[best_f1_idx]['f1']:.4f})\n\n")

        f.write("详细结果:\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"模型: {r['model_file']}\n")
            if r['status'] == 'success':
                f.write(f"  准确率: {r['accuracy']:.4f}\n")
                f.write(f"  AUC: {r['auc']:.4f}\n")
                f.write(f"  F1: {r['f1']:.4f}\n")
                f.write(f"  精确率: {r['precision']:.4f}\n")
                f.write(f"  召回率: {r['recall']:.4f}\n")
            else:
                f.write(f"  状态: {r['status']}\n")
            f.write("\n")
    print(f"📝 报告已保存到: {report_path}")


def cleanup_models(model_dir, results):
    """
    清理模型文件：
    - 若有成功测试的结果：保留准确率最高的那个（若位于 epochs 目录，则复制一份到上级目录）
    - 删除 epochs 目录中的所有 .h5
    """
    success_results = [r for r in results if r['status'] == 'success']
    if not success_results:
        print("❌ 没有成功测试的模型，跳过清理步骤")
        return

    df_success = pd.DataFrame(success_results)
    best_acc_idx = df_success['accuracy'].idxmax()
    best_model = success_results[best_acc_idx]
    best_model_path = best_model['model_path']
    best_model_file = best_model['model_file']
    print(f"\n🏆 准确率最高的模型: {best_model_file} (准确率: {best_model['accuracy']:.4f})")

    epochs_dir = os.path.join(model_dir, "epochs")
    if not os.path.exists(epochs_dir):
        print("📁 epochs目录不存在，无需清理")
        return

    epoch_files = glob.glob(os.path.join(epochs_dir, "*.h5"))
    if not epoch_files:
        print("📁 epochs目录中没有模型文件，无需清理")
        return

    print(f"🗑️  开始清理epochs目录，将删除 {len(epoch_files)} 个模型文件...")
    deleted = 0
    for mf in epoch_files:
        try:
            if mf == best_model_path:
                target_path = os.path.join(model_dir, os.path.basename(mf))
                if not os.path.exists(target_path):
                    shutil.copy2(mf, target_path)
                    print(f"📋 已将最佳模型复制到: {target_path}")
            os.remove(mf)
            deleted += 1
            print(f"🗑️  已删除: {os.path.basename(mf)}")
        except Exception as e:
            print(f"❌ 删除失败: {os.path.basename(mf)} - {str(e)}")

    if not os.listdir(epochs_dir):
        try:
            os.rmdir(epochs_dir)
            print("🗑️  已删除空的epochs目录")
        except Exception as e:
            print(f"⚠️  删除epochs目录失败: {str(e)}")


def rename_model_directory(model_dir, results):
    """
    按最高准确率为目录添加后缀 _acc_xxxx
    """
    success_results = [r for r in results if r['status'] == 'success']
    if not success_results:
        print("❌ 没有成功测试的模型，跳过重命名步骤")
        return model_dir

    df_success = pd.DataFrame(success_results)
    best_accuracy = df_success['accuracy'].max()

    current_dir_name = os.path.basename(model_dir)
    parent_dir = os.path.dirname(model_dir)
    suffix = f"_acc_{best_accuracy:.4f}"

    if "_acc_" in current_dir_name:
        base = current_dir_name.split("_acc_")[0]
        new_dir_name = f"{base}{suffix}"
    else:
        new_dir_name = f"{current_dir_name}{suffix}"

    new_model_dir = os.path.join(parent_dir, new_dir_name)
    if new_model_dir == model_dir:
        print("📁 目录名已包含准确率信息，无需重命名")
        return model_dir

    try:
        if os.path.exists(new_model_dir):
            print(f"⚠️  目标目录已存在: {new_dir_name}")
            ts = datetime.now().strftime("%H%M%S")
            new_dir_alt = os.path.join(parent_dir, f"{new_dir_name}_{ts}")
            os.rename(model_dir, new_dir_alt)
            print(f"🔄 已重命名为: {os.path.basename(new_dir_alt)}")
            return new_dir_alt
        else:
            os.rename(model_dir, new_model_dir)
            print("🔄 模型目录已重命名:")
            print(f"   原目录: {current_dir_name}")
            print(f"   新目录: {new_dir_name}")
            return new_model_dir
    except Exception as e:
        print(f"❌ 重命名目录失败: {str(e)}")
        return model_dir


def display_summary(results):
    success_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] != 'success']

    print("\n" + "=" * 80)
    print("🎯 测试结果汇总")
    print("=" * 80)
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
    print('专用模型测试脚本 - 支持测试单个模型或整个目录')
    print('用法: python compare_all_epochs.py [选项]')
    print('选项:')
    print('-h, --help\t\t', '显示帮助信息')
    print('-d, --model_dir\t', '模型目录路径')
    print('-f, --model_file\t', '单个模型文件路径')
    print()
    print('示例:')
    print('python compare_all_epochs.py -d /path/to/model/directory')
    print('python compare_all_epochs.py -f /path/to/model.h5')
    print()


# ========= CLI =========
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
        print(f'\x1b[1;37;41mERROR: 路径不存在: {model_path}\x1b[0m')
        sys.exit(2)

    test_model_path(model_path)
