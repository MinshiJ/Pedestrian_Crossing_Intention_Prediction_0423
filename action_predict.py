# -*- coding: utf-8 -*-
import os
import time
import tarfile
import pickle
import yaml
import json
import wget
import cv2
import scipy.misc
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

from utils import *
from custom_callbacks import EpochSaveCallback, DetailedLoggingCallback, MetricsVisualizationCallback
from base_models import AlexNet, C3DNet, convert_to_fcn, C3DNet2
from base_models import I3DNet

from tensorflow.keras.layers import (Input, Concatenate, Dense, GRU, LSTM, GRUCell,
                                     Dropout, LSTMCell, RNN, Flatten, Average, Add,
                                     ConvLSTM2D, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Lambda, dot, concatenate, Activation)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model, Sequence, register_keras_serializable
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.activations import gelu

try:
    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
except ImportError:
    from tensorflow.keras.layers.experimental import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import BinaryCrossentropy


# ================================
# Base Class
# ================================
class ActionPredict(object):
    """
    A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='vgg16',
                 **kwargs):
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = True  # 仅用生成器

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset (默认实现，子类可覆盖)
        """
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw['activities'].copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()

        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])

        d['box_org'] = d['box'].copy()
        d['tte'] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
        else:
            overlap = opts['overlap']
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            for k in d.keys():
                seqs = []
                for seq in d[k]:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    seqs.extend([seq[i:i + obs_length] for i in
                                 range(start_idx, end_idx + 1, olap_res)])
                d[k] = seqs

            for seq in data_raw['bbox']:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                 range(start_idx, end_idx + 1, olap_res)])

        if normalize:
            for k in d.keys():
                if k != 'tte':
                    if k != 'box' and k != 'center':
                        for i in range(len(d[k])):
                            d[k][i] = d[k][i][1:]
                    else:
                        for i in range(len(d[k])):
                            d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                d[k] = np.array(d[k])
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        d['crossing'] = np.array(d['crossing'])[:, 0, :]
        pos_count = np.count_nonzero(d['crossing'])
        neg_count = len(d['crossing']) - pos_count

        return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        基于 crossing 标签做简单平衡（翻转+欠采样）
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            gt_augment = 1 if num_neg_samples > num_pos_samples else 0
            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]] for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            dif_samples = abs(num_neg_samples - num_pos_samples)
            np.random.seed(42)
            np.random.shuffle(rm_index)
            rm_index = rm_index[0:dif_samples]

            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_data(self, data_type, data_raw, model_opts):
        """
        统一只用 DataGenerator。无 numpy 返回分支。
        """
        self._generator = True  # 强制只用生成器
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict = {}
        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        _data, data_sizes, data_types = [], [], []
        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'mask' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features = get_pose(data['image'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        gen = DataGenerator(
            data=_data,
            labels=[data['crossing'], data.get('trajectory', None)],  # 第二输出可为 None
            data_sizes=data_sizes,
            process=process,
            global_pooling=self._global_pooling,
            input_type_list=model_opts['obs_input_type'],
            batch_size=model_opts['batch_size'],
            shuffle=data_type != 'test',
            to_fit=data_type != 'test'   # test 阶段返回 X；train/val 返回 (X,y,sw)
        )
        packed = (gen, None)

        return {'data': packed,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def log_configs(self, config_path, batch_size, epochs, lr, model_opts):
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts,
                       'train_opts': {'batch_size': batch_size, 'epochs': epochs, 'lr': lr}},
                      fid, default_flow_style=False)
        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count):
        """
        其它模型可用；双输出训练不再用 class_weight=，改用 DataGenerator.sample_weight。
        """
        if not apply_weights:
            return None
        total = sample_count['neg_count'] + sample_count['pos_count']
        neg_weight = sample_count['pos_count'] / total
        pos_weight = sample_count['neg_count'] / total
        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        callbacks = []
        if learning_scheduler and 'early_stop' in learning_scheduler:
            default_params = {'monitor': 'val_loss', 'restore_best_weights': True,
                              'min_delta': 1.0, 'patience': 5, 'verbose': 1}
            default_params.update(learning_scheduler['early_stop'])
            callbacks.append(EarlyStopping(**default_params))

        # plateau（可选）
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', mode='min', factor=0.2, patience=5,
            cooldown=1, min_lr=1e-7, verbose=1)
        callbacks.append(reduce_lr)

        # checkpoint（如需）
        # callbacks.append(ModelCheckpoint(...))

        return callbacks

    def get_optimizer(self, optimizer):
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], f"{optimizer} not implemented"
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop

    def train(self, data_train,
              data_val,
              batch_size=2,
              epochs=60,
              lr=0.000005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None):

        learning_scheduler = learning_scheduler or {}
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, save_path = get_path(**path_params, file_name='model.h5')

        # 生成器（train/val 均为生成器）
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size})
        data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data'] if data_val is not None else None
        if data_val is not None:
            data_val = data_val[0]  # 取生成器

        # 构建模型
        train_model = self.get_model(data_train['data_params'])

        # 保存结构图
        plot_model(
            train_model,
            to_file=os.path.join(save_path, 'model_structure.png'),
            show_layer_names=True,
            rankdir='TB',
        )

        # 编译
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(
            loss={
                'intention': 'binary_crossentropy',
                'etraj': 'mse'
            },
            loss_weights={
                'intention': 1,
                'etraj': 0  # 轨迹 loss 权重可调
            },
            optimizer=optimizer,
            metrics={
                'intention': ['accuracy'],
                'etraj': ['mae']
            }
        )

        # 回调
        callbacks = []
        callbacks.append(EpochSaveCallback(
            save_dir=os.path.dirname(model_path),
            save_weights_only=False,
            save_format='h5'
        ))
        callbacks.append(DetailedLoggingCallback(
            log_dir=os.path.dirname(model_path),
            log_frequency=1
        ))
        if model_opts.get('use_lr_scheduler', False):
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', mode='min',
                factor=0.2, patience=5, cooldown=1, min_lr=1e-7, verbose=1))

        # 训练（x=生成器，y=None）
        history = train_model.fit(
            x=data_train['data'][0],
            y=None,
            batch_size=None,
            epochs=epochs,
            validation_data=data_val,
            verbose=1,
            callbacks=callbacks
        )

        print('Train model is saved to {}'.format(model_path))
        train_model.save(model_path)

        # 保存 model_opts
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        # 保存训练曲线
        from training_plots import save_training_plots
        save_training_plots(history, path_params, model_opts['model'])

        # 保存 configs
        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs, lr, model_opts)

        # 保存 history 为 yaml
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        history_data = {}
        for key, values in history.history.items():
            history_data[key] = values.tolist() if hasattr(values, 'tolist') else list(values)
        with open(history_path, 'w') as fid:
            yaml.dump(history_data, fid, default_flow_style=False, allow_unicode=True, indent=2)

        return saved_files_path

    def test(self, data_test, model_path=''):
        """
        仅用生成器进行测试；从生成器读取 y_true。
        """
        # 读取配置与模型
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)

        test_model = load_model(os.path.join(model_path, 'model.h5'))

        # 生成器
        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})
        gen = test_data['data'][0]
        if not isinstance(gen, Sequence):
            raise RuntimeError("Test data must be a Sequence generator in generator-only mode.")

        # 预测
        test_results = test_model.predict(gen, verbose=1)

        # 解析输出
        if isinstance(test_results, (list, tuple)) and len(test_results) >= 1:
            intention_results = test_results[0]
            etraj_pred = test_results[1] if len(test_results) > 1 else None
        else:
            intention_results = test_results
            etraj_pred = None

        # 从生成器拿标签
        labels = getattr(gen, "labels", None)
        if not (isinstance(labels, list) and len(labels) > 0 and labels[0] is not None):
            raise ValueError("Generator does not contain ground-truth labels for 'intention'.")
        y_true = np.asarray(labels[0]).reshape(-1)

        # 指标
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

        # 保存结果
        test_output_data = {
            'test_info': {
                'model_path': model_path,
                'test_samples': int(y_true.shape[0]),
                'test_date': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'metrics': {
                'accuracy': float(acc),
                'auc': None if np.isnan(auc) else float(auc),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            },
            'predictions': {
                'ground_truth': y_true.tolist(),
                'intention_predictions': y_prob.tolist(),
                'etraj_predictions': etraj_pred.tolist() if etraj_pred is not None else None,
                'time_to_event': test_data['tte'].flatten().tolist() if hasattr(test_data['tte'], 'flatten') else test_data['tte']
            }
        }

        yaml_output_path = os.path.join(model_path, 'test_output.yaml')
        with open(yaml_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(test_output_data, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)
        print(f"Test results saved to: {yaml_output_path}")

        print('\n' + '\033[96m' + '='*70 + '\033[0m')
        print('\033[1m\033[92m🎯 MODEL TEST RESULTS 🎯\033[0m')
        print('\033[96m' + '='*70 + '\033[0m')
        print('\033[93mAccuracy:   \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(acc))
        print('\033[94mAUC:        \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(0 if np.isnan(auc) else auc))
        print('\033[95mF1-Score:   \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(f1))
        print('\033[96mPrecision:  \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(precision))
        print('\033[91mRecall:     \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(recall))
        print('\033[96m' + '='*70 + '\033[0m\n')

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')
        if not os.path.exists(save_results_path):
            with open(save_results_path, 'w') as fid:
                yaml.dump({'acc': '{:.4f}'.format(acc),
                           'auc': None if np.isnan(auc) else '{:.4f}'.format(auc),
                           'f1': '{:.4f}'.format(f1),
                           'precision': '{:.4f}'.format(precision),
                           'recall': '{:.4f}'.format(recall)}, fid)
        return acc, auc, f1, precision, recall


def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))


# ================================
# Data Generator with sample_weight dict
# ================================
class DataGenerator(Sequence):

    def __init__(self,
                 data=None,
                 labels=None,
                 data_sizes=None,
                 process=False,
                 global_pooling=None,
                 input_type_list=None,
                 batch_size=32,
                 shuffle=True,
                 to_fit=True,
                 stack_feats=False,
                 class_weight=None):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list

        base_len = len(self.labels[0]) if (isinstance(self.labels, list) and len(self.labels) > 0) else len(self.labels)
        self.batch_size = 1 if base_len < batch_size else batch_size

        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()
        self.class_weight = class_weight  # 若外部未提供，则自动估计

        # —— 自动依据训练标签估计正负类权重（仅用于 intention），不需要配置文件
        y_all = self.labels[0] if (isinstance(self.labels, list) and len(self.labels) > 0) else self.labels
        if self.class_weight is None and y_all is not None:
            y_flat = np.asarray(y_all).astype(np.int32).reshape(-1)
            pos = int(np.sum(y_flat))
            neg = int(y_flat.shape[0] - pos)
            total = max(1, pos + neg)
            # 经典做法：权重与频率成反比；谁少谁权重大
            self.class_weight = {0: pos / total, 1: neg / total}
            print(f"[DataGenerator] auto class_weight -> {self.class_weight}")

    def __len__(self):
        return int(np.floor(len(self.data[0]) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            sw = self._generate_sample_weight(y)
            return X, y, sw
        else:
            return X

    def _get_img_features(self, cached_path):
        with open(cached_path, 'rb') as fid:
            try:
                img_features = pickle.load(fid)
            except:
                img_features = pickle.load(fid, encoding='bytes')
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()
        return img_features

    def _generate_X(self, indices):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1] // len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            features_batch[i, ] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i, ..., j * num_ch:j * num_ch + num_ch] = img_features
                            else:
                                features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        if isinstance(self.labels, list) and len(self.labels) > 1:
            intention_labels = np.array(self.labels[0][indices])
            etraj_labels = np.array(self.labels[1][indices]) if self.labels[1] is not None else None
            return [intention_labels, etraj_labels]
        else:
            if isinstance(self.labels, list):
                return np.array(self.labels[0][indices])
            else:
                return np.array(self.labels[indices])

    def _generate_sample_weight(self, y):
        """
        统一规则（无配置、无模式）：
        - intention：按类不平衡自动权重（class_weight）
        - etraj：    与 intention 完全一致（按 y_int 的 class_weight），
                     这样“正类/负类轨迹”自然得到不同权重，且不需要手动设置任何参数。
        """
        if isinstance(y, list) and len(y) > 1:
            y_int = np.asarray(y[0]).astype(np.int32).reshape(-1)
            etraj_labels = y[1]

            # intention 权重
            if self.class_weight is not None:
                sw_int = np.where(y_int == 1, self.class_weight[1], self.class_weight[0]).astype('float32')
            else:
                sw_int = np.ones_like(y_int, dtype='float32')

            # etraj 权重 = intention 权重（无需配置）
            sw_etraj = sw_int.copy()

            # 若轨迹标签缺失/无效（比如 NaN/Inf），则将该样本的 etraj 权重置 0，避免污染回归损失
            if etraj_labels is None:
                sw_etraj[:] = 0.0
            else:
                etraj_np = np.asarray(etraj_labels)
                invalid = ~np.isfinite(etraj_np).all(axis=1)
                if invalid.any():
                    sw_etraj[invalid] = 0.0

            return {'intention': sw_int, 'etraj': sw_etraj}

        else:
            y_int = np.asarray(y).astype(np.int32).reshape(-1)
            if self.class_weight is not None:
                sw = np.where(y_int == 1, self.class_weight[1], self.class_weight[0]).astype('float32')
            else:
                sw = np.ones_like(y_int, dtype='float32')
            return sw


# ================================
# Custom Layers
# ================================
@tf.keras.utils.register_keras_serializable()
class CLSTokenLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.cls_token = self.add_weight(
            shape=(1, 1, d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token"
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.tile(self.cls_token, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


# ================================
# Transformer_depth (Dual-head)
# ================================
class Transformer_depth(ActionPredict):
    """
    多模态 Transformer：意图（二分类）+ 轨迹（4维框回归）
    输入：bbox, depth, vehspd, pedspd（序列）
    输出：intention（1），etraj（4）
    """
    def __init__(self, num_heads=8, d_model=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = dropout
        self.dataset = kwargs['dataset']
        self.sample = kwargs['sample_type']

    def embedding_norm_block(self, input_tensor, name=None):
        # x = Dense(self.d_model, activation=None, kernel_regularizer=regularizers.L2(0.000005), name=f'{name}_embedding_norm')(input_tensor)
        x = Dense(self.d_model, activation=None, name=f'{name}_embedding_norm')(input_tensor)
        x = LayerNormalization(name=f'{name}_ln')(x)
        return x

    def cmim_block(self, x1, x2, dropout=0.1, name=None):
        attn1 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            output_shape=self.d_model,
            # kernel_regularizer=regularizers.L2(0.000005),
            dropout=dropout,
            name=f'{name}_attn1'
        )
        attn2 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            output_shape=self.d_model,
            # kernel_regularizer=regularizers.L2(0.000005),
            dropout=dropout,
            name=f'{name}_attn2'
        )
        y1 = attn1(query=x2, value=x1, key=x1)
        y1 = Dropout(dropout)(y1)
        y1 = Add(name=f'{name}_add1')([x1, y1])

        y2 = attn2(query=x1, value=x2, key=x2)
        y2 = Dropout(dropout)(y2)
        y2 = Add(name=f'{name}_add2')([x2, y2])

        return Add(name=f'{name}_fuse')([y1, y2])

    def fem_block(self, x, dropout=0.1, name=None):
        x = LayerNormalization(name=f'{name}_fem_norm')(x)
        shortcut = x
        x = Dense(2 * self.d_model, activation=tf.nn.gelu, kernel_regularizer=regularizers.L2(0.005), name=f'{name}_fem_ffn1_dense1')(x)
        x = Dense(self.d_model, activation=None, kernel_regularizer=regularizers.L2(0.005), name=f'{name}_fem_ffn1_dense2')(x)
        x = Dropout(dropout, name=f'{name}_fem_drop')(x)
        x = Add(name=f'{name}_fem_add')([shortcut, x])
        return x

    def positional_encoding(self, x):
        def compute_pos_encoding(inputs):
            seq_len = tf.shape(inputs)[1]
            d_model = tf.shape(inputs)[2]
            pos = tf.range(tf.cast(seq_len, tf.float32))[:, tf.newaxis]
            i = tf.range(tf.cast(d_model, tf.float32))[tf.newaxis, :]
            angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            angle_rads = pos * angle_rates
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])
            pos_encoding = tf.concat([sines, cosines], axis=-1)

            def pad_encoding():
                return tf.pad(pos_encoding, [[0, 0], [0, d_model - tf.shape(pos_encoding)[-1]]])

            def slice_encoding():
                return pos_encoding[:, :d_model]

            pos_encoding_adjusted = tf.cond(
                tf.shape(pos_encoding)[-1] < d_model,
                pad_encoding,
                slice_encoding
            )
            pos_encoding_adjusted = pos_encoding_adjusted[tf.newaxis, :, :]
            return inputs + pos_encoding_adjusted
        return Lambda(compute_pos_encoding, name="positional_encoding")(x)

    def mhsa_block(self, x, dropout=0.1, name=None, attention_mask=None):
        x_norm = LayerNormalization(name=f'{name}_mhsa_norm')(x)
        attn = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            value_dim=self.d_model // self.num_heads,
            output_shape=self.d_model,
            # kernel_regularizer=regularizers.L2(0.000005),
            dropout=dropout,
            name=f'{name}_mhsa'
        )
        attn_out = attn(query=x_norm, value=x_norm, key=x_norm)
        x = Dropout(dropout, name=f'{name}_mhsa_drop')(attn_out)
        x = Add(name=f'{name}_mhsa_res')([x, x_norm])
        return x

    def get_model(self, data_params):
        bbox_in = Input(shape=(None, 4), name='bbox')
        depth_in = Input(shape=(None, 1), name='depth')
        vehspd_in = Input(shape=(None, 1), name='vehspd')
        if 'watch' in self.dataset:
            pedspd_in = Input(shape=(None, 3), name='pedspd')
        else:
            pedspd_in = Input(shape=(None, 4), name='pedspd')

        bbox = self.embedding_norm_block(bbox_in, name='bbox')
        depth = self.embedding_norm_block(depth_in, name='depth')
        vehspd = self.embedding_norm_block(vehspd_in, name='vehspd')
        pedspd = self.embedding_norm_block(pedspd_in, name='pedspd')

        x = self.cmim_block(vehspd, pedspd, name='cmim_vehspd_pedspd')
        x = self.fem_block(x, name='fem_vehspd_pedspd')
        x = self.cmim_block(depth, x, name='cmim_depth_vehspd_pedspd')
        x = self.fem_block(x, name='fem_depth_vehspd_pedspd')
        x = self.cmim_block(bbox, x, name='cmim_all')
        x = self.fem_block(x, name='fem_all')

        cls_token = CLSTokenLayer(self.d_model)(x)
        x = Concatenate(axis=1, name='add_cls')([cls_token, x])

        x = self.positional_encoding(x)
        x = self.mhsa_block(x, dropout=0.1, name='mhsa_1')
        x = self.fem_block(x, dropout=0.1, name='fem_after_mhsa_1')

        cls_out = Lambda(lambda t: t[:, 0, :], name='cls_slice')(x)

        # —— Intention head
        ci = Dropout(0.2, name='cls_dropout')(cls_out)
        h = Dense(128, activation='gelu', kernel_regularizer=regularizers.l2(5e-3), name='head_fc1')(ci)
        h = Dropout(0.1, name='head_dropout1')(h)
        intention = Dense(1, activation='sigmoid', name='intention')(h)

        # —— Etraj head（独立分支）
        ce = Dropout(0.1, name='cls_dropout_e')(cls_out)
        e = Dense(128, activation='gelu', kernel_regularizer=regularizers.l2(5e-4), name='head_fc2')(ce)
        e = Dropout(0.1, name='head_dropout2')(e)
        etraj = Dense(4, activation=None, name='etraj')(e)

        model = Model(inputs=[bbox_in, depth_in, vehspd_in, pedspd_in],
                      outputs=[intention, etraj],
                      name='Transformer_depth')
        
        return model

    def get_data(self, data_type, data_raw, model_opts):
        """
        统一只用 DataGenerator（双输出：intention + etraj）
        """
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = True  # 强制只用生成器
        process = model_opts.get('process', True)

        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        # 为 JAAD/PIE 填充速度特征键名
        if model_opts['dataset'] in ('jaad', 'pie'):
            data['vehicle_speed'] = data['speed']
            data['ped_speed'] = data['ped_center_diff']

        _data, data_sizes, data_types = [], [], []
        for d_type in model_opts['obs_input_type']:
            features = data[d_type]
            feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        gen = DataGenerator(
            data=_data,
            labels=[data['crossing'], data['trajectory']],
            data_sizes=data_sizes,
            process=process,
            global_pooling=None,
            input_type_list=model_opts['obs_input_type'],
            batch_size=model_opts['batch_size'],
            shuffle=data_type != 'test',
            to_fit=data_type != 'test'
        )
        packed = (gen, None)

        return {'data': packed,
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        专为 JAAD/PIE 适配：生成 ped_center_diff、trajectory(4维) 与 depth，
        并保持与训练/验证一致的切片方式。
        """
        # ---------- JAAD ----------
        if opts['dataset'] == 'jaad':
            d = {'center': data_raw['center'].copy(),
                 'box': data_raw['bbox'].copy(),
                 'ped_id': data_raw['pid'].copy(),
                 'crossing': data_raw['activities'].copy(),
                 'image': data_raw['image'].copy()}

            balance = opts['balance_data'] if data_type == 'train' else False
            obs_length = opts['obs_length']
            time_to_event = opts['time_to_event']
            normalize = opts['normalize_boxes']

            try:
                d['speed'] = data_raw['obd_speed'].copy()
            except KeyError:
                d['speed'] = data_raw['vehicle_act'].copy()

            if balance:
                self.balance_data_samples(d, data_raw['image_dimension'][0])

            d['tte'] = []

            if isinstance(time_to_event, int):
                for k in d.keys():
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
                overlap = 0
            else:
                overlap = opts['overlap']
                olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
                olap_res = 1 if olap_res < 1 else olap_res
                for k in d.keys():
                    seqs = []
                    for seq in d[k]:
                        start_idx = len(seq) - obs_length - time_to_event[1]
                        end_idx = len(seq) - obs_length - time_to_event[0]
                        seqs.extend([seq[i:i + obs_length] for i in
                                     range(start_idx, end_idx + 1, olap_res)])
                    d[k] = seqs

            # 计算 ped_center_diff 与 轨迹（4维 bbox）
            d['ped_center_diff'] = []
            d['trajectory'] = []
            for idx, seq in enumerate(data_raw['bbox']):
                # 帧间差分
                diffs = []
                for j in range(1, len(seq)):
                    diff = np.array(seq[j]) - np.array(seq[j - 1])
                    diffs.append(diff)
                diffs = [diffs[0]] + diffs

                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['ped_center_diff'].extend([diffs[i:i + obs_length] for i in
                                             range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
                # 下一帧 4维 bbox（像素坐标）
                # d['trajectory'].extend([[seq[i + obs_length][0] / data_raw['image_dimension'][0], seq[i + obs_length][1] / data_raw['image_dimension'][1],
                #                          seq[i + obs_length][2] / data_raw['image_dimension'][0], seq[i + obs_length][3] / data_raw['image_dimension'][1]]
                #                         for i in range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
                d['trajectory'].extend([[seq[i + obs_length][0] + time_to_event[0] - 1, seq[i + obs_length][1] + time_to_event[0] - 1,
                                         seq[i + obs_length][2] + time_to_event[0] - 1, seq[i + obs_length][3] + time_to_event[0] - 1]
                                        for i in range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])

            # depth（尝试缓存）
            cache_dir = 'JAAD/data_cache'
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f'depth_{self.dataset}_{self.sample}_{data_type}_obs{obs_length}_tte{time_to_event[0]}-{time_to_event[1]}_overlap{overlap}.pkl'
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    d['depth'] = pickle.load(f)
            else:
                d['depth'] = []
                for idx, seq in enumerate(data_raw['bbox']):
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                     range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
                    images = data_raw['image'][idx][start_idx:end_idx + obs_length + 1]
                    boxes = data_raw['bbox'][idx][start_idx:end_idx + obs_length + 1]
                    depth_seq = []
                    for image_path, box in zip(images, boxes):
                        depth_image_path = image_path.replace('/images/', '/image_depth_gray/')
                        img = cv2.imread(depth_image_path)
                        img_height, img_width = img.shape[:2]
                        x1, y1, x2, y2 = box
                        x1 = max(0, min(int(x1), img_width - 1))
                        y1 = max(0, min(int(y1), img_height - 1))
                        x2 = max(x1 + 1, min(int(x2), img_width))
                        y2 = max(y1 + 1, min(int(y2), img_height))
                        bbox_region = img[y1:y2, x1:x2]
                        if bbox_region.size == 0:
                            depth_seq.append(None)
                            continue
                        pixel_mean = np.mean(bbox_region)
                        depth_seq.append(float(pixel_mean))
                    d['depth'].extend([depth_seq[i:i + obs_length] for i in
                                       range(0, end_idx - start_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
                with open(cache_path, 'wb') as f:
                    pickle.dump(d['depth'], f, pickle.HIGHEST_PROTOCOL)

            if normalize:
                for k in d.keys():
                    if k != 'tte':
                        if k != 'box' and k != 'center':
                            for i in range(len(d[k])):
                                d[k][i] = d[k][i][1:]
                        else:
                            for i in range(len(d[k])):
                                d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                    d[k] = np.array(d[k])
            else:
                for k in d.keys():
                    d[k] = np.array(d[k])

            d['crossing'] = np.array(d['crossing'])[:, 0, :]
            pos_count = np.count_nonzero(d['crossing'])
            neg_count = len(d['crossing']) - pos_count

        # ---------- PIE ----------
        if opts['dataset'] == 'pie':
            d = {'center': data_raw['center'].copy(),
                 'box': data_raw['bbox'].copy(),
                 'ped_id': data_raw['pid'].copy(),
                 'crossing': data_raw['activities'].copy(),
                 'image': data_raw['image'].copy()}

            balance = opts['balance_data'] if data_type == 'train' else False
            obs_length = opts['obs_length']
            time_to_event = opts['time_to_event']
            normalize = opts['normalize_boxes']

            try:
                d['speed'] = data_raw['obd_speed'].copy()
            except KeyError:
                d['speed'] = data_raw['vehicle_act'].copy()
                print('Jaad dataset does not have speed information')
                print('Vehicle actions are used instead')

            if balance:
                self.balance_data_samples(d, data_raw['image_dimension'][0])
            d['box_org'] = d['box'].copy()
            d['tte'] = []

            if isinstance(time_to_event, int):
                for k in d.keys():
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d['tte'] = [[time_to_event]] * len(data_raw['bbox'])
                overlap = 0
            else:
                overlap = opts['overlap']
                olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
                olap_res = 1 if olap_res < 1 else olap_res
                for k in d.keys():
                    seqs = []
                    for seq in d[k]:
                        start_idx = len(seq) - obs_length - time_to_event[1]
                        end_idx = len(seq) - obs_length - time_to_event[0]
                        seqs.extend([seq[i:i + obs_length] for i in
                                     range(start_idx, end_idx + 1, olap_res)])
                    d[k] = seqs

            d['ped_center_diff'] = []
            d['trajectory'] = []
            for idx, seq in enumerate(data_raw['bbox']):
                diffs = []
                for j in range(1, len(seq)):
                    diff = np.array(seq[j]) - np.array(seq[j - 1])
                    diffs.append(diff)
                diffs = [diffs[0]] + diffs

                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['ped_center_diff'].extend([diffs[i:i + obs_length] for i in
                                             range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
                # PIE 这里示范做归一化（如不需要可改为像素坐标）
                d['trajectory'].extend([[seq[i + obs_length][0], seq[i + obs_length][1],
                                         seq[i + obs_length][2], seq[i + obs_length][3]]
                                        for i in range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])

            cache_dir = 'PIE/data_cache'
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f'depth_{self.dataset}_obs{obs_length}_tte{time_to_event[0]}-{time_to_event[1]}_overlap{overlap}.pkl'
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.exists(cache_path):
                print(f"Loading depth data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    d['depth'] = pickle.load(f)
                print(f"Loaded {len(d['depth'])} depth sequences from cache")
            else:
                print(f"Computing depth data and saving to cache: {cache_path}")
                d['depth'] = []
                for idx, seq in enumerate(data_raw['bbox']):
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                     range(start_idx, end_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])
                    images = data_raw['image'][idx][start_idx:end_idx + obs_length + 1]
                    boxes = data_raw['bbox'][idx][start_idx:end_idx + obs_length + 1]
                    depth_seq = []
                    for image_path, box in zip(images, boxes):
                        depth_image_path = image_path.replace('/images/', '/images_depth_gray/')
                        img = cv2.imread(depth_image_path)
                        img_height, img_width = img.shape[:2]
                        x1, y1, x2, y2 = box
                        x1 = max(0, min(int(x1), img_width - 1))
                        y1 = max(0, min(int(y1), img_height - 1))
                        x2 = max(x1 + 1, min(int(x2), img_width))
                        y2 = max(y1 + 1, min(int(y2), img_height))
                        bbox_region = img[y1:y2, x1:x2]
                        if bbox_region.size == 0:
                            depth_seq.append(None)
                            continue
                        pixel_mean = np.mean(bbox_region)
                        depth_seq.append(float(pixel_mean))
                    d['depth'].extend([depth_seq[i:i + obs_length] for i in
                                       range(0, end_idx - start_idx + 1, 1 if overlap == 0 else int((1 - overlap) * obs_length))])

                print(f"Saving depth data to cache: {cache_path}")
                with open(cache_path, 'wb') as f:
                    pickle.dump(d['depth'], f, pickle.HIGHEST_PROTOCOL)
                print(f"Saved {len(d['depth'])} depth sequences to cache")

            if normalize:
                for k in d.keys():
                    if k != 'tte':
                        if k != 'box' and k != 'center':
                            for i in range(len(d[k])):
                                d[k][i] = d[k][i][1:]
                        else:
                            for i in range(len(d[k])):
                                d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
                    d[k] = np.array(d[k])
            else:
                for k in d.keys():
                    d[k] = np.array(d[k])

            d['crossing'] = np.array(d['crossing'])[:, 0, :]
            pos_count = np.count_nonzero(d['crossing'])
            neg_count = len(d['crossing']) - pos_count
            print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count
