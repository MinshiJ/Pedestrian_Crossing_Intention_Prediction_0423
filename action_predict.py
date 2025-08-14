
import time
import yaml
import wget
import cv2
from utils import *
from base_models import AlexNet, C3DNet, convert_to_fcn, C3DNet2
from base_models import I3DNet
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import GRU, LSTM, GRUCell
from tensorflow.keras.layers import Dropout, LSTMCell, RNN
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Average, Add
from tensorflow.keras.layers import ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda, dot, concatenate, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.activations import gelu

try:
    from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
except ImportError:
    # TensorFlow 2.6及以下版本使用这个路径
    from tensorflow.keras.layers.experimental import LayerNormalization
    from tensorflow.keras.layers import MultiHeadAttention
# from keras_hub.modeling_layers import SinePositionEncoding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


## For deeplabV3 (segmentation)
import numpy as np
from PIL import Image
import matplotlib
import tensorflow as tf
from matplotlib import gridspec
from matplotlib import pyplot as plt
import tarfile
import os
import time
import scipy.misc
import cv2

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np



# TODO: Make all global class parameters to minimum , e.g. no model generation
class ActionPredict(object):
    """
        A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='vgg16',
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
            backbone: Backbone for generating convolutional features
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = None # use data generator for train/test 

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
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
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] # if data_type == 'train' else 0.0
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
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
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

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def get_data(self, data_type, data_raw, model_opts):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """

        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

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

        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=self._global_pooling,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing']) # set y to None
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts, 
                       'train_opts': {'batch_size':batch_size, 'epochs': epochs, 'lr': lr}},
                       fid, default_flow_style=False)

        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count):
        """
        Computes class weights for imbalanced data used during training
        Args:
            apply_weights: Whether to apply weights
            sample_count: Positive and negative sample counts
        Returns:
            A dictionary of class weights or None if no weights to be calculated
        """
        if not apply_weights:
            return None

        total = sample_count['neg_count'] + sample_count['pos_count']
        # formula from sklearn
        # neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        # pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0
        
        # use simple ratio
        neg_weight = sample_count['pos_count']/total
        pos_weight = sample_count['neg_count']/total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callabcks for training
        Args:
            learning_scheduler: Whether to use callbacks
        Returns:
            A list of call backs or None if learning_scheduler is false
        """
        callbacks = None

        # # Set up learning schedulers
        # if learning_scheduler:
        #     callbacks = []
            # if 'early_stop' in learning_scheduler:
            #     default_params = {'monitor': 'val_loss','restore_best_weights': True,
            #                       'min_delta': 1.0, 'patience': 5,
            #                       'verbose': 1}
            #     default_params.update(learning_scheduler['early_stop'])
            #     callbacks.append(EarlyStopping(**default_params))

        #     if 'plateau' in learning_scheduler:
        #         default_params = {'monitor': 'val_loss',
        #                           'factor': 0.2, 'patience': 5,
        #                           'cooldown': 0,
        #                           'min_lr': 1e-08, 'verbose': 1}
        #         default_params.update(learning_scheduler['plateau'])
        #         callbacks.append(ReduceLROnPlateau(**default_params))


        #     if 'checkpoint' in learning_scheduler:
        #         default_params = {'filepath': model_path, 'monitor': 'val_loss',
        #                           'mode': 'min',
        #                           'save_best_only': True, 'save_weights_only': False,
        #                           'save_freq': 'epoch', 'verbose': 2}
        #         default_params.update(learning_scheduler['checkpoint'])
        #         callbacks.append(ModelCheckpoint(**default_params))

        if learning_scheduler:
            callbacks = []
            if 'early_stop' in learning_scheduler:
            # —— 1. 提前停止：连续 patience 轮 val_loss 不下降就早停
                default_params = {'monitor': 'val_loss','restore_best_weights': True,
                                  'min_delta': 1.0, 'patience': 5,
                                  'verbose': 1}
                default_params.update(learning_scheduler['early_stop'])
                callbacks.append(EarlyStopping(**default_params))
            

        # —— 2. 学习率衰减
        if 'plateau' in learning_scheduler:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                # monitor='val_loss',
                monitor='loss',
                mode='min',
                factor=0.2,               # 学习率下降幅度大
                patience=5,               # 多等几轮
                cooldown=1,               # 给它缓一轮
                min_lr=1e-7,
                verbose=1)
            callbacks.append(reduce_lr)

        # —— 3. 模型断点保存：只在 val_loss 改善时保存
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/best.ckpt',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)

        # callbacks = [early_stop, reduce_lr, ckpt]
        # callbacks = [reduce_lr, ckpt]
        # callbacks = [reduce_lr]

        return callbacks

    def get_optimizer(self, optimizer):
        """
        Return an optimizer object
        Args:
            optimizer: The type of optimizer. Supports 'adam', 'sgd', 'rmsprop'
        Returns:
            An optimizer object
        """
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
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
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, _ = get_path(**path_params, file_name='model.h5')

        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size}) 

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Create model
        train_model = self.get_model(data_train['data_params'])

        # Train the model
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        optimizer = self.get_optimizer(optimizer)(lr=lr)
        train_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        ## reivse fit
        # callbacks = self.get_callbacks(learning_scheduler, model_path)
        callbacks = []
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        #     # monitor='val_loss',
        #     monitor='loss',
        #     mode='min',
        #     factor=0.2,               # 学习率下降幅度大
        #     patience=5,               # 多等几轮
        #     cooldown=1,               # 给它缓一轮
        #     min_lr=1e-7,
        #     verbose=1)
        # callbacks.append(reduce_lr)

        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/best.ckpt',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_weights_only=True,
            verbose=1)
        callbacks.append(ckpt)
        
        # data_val = data_val.batch(batch_size)
        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=None,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        # print(history.history.keys())

        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        if model_opts['model'] == 'Transformer_depth':
            from training_plots import save_training_plots
            save_training_plots(history, path_params, model_opts['model'])

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        # with open(history_path, 'wb') as fid:
        #     pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)
        
        # 转换 history.history 为可序列化的格式
        history_data = {}
        for key, values in history.history.items():
            # 将numpy数组转换为Python列表
            if hasattr(values, 'tolist'):
                history_data[key] = values.tolist()
            else:
                history_data[key] = list(values)

        # 保存为YAML格式
        with open(history_path, 'w') as fid:
            yaml.dump(history_data, fid, default_flow_style=False, 
                    allow_unicode=True, indent=2)
            
        return saved_files_path

    # Test Functions
    def test(self, data_test, model_path=''):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
        Returns:
            Evaluation metrics
        """
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)
            # try:
            #     model_opts = pickle.load(fid)
            # except:
            #     model_opts = pickle.load(fid, encoding='bytes')

        test_model = load_model(os.path.join(model_path, 'model.h5'))
        # test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0], batch_size=1, verbose=1)
        
        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        roc = roc_curve(test_data['data'][1], test_results)
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        pre_recall = precision_recall_curve(test_data['data'][1], test_results)

        
        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                        # 'pid': test_data['ped_id'],
                        'gt':test_data['data'][1],
                        'y': test_results,
                        # 'image': test_data['image']
                        }, 
                        picklefile)


        # print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))
        print('\n' + '\033[96m' + '='*70 + '\033[0m')
        print('\033[1m\033[92m🎯 MODEL TEST RESULTS 🎯\033[0m')
        print('\033[96m' + '='*70 + '\033[0m')
        print('\033[93mAccuracy:   \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(acc))
        print('\033[94mAUC:        \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(auc))
        print('\033[95mF1-Score:   \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(f1))
        print('\033[96mPrecision:  \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(precision))
        print('\033[91mRecall:     \033[0m\033[1m\033[92m{:.4f}\033[0m'.format(recall))
        print('\033[96m' + '='*70 + '\033[0m\n')


        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': '{:.4f}'.format(acc),
                    'auc': '{:.4f}'.format(auc),
                    'f1': '{:.4f}'.format(f1),
                    # 'roc': '{:.4f}'.format(roc),
                    'precision': '{:.4f}'.format(precision),
                    'recall': '{:.4f}'.format(recall),
                    # 'pre_recall_curve': '{:.4f}'.format(pre_recall)
                    }

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)
        return acc, auc, f1, precision, recall

    def get_model(self, data_params):
        """
        Generates a model
        Args:
            data_params: Data parameters to use for model generation
        Returns:
            A model
        """
        raise NotImplementedError("get_model should be implemented")


def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))


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
                 stack_feats=False):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size  else batch_size        
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data[0])/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]

        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
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
            num_ch = features_batch.shape[-1]//len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str):
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            # for static model if only one image in the sequence
                            features_batch[i, ] = img_features
                        else:
                            if self.stack_feats and 'flow' in input_type:
                                features_batch[i,...,j*num_ch:j*num_ch+num_ch] = img_features
                            else:
                                features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices):
        if 'depth' in self.input_type_list:
            # # 如果有深度图，labels[0]是行人过街意图标签，labels[1]是下一帧的xy坐标
            # intention_labels = np.array(self.labels[0][indices])
            # etraj_labels = np.array(self.labels[1][indices])  # 下一帧xy坐标
            # return [intention_labels, etraj_labels]
            return np.array(self.labels[indices])
        else:
            return np.array(self.labels[indices])

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
        # 返回可广播的 token tensor
        return tf.tile(self.cls_token, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model
        })
        return config

@tf.keras.utils.register_keras_serializable()
class LearnablePosEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len  = max_len
        self.d_model  = d_model
        # (max_len, d_model) 可学习矩阵
        self.pos_emb = self.add_weight(
            shape=(max_len, d_model),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )

    def call(self, inputs):
        # inputs: (batch, seq_len, d_model)
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_emb[tf.newaxis, :seq_len, :]

    # ✨ 关键补丁：序列化配置 ✨
    def get_config(self):
        # 先拿父类的基本配置
        config = super().get_config().copy()
        # 把自定义参数写进去
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class CMIMBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, num_hidden_units, dropout=0.1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.num_hidden_units = num_hidden_units
        self.dropout_rate = dropout

        # 注意力层
        self.attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=num_hidden_units, name=f'{name}_attn1')
        self.attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=num_hidden_units, name=f'{name}_attn2')

        # Dropout + LayerNorm
        self.dropout = Dropout(dropout)
        self.ln1 = LayerNormalization(name=f'{name}_ln1')
        self.ln2 = LayerNormalization(name=f'{name}_ln2')

        # 可学习缩放因子
        self.gamma1 = self.add_weight(
            name=f'{name}_gamma1',
            shape=(1,),
            initializer=tf.constant_initializer(0.1),
            trainable=True
        )
        self.gamma2 = self.add_weight(
            name=f'{name}_gamma2',
            shape=(1,),
            initializer=tf.constant_initializer(0.1),
            trainable=True
        )

    def call(self, x1, x2, training=False):
        # x2 attends to x1
        y1_attn = self.attn1(query=x2, value=x1, key=x1)
        y1_attn = self.dropout(y1_attn, training=training)
        y1 = x1 + self.gamma1 * y1_attn
        y1 = self.ln1(y1)

        # x1 attends to x2
        y2_attn = self.attn2(query=x1, value=x2, key=x2)
        y2_attn = self.dropout(y2_attn, training=training)
        y2 = x2 + self.gamma2 * y2_attn
        y2 = self.ln2(y2)

        return y1 + y2

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "num_hidden_units": self.num_hidden_units,
            "dropout": self.dropout_rate,
            "name": self.name,
        })
        return config

class Transformer_depth(ActionPredict):
    """
    多模态Transformer网络结构，支持行人过街意图与轨迹联合预测。
    输入：Bounding Box, Depth, Vehicle Speed, Pedestrian Speed（均为序列）
    输出：Intention（二分类），E-Traj（下一帧xy坐标）
    """
    def __init__(self, num_heads=4, num_hidden_units=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_hidden_units = num_hidden_units
        self.dropout = dropout
        self.dataset = kwargs['dataset']
        self.sample = kwargs['sample_type']


    # def embedding_norm_block(self, input_tensor, name=None):
    #     """Dense embedding + LayerNorm"""
    #     # x = Dense(self.ffn_units, activation=None, name=f'{name}_emb')(input_tensor)
    #     x = Dense(self.ffn_units, activation=gelu, name=f'{name}_emb2')(x)
    #     x = LayerNormalization(name=f'{name}_ln')(x)
    #     return x
    
    # # 典型 Transformer FFN Block
    # def embedding_norm_block(self, input_tensor, name=None):
    #     shortcut = input_tensor
    #     x = Dense(4 * self.ffn_units, activation='gelu', name=f'{name}_ffn1')(input_tensor)
    #     x = Dense(self.ffn_units, activation=None, name=f'{name}_ffn2')(x)
    #     x = Dropout(self.dropout)(x)
    #     x = Add(name=f'{name}_res')([shortcut, x])
    #     x = LayerNormalization(name=f'{name}_ln')(x)
    #     return x

    def embedding_norm_block(self, input_tensor, name=None):
        """Transformer-style FFN block: Dense + GELU + Dropout + Residual + LayerNorm"""
        
        # FFN sub-layer
        x = Dense(2 * self.num_hidden_units, activation=gelu, name=f'{name}_ffn1')(input_tensor)
        # x = Dense(2 * self.num_hidden_units, activation='tanh', name=f'{name}_ffn1')(input_tensor)
        x = Dense(self.num_hidden_units, activation=None, name=f'{name}_ffn2')(x)
        
        # Layer normalization
        x = LayerNormalization(name=f'{name}_ln')(x)
        
        return x

    # def fem_block(self, x, name=None):
    #     """Feature Enhancement Module: Norm -> 2层FFN -> Dropout&Add (残差)"""
    #     shortcut = x
    #     x = LayerNormalization(name=f'{name}_fem_norm')(x)
    #     x = Dense(self.ffn_units, activation='relu', name=f'{name}_fem_ffn1')(x)
    #     x = Dropout(self.dropout, name=f'{name}_fem_drop1')(x)
    #     x = Dense(self.ffn_units, activation='relu', name=f'{name}_fem_ffn2')(x)
    #     x = Dropout(self.dropout, name=f'{name}_fem_drop2')(x)
    #     x = Add(name=f'{name}_fem_add')([shortcut, x])
    #     return x

    def fem_block(self, x, name=None):
        """Feature Enhancement Module: PreNorm -> FFN (GELU+Linear) -> Residual Add"""
        x = LayerNormalization(name=f'{name}_fem_norm')(x)
        shortcut = x
        x = Dense(2 * self.num_hidden_units, activation=tf.nn.gelu, name=f'{name}_fem_ffn1_dense1')(x)
        x = Dense(self.num_hidden_units, activation=None, name=f'{name}_fem_ffn1_dense2')(x)
        x = Dense(2 * self.num_hidden_units, activation=tf.nn.gelu, name=f'{name}_fem_ffn2_dense1')(x)
        x = Dense(self.num_hidden_units, activation=None, name=f'{name}_fem_ffn2_dense2')(x)
        x = Dropout(self.dropout, name=f'{name}_fem_drop')(x)
        x = Add(name=f'{name}_fem_add')([shortcut, x])
        return x

    # def cmim_block(self, x1, x2, name=None):
    #     """Cross-Modal Interaction Module: 双分支多头注意力+残差"""
    #     attn1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.ffn_units, name=f'{name}_attn1')
    #     attn2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.ffn_units, name=f'{name}_attn2')
    #     y1 = attn1(query=x1, value=x2, key=x2)
    #     y2 = attn2(query=x2, value=x1, key=x1)
    #     y1 = Add(name=f'{name}_add1')([x1, y1])
    #     y2 = Add(name=f'{name}_add2')([x2, y2])
    #     return y1 + y2

    def cmim_block(self, x1, x2, name=None):
        """Cross-Modal Interaction Module: 双向交叉注意力 + 残差"""
        attn1 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_hidden_units, name=f'{name}_attn1')
        y1 = attn1(query=x2, value=x1, key=x1)
        y1 = Dropout(self.dropout)(y1)
        y1 = Add(name=f'{name}_add1')([x1, y1])
        y1 = LayerNormalization(name=f'{name}_ln1')(y1)

        attn2 = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_hidden_units, name=f'{name}_attn2')
        y2 = attn2(query=x1, value=x2, key=x2)
        y2 = Dropout(self.dropout)(y2)
        y2 = Add(name=f'{name}_add2')([x2, y2])
        y2 = LayerNormalization(name=f'{name}_ln2')(y2)

        return y1 + y2

    def positional_encoding(self, x):
        """正余弦位置编码"""
        def compute_pos_encoding(inputs):
            seq_len = tf.shape(inputs)[1]
            d_model = tf.shape(inputs)[2]
            
            # 创建位置和维度索引
            pos = tf.range(tf.cast(seq_len, tf.float32))[:, tf.newaxis]
            i = tf.range(tf.cast(d_model, tf.float32))[tf.newaxis, :]
            
            # 计算角度率
            angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            angle_rads = pos * angle_rates
            
            # 对偶数索引应用sin，对奇数索引应用cos
            sines = tf.sin(angle_rads[:, 0::2])
            cosines = tf.cos(angle_rads[:, 1::2])
            
            # 拼接sin和cos
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            
            # 使用tf.cond处理维度匹配，避免直接使用Python if
            def pad_encoding():
                return tf.pad(pos_encoding, [[0, 0], [0, d_model - tf.shape(pos_encoding)[-1]]])
            
            def slice_encoding():
                return pos_encoding[:, :d_model]
            
            pos_encoding_adjusted = tf.cond(
                tf.shape(pos_encoding)[-1] < d_model,
                pad_encoding,
                slice_encoding
            )
            
            # 添加batch维度并与输入相加
            pos_encoding_adjusted = pos_encoding_adjusted[tf.newaxis, :, :]
            return inputs + pos_encoding_adjusted

        return Lambda(compute_pos_encoding, name="positional_encoding")(x)
    
    def mhsa_block(self, x, name=None):
        """Multi-Head Self Attention + 残差"""
        x = LayerNormalization(name=f'{name}_mhsa_norm')(x)
        shortcut = x
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.num_hidden_units, name=f'{name}_mhsa')(x, x)
        x = Dropout(self.dropout, name=f'{name}_mhsa_drop')(x)
        return x + shortcut

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

        # x = self.cmim_block(vehspd, pedspd, name='cmim_vehspd_pedspd')
        # x = self.fem_block(x, name='fem_vehspd_pedspd')
        # x = self.cmim_block(depth, x, name='cmim_depth_vehspd_pedspd')
        # x = self.fem_block(x, name='fem_depth_vehspd_pedspd')
        # x = self.cmim_block(bbox, x, name='cmim_all')
        # x = self.fem_block(x, name='fem_all')

        x = CMIMBlock(4, 128, 0.1, name='cmim_vehspd_pedspd')(vehspd, pedspd)
        x = self.fem_block(x, name='fem_vehspd_pedspd')

        x = CMIMBlock(4, 128, 0.1, name='cmim_depth_vehspd_pedspd')(depth, x)
        x = self.fem_block(x, name='fem_depth_vehspd_pedspd')

        x = CMIMBlock(4, 128, 0.1, name='cmim_all')(bbox, x)
        x = self.fem_block(x, name='fem_all')

        # # cls_token = tf.zeros_like(x[:, :1, :])
        cls_token = CLSTokenLayer(self.num_hidden_units)(x)
        x = Concatenate(axis=1, name='add_cls')([cls_token, x])

        # Add positional encoding
        x = self.positional_encoding(x)
        # x = LearnablePosEncoding(max_len=128, d_model=self.num_hidden_units)(x)

        x = self.mhsa_block(x, name='mhsa_1')
        x = self.fem_block(x, name='fem_after_mhsa_1')
        x = self.mhsa_block(x, name='mhsa_2')
        x = self.fem_block(x, name='fem_after_mhsa_2')
        # x = self.mhsa_block(x, name='mhsa_3')
        # x = self.fem_block(x, name='fem_after_mhsa_3')
        # x = self.mhsa_block(x, name='mhsa_4')
        # x = self.fem_block(x, name='fem_after_mhsa_4')


        cls_out = Lambda(lambda t: t[:, 0, :], name='cls_slice')(x)
        intention = Dense(1, activation='sigmoid', name='intention')(cls_out)
        # etraj = Dense(2, activation=None, name='etraj')(cls_out)

        # model = Model(inputs=[bbox_in, depth_in, vehspd_in, pedspd_in], outputs=[intention, etraj], name='Transformer_depth')
        model = Model(inputs=[bbox_in, depth_in, vehspd_in, pedspd_in], outputs=intention, name='Transformer_depth')
        plot_model(model, to_file='model_imgs/Transformer_depth.png')
        return model

    def get_data(self, data_type, data_raw, model_opts):
        assert model_opts['obs_length'] == 16
        model_opts['normalize_boxes'] = False
        self._generator = model_opts.get('generator', False)
        # data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)
        # data = convert_array_of_lists_to_list_of_arrays()
        # data_type_sizes_dict['box'] = data['box'].shape[1:]
        # if 'speed' in data.keys():
        #     data_type_sizes_dict['speed'] = data['speed'].shape[1:]
        # # if 'context_cnn' in data.keys():
        #     data_type_sizes_dict['context_cnn'] = data['context_cnn'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        # model_opts_3d = model_opts.copy()
        if model_opts['dataset'] == 'jaad' or 'pie':
            data['vehicle_speed'] = data['speed']
            data['ped_speed'] = data['ped_center_diff']

        for d_type in model_opts['obs_input_type']:
            features = data[d_type]
            feat_shape = features.shape[1:]
            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)
        # create the final data file to be returned
        if self._generator:
            _data = (DataGenerator(data=_data,
                                   labels=data['crossing'],
                                   data_sizes=data_sizes,
                                   process=process,
                                   global_pooling=None,
                                   input_type_list=model_opts['obs_input_type'],
                                   batch_size=model_opts['batch_size'],
                                   shuffle=data_type != 'test',
                                   to_fit=data_type != 'test'), data['crossing'])  # set y to None
        # global_pooling=self._global_pooling,
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                # 'ped_id': data['ped_id'],
                'tte': data['tte'],
                # 'image': data['image'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def get_data_sequence(self, data_type, data_raw, opts):
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
##########################################################################################
# 处理JAAD数据集
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
                print('Jaad dataset does not have speed information')
                print('Vehicle actions are used instead')
            if balance:
                self.balance_data_samples(d, data_raw['image_dimension'][0])
            # d['box_org'] = d['box'].copy()
            d['tte'] = []

            if isinstance(time_to_event, int):
                for k in d.keys():
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
            else:
                overlap = opts['overlap'] # if data_type == 'train' else 0.0
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
            for idx, seq in enumerate(data_raw['bbox']):
                diffs = []
                for j in range(1, len(seq)):
                    diff = np.array(seq[j]) - np.array(seq[j-1])
                    diffs.append(diff)
                # 将第一个差值复制放在开头以保持序列长度
                diffs = [diffs[0]] + diffs

                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['ped_center_diff'].extend([diffs[i:i + obs_length] for i in
                                             range(start_idx, end_idx + 1, olap_res)])


            # 计算深度信息，先检查缓存
            import os
            import pickle
            
            cache_dir = 'JAAD/data_cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # 生成唯一的缓存文件名
            cache_filename = f'depth_{self.dataset}_{self.sample}_{data_type}_obs{obs_length}_tte{time_to_event[0]}-{time_to_event[1]}_overlap{overlap}.pkl'
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # 检查缓存文件是否存在
            if os.path.exists(cache_path):
                print(f"Loading depth data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    d['depth'] = pickle.load(f)
                print(f"Loaded {len(d['depth'])} depth sequences from cache")
            else:
                print(f"Computing depth data and saving to cache: {cache_path}")
                # 计算深度信息
                d['depth'] = []
                for idx, seq in enumerate(data_raw['bbox']):
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                    range(start_idx, end_idx + 1, olap_res)])
                    images = data_raw['image'][idx][start_idx:end_idx + obs_length + 1]
                    boxes = data_raw['bbox'][idx][start_idx:end_idx + obs_length + 1]
                    depth_seq = []
                    for image_path, box in zip(images, boxes):
                        # 修改图像路径：将 'images' 替换为 'image_depth_gray'
                        depth_image_path = image_path.replace('/images/', '/image_depth_gray/')     
                        # 读取图像
                        img = cv2.imread(depth_image_path)
                        # if img is None:
                        #     print(f"Warning: Cannot read image {depth_image_path}")
                        #     # depth_seq.append(0.0)  # 或者跳过
                        #     continue
                        # 获取图像尺寸和边界框坐标
                        img_height, img_width = img.shape[:2]
                        x1, y1, x2, y2 = box
                        
                        # 确保边界框坐标在有效范围内
                        x1 = max(0, min(int(x1), img_width - 1))
                        y1 = max(0, min(int(y1), img_height - 1))
                        x2 = max(x1 + 1, min(int(x2), img_width))
                        y2 = max(y1 + 1, min(int(y2), img_height))
                        
                        # 提取边界框区域
                        bbox_region = img[y1:y2, x1:x2]
                        
                        if bbox_region.size == 0:
                            print(f"Warning: Empty bbox region for image {image_path}")
                            depth_seq.append(None)
                            continue
                        
                        # 计算像素平均值（所有通道的平均值）
                        pixel_mean = np.mean(bbox_region)
                        depth_seq.append(float(pixel_mean))
                    d['depth'].extend([depth_seq[i:i + obs_length] for i in
                                    range(0, end_idx - start_idx + 1, olap_res)])
                    
                    # 显示进度
                    if (idx + 1) % 10 == 0:
                        print(f"Processed depth for {idx + 1}/{len(data_raw['bbox'])} sequences")

                # 保存到缓存
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

##########################################################################################
# 处理PIE数据集
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
                d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
            else:
                overlap = opts['overlap'] # if data_type == 'train' else 0.0
                olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
                olap_res = 1 if olap_res < 1 else olap_res
                for k in d.keys():
                    seqs = []
                    for seq in d[k]:
                        start_idx = len(seq) - obs_length - time_to_event[1]
                        end_idx = len(seq) - obs_length - time_to_event[0]
                        seqs.extend([seq[i:i + obs_length] for i in
                                    range(start_idx, end_idx + 1, olap_res)])

                        # # 计算序列长度
                        # sequence_length = len([seq[i:i + obs_length] for i in
                        #             range(start_idx, end_idx + 1, olap_res)])
                        # # 记录到文件
                        # with open(f'sequence_lengths_{data_type}_{self.dataset}_{self.sample}.txt', 'a') as log_file:
                        #     log_file.write(f"{start_idx:4d}, {end_idx:4d}, {sequence_length:4d}\n")
                    d[k] = seqs

            d['ped_center_diff'] = []
            for idx, seq in enumerate(data_raw['bbox']):
                diffs = []
                for j in range(1, len(seq)):
                    diff = np.array(seq[j]) - np.array(seq[j-1])
                    diffs.append(diff)
                # 将第一个差值复制放在开头以保持序列长度
                diffs = [diffs[0]] + diffs

                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                d['ped_center_diff'].extend([diffs[i:i + obs_length] for i in
                                             range(start_idx, end_idx + 1, olap_res)])


            # 计算深度信息，先检查缓存
            import os
            import pickle
            
            cache_dir = 'PIE/data_cache'
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # 生成唯一的缓存文件名
            cache_filename = f'depth_{self.dataset}_obs{obs_length}_tte{time_to_event[0]}-{time_to_event[1]}_overlap{overlap}.pkl'
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # 检查缓存文件是否存在
            if os.path.exists(cache_path):
                print(f"Loading depth data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    d['depth'] = pickle.load(f)
                print(f"Loaded {len(d['depth'])} depth sequences from cache")
            else:
                print(f"Computing depth data and saving to cache: {cache_path}")
                # 计算深度信息
                d['depth'] = []
                for idx, seq in enumerate(data_raw['bbox']):
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                    range(start_idx, end_idx + 1, olap_res)])
                    images = data_raw['image'][idx][start_idx:end_idx + obs_length + 1]
                    boxes = data_raw['bbox'][idx][start_idx:end_idx + obs_length + 1]
                    depth_seq = []
                    for image_path, box in zip(images, boxes):
                        # 修改图像路径：将 'images' 替换为 'image_depth_gray'
                        depth_image_path = image_path.replace('/images/', '/images_depth_gray/')     
                        # 读取图像
                        img = cv2.imread(depth_image_path)
                        # if img is None:
                        #     print(f"Warning: Cannot read image {depth_image_path}")
                        #     # depth_seq.append(0.0)  # 或者跳过
                        #     continue
                        # 获取图像尺寸和边界框坐标
                        img_height, img_width = img.shape[:2]
                        x1, y1, x2, y2 = box
                        
                        # 确保边界框坐标在有效范围内
                        x1 = max(0, min(int(x1), img_width - 1))
                        y1 = max(0, min(int(y1), img_height - 1))
                        x2 = max(x1 + 1, min(int(x2), img_width))
                        y2 = max(y1 + 1, min(int(y2), img_height))
                        
                        # 提取边界框区域
                        bbox_region = img[y1:y2, x1:x2]
                        
                        if bbox_region.size == 0:
                            print(f"Warning: Empty bbox region for image {image_path}")
                            depth_seq.append(None)
                            continue
                        
                        # 计算像素平均值（所有通道的平均值）
                        pixel_mean = np.mean(bbox_region)
                        depth_seq.append(float(pixel_mean))
                    d['depth'].extend([depth_seq[i:i + obs_length] for i in
                                    range(0, end_idx - start_idx + 1, olap_res)])
                    
                    # 显示进度
                    if (idx + 1) % 10 == 0:
                        print(f"Processed depth for {idx + 1}/{len(data_raw['bbox'])} sequences")

                # 保存到缓存
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

##########################################################################################
# 处理Watch_Ped数据集
        if 'watch' in opts['dataset']:
            d = {'depth': data_raw['depth'].copy(),
                'box': data_raw['bbox'].copy(),
                'ped_speed': data_raw['ped_speed'].copy(),
                'vehicle_speed': data_raw['vehicle_speed'].copy(),
                'crossing': data_raw['crossing'].copy(),             'image_dimension': data_raw['image_dimension'].copy()}

            balance = opts['balance_data'] if data_type == 'train' else False
            obs_length = opts['obs_length']
            time_to_event = opts['time_to_event']
            normalize = opts['normalize_boxes']

            d['box_org'] = d['box'].copy()
            d['tte'] = []
            d['trajectory'] = []  # 存储轨迹数据

            if isinstance(time_to_event, int):
                for k in d.keys():
                    for i in range(len(d[k])):
                        d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
                d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
            else:
                overlap = opts['overlap'] # if data_type == 'train' else 0.0
                olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
                olap_res = 1 if olap_res < 1 else olap_res
                crossing_seqs = []
                image_dim_seqs = []
                trajectory_seqs = []
                for k in d.keys():
                    seqs = []
                    for idx, seq in enumerate(d[k]):
                        if k == 'box':
                            # 检查序列长度是否足够
                            if len(seq) < obs_length + time_to_event[0]:
                                # print(f"Warning: Sequence length {len(seq)} is too short for obs_length {obs_length} and time_to_event {time_to_event}")
                                continue  # 跳过这个序列
                            start_idx = len(seq) - obs_length - time_to_event[1]
                            end_idx = len(seq) - obs_length - time_to_event[0]
                            # 确保索引不为负数
                            if start_idx < 0:
                                start_idx = 0
                            if end_idx < start_idx:
                                continue  # 跳过无效范围
                            num_samples = len(range(start_idx, end_idx + 1, olap_res))
                            crossing_seqs.extend([d['crossing'][idx] for _ in range(num_samples)])
                            image_dim_seqs.extend([d['image_dimension'][idx] for _ in range(num_samples)])
                            # 获取当前序列的图像尺寸
                            img_width, img_height = d['image_dimension'][idx]
                            # seqs.extend([seq[i:i + obs_length] for i in
                            #         range(start_idx, end_idx + 1, olap_res)])
                            # trajectory_seqs.extend([[(seq[i + obs_length][0] + seq[i + obs_length][1]) / 2,
                            #                          (seq[i + obs_length][2] + seq[i + obs_length][3]) / 2] for i in
                            #         range(start_idx, end_idx + 1, olap_res)])
                            # 归一化边界框坐标并添加到seqs
                            normalized_box_seqs = []
                            for i in range(start_idx, end_idx + 1, olap_res):
                                box_seq = seq[i:i + obs_length]
                                # 归一化每个边界框 [x1, y1, x2, y2]
                                normalized_seq = []
                                for box in box_seq:
                                    normalized_box = [
                                        box[0] / img_width,   # x1 / width
                                        box[1] / img_height,  # y1 / height
                                        box[2] / img_width,   # x2 / width
                                        box[3] / img_height   # y3 / height
                                    ]
                                    normalized_seq.append(normalized_box)
                                normalized_box_seqs.append(normalized_seq)
                            seqs.extend(normalized_box_seqs)
                            
                            # 归一化轨迹坐标
                            normalized_traj_seqs = []
                            for i in range(start_idx, end_idx + 1, olap_res):
                                center_x = (seq[i + obs_length][0] + seq[i + obs_length][2]) / 2
                                center_y = (seq[i + obs_length][1] + seq[i + obs_length][3]) / 2
                                # 归一化轨迹坐标
                                normalized_traj = [
                                    center_x / img_width,   # 中心点x坐标 / 图像宽度
                                    center_y / img_height   # 中心点y坐标 / 图像高度
                                ]
                                normalized_traj_seqs.append(normalized_traj)
                            trajectory_seqs.extend(normalized_traj_seqs)
                            # d['crossing'] = crossing_seqs
                            # d['image_dimension'] = image_dim_seqs
                            
                        elif k != 'crossing' and k != 'image_dimension' and k != 'trajectory':
                            if len(seq) < obs_length + time_to_event[0]:
                                # print(f"Warning: Sequence length {len(seq)} is too short for obs_length {obs_length} and time_to_event {time_to_event}")
                                continue  # 跳过这个序列
                            start_idx = len(seq) - obs_length - time_to_event[1]
                            end_idx = len(seq) - obs_length - time_to_event[0]
                            # 确保索引不为负数
                            if start_idx < 0:
                                start_idx = 0
                            if end_idx < start_idx:
                                continue  # 跳过无效范围
                            seqs.extend([seq[i:i + obs_length] for i in
                                    range(start_idx, end_idx + 1, olap_res)])
                    d[k] = seqs
                d['crossing'] = crossing_seqs
                d['image_dimension'] = image_dim_seqs   
                d['trajectory'] = trajectory_seqs

                for seq in data_raw['bbox']:
                    start_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - time_to_event[0]
                    d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                                    range(start_idx, end_idx + 1, olap_res)])
            # if normalize:
            #     for k in d.keys():
            #         if k != 'tte':
            #             if k != 'box' and k != 'center':
            #                 for i in range(len(d[k])):
            #                     d[k][i] = d[k][i][1:]
            #             else:
            #                 for i in range(len(d[k])):
            #                     d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
            #         d[k] = np.array(d[k])
            # else:
            for k in d.keys():
            # if k != 'tte':
                for i in range(len(d[k])):
                    if not isinstance(d[k][i], np.ndarray):
                        d[k][i] = np.array(d[k][i])
            # for k in d.keys():
                d[k] = np.array(d[k])

            # d['crossing'] = np.array(d['crossing'])[:, 0, :]
            pos_count = np.count_nonzero(d['crossing'])
            neg_count = len(d['crossing']) - pos_count
            print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        return d, neg_count, pos_count
