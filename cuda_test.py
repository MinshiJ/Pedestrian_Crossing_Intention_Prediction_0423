import tensorflow as tf
print("TensorFlow 版本：", tf.__version__)
print("GPU 是否可用：", tf.config.list_physical_devices('GPU'))
