import tensorflow as tf
import pickle
import numpy as np
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 加载LSTM模型
lstm_model = tf.keras.models.load_model('../../models/lstm_model.keras')

# 创建保存模型的目录
os.makedirs('1', exist_ok=True)

# 保存模型为SavedModel格式
tf.saved_model.save(lstm_model, '1')

print("模型已导出到TensorFlow SavedModel格式，位于 deployments/tf_serving/1")