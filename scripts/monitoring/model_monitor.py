import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
import datetime
import json

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入预处理和预测函数
exec(open('../../models/ensemble_predict_func.py').read())

class ModelMonitor:
    def __init__(self, model_dir='../../models', data_dir='../../data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.log_dir = '../../logs'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 加载模型和相关组件
        self.tfidf_pipeline = pickle.load(open(f'{model_dir}/best_tfidf_model.pkl', 'rb'))
        self.lstm_model = tf.keras.models.load_model(f'{model_dir}/lstm_model')
        self.tokenizer = pickle.load(open(f'{model_dir}/tokenizer.pkl', 'rb'))
        self.label_encoder = pickle.load(open(f'{model_dir}/label_encoder.pkl', 'rb'))
        
        # 加载测试数据
        self.test_df = pd.read_csv(f'{data_dir}/processed/test.csv')
        
        # 性能记录
        self.performance_history = []
        self.load_performance_history()
        
    def load_performance_history(self):
        """加载历史性能记录"""
        history_file = f'{self.log_dir}/performance_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                try:
                    self.performance_history = json.load(f)
                except:
                    self.performance_history = []
    
    def save_performance_history(self):
        """保存性能记录"""
        history_file = f'{self.log_dir}/performance_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f)
    
    def evaluate_model(self, sample_size=None):
        """评估模型在测试集上的性能"""
        # 如果指定了样本大小，则随机采样
        if sample_size and sample_size < len(self.test_df):
            test_sample = self.test_df.sample(sample_size, random_state=42)
        else:
            test_sample = self.test_df
        
        # 进行预测
        y_pred = []
        for text in test_sample['processed_text']:
            pred, _ = ensemble_predict(text, self.tfidf_pipeline, self.lstm_model, 
                                     self.tokenizer, self.label_encoder)
            y_pred.append(pred)
        
        # 计算性能指标
        accuracy = accuracy_score(test_sample['category'], y_pred)
        report = classification_report(test_sample['category'], y_pred, output_dict=True)
        
        # 记录性能
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        performance = {
            'timestamp': timestamp,
            'accuracy': accuracy,
            'report': report
        }
        self.performance_history.append(performance)
        self.save_performance_history()
        
        return performance
    
    def monitor_data_drift(self, new_data_path):
        """监控数据漂移"""
        # 加载新数据
        try:
            new_data = pd.read_csv(new_data_path)
        except:
            print(f"无法加载新数据: {new_data_path}")
            return None
        
        # 计算类别分布
        original_dist = self.test_df['category'].value_counts(normalize=True)
        new_dist = new_data['category'].value_counts(normalize=True)
        
        # 计算分布差异（使用JS散度或KL散度）
        def kl_divergence(p, q):
            # 确保所有类别都在两个分布中
            all_categories = set(list(p.index) + list(q.index))
            p_complete = pd.Series({cat: p.get(cat, 0.001) for cat in all_categories})
            q_complete = pd.Series({cat: q.get(cat, 0.001) for cat in all_categories})
            
            # 归一化
            p_complete = p_complete / p_complete.sum()
            q_complete = q_complete / q_complete.sum()
            
            # 计算KL散度
            return np.sum(p_complete * np.log(p_complete / q_complete))
        
        drift_score = kl_divergence(original_dist, new_dist)
        
        # 记录数据漂移
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        drift_report = {
            'timestamp': timestamp,
            'drift_score': drift_score,
            'original_distribution': original_dist.to_dict(),
            'new_distribution': new_dist.to_dict()
        }
        
        # 保存报告
        drift_file = f'{self.log_dir}/data_drift_{timestamp.replace(":", "-").replace(" ", "_")}.json'
        with open(drift_file, 'w') as f:
            json.dump(drift_report, f)
        
        return drift_report
    
    def plot_performance_history(self):
        """绘制性能历史记录"""
        if not self.performance_history:
            print("没有性能历史记录可供绘制")
            return
        
        # 提取时间戳和准确率
        timestamps = [p['timestamp'] for p in self.performance_history]
        accuracies = [p['accuracy'] for p in self.performance_history]
        
        # 绘制准确率随时间变化的趋势
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, accuracies, 'o-')
        plt.xlabel('时间')
        plt.ylabel('准确率')
        plt.title('模型性能随时间的变化')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.log_dir}/performance_trend.png')
        plt.close()
        
        # 最新的分类报告
        latest_report = self.performance_history[-1]['report']
        categories = sorted(list(latest_report.keys() - ['accuracy', 'macro avg', 'weighted avg']))
        
        # 绘制各类别的F1分数
        f1_scores = [latest_report[cat]['f1-score'] for cat in categories]
        plt.figure(figsize=(10, 6))
        plt.bar(categories, f1_scores)
        plt.xlabel('类别')
        plt.ylabel('F1分数')
        plt.title('各类别的F1分数')
        plt.ylim(0, 1)
        plt.savefig(f'{self.log_dir}/category_f1_scores.png')
        plt.close()

# 使用示例
if __name__ == "__main__":
    monitor = ModelMonitor()
    # 评估模型
    performance = monitor.evaluate_model(sample_size=100)
    print(f"模型准确率: {performance['accuracy']:.4f}")
    
    # 绘制性能历史
    monitor.plot_performance_history()
    
    print("监控完成! 结果已保存到logs目录。")