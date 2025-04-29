import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle
import os

# 加载数据
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')

# 构建基准模型 - 使用TF-IDF特征和朴素贝叶斯分类器
print("构建基准模型...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# 训练模型
pipeline.fit(train_df['text'], train_df['category'])

# 评估模型
val_pred = pipeline.predict(val_df['text'])
accuracy = accuracy_score(val_df['category'], val_pred)
print(f"基准模型验证集准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(val_df['category'], val_pred))

# 保存模型
os.makedirs('models', exist_ok=True)
with open('models/baseline_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("基准模型已保存到 models/baseline_model.pkl")