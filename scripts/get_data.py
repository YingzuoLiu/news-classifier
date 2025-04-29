import os
import pandas as pd
import requests
import tarfile
from sklearn.model_selection import train_test_split

# 创建数据目录
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# 下载BBC新闻数据集
url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
response = requests.get(url)

# 保存压缩文件
with open('data/raw/bbc-fulltext.zip', 'wb') as f:
    f.write(response.content)

# 解压文件
import zipfile
with zipfile.ZipFile('data/raw/bbc-fulltext.zip', 'r') as zip_ref:
    zip_ref.extractall('data/raw')

# 处理数据
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
texts = []
labels = []

for category in categories:
    path = f'data/raw/bbc/{category}'
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='latin1') as file:
            texts.append(file.read())
            labels.append(category)

# 创建DataFrame
news_df = pd.DataFrame({
    'text': texts,
    'category': labels
})

# 保存完整数据集
news_df.to_csv('data/processed/news_dataset.csv', index=False)

# 分割数据集
train_df, temp_df = train_test_split(news_df, test_size=0.3, random_state=42, stratify=news_df['category'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['category'])

# 保存分割后的数据集
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"数据集已分割: 训练集:{len(train_df)}条, 验证集:{len(val_df)}条, 测试集:{len(test_df)}条")