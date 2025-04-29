import json
import requests
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 加载tokenizer和label encoder
tokenizer = pickle.load(open('../../models/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('../../models/label_encoder.pkl', 'rb'))

# 预处理函数
def preprocess_for_lstm(text, tokenizer, max_len=1000):
    # 这里添加与训练时相同的预处理步骤
    # 简化版本：
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len)
    return padded

# 示例新闻文本
test_text = """
Manchester United delivered a commanding performance to defeat Arsenal 3-1 at Old Trafford on Sunday.
Marcus Rashford scored twice and Antony marked his debut with a goal as United made it four wins in a row.
Arsenal had taken the lead through Bukayo Saka, but couldn't maintain their perfect start to the season.
"""

# 预处理文本
processed = preprocess_for_lstm(test_text, tokenizer)

# 准备请求数据
data = json.dumps({
    "signature_name": "serving_default",
    "instances": processed.tolist()
})

# 发送请求到TF Serving
url = "http://localhost:8501/v1/models/news_classifier:predict"
response = requests.post(url, data=data)
predictions = response.json()["predictions"]

# 获取预测类别
pred_idx = np.argmax(predictions[0])
predicted_category = label_encoder.inverse_transform([pred_idx])[0]
print(f"预测的新闻类别: {predicted_category}")

# 显示各类别概率
categories = label_encoder.classes_
probabilities = [(cat, prob) for cat, prob in zip(categories, predictions[0])]
probabilities.sort(key=lambda x: x[1], reverse=True)

print("\n各类别概率:")
for cat, prob in probabilities:
    print(f"{cat}: {prob:.4f}")