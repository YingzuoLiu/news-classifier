from flask import Flask, request, jsonify, render_template_string
import pickle
import tensorflow as tf
import numpy as np
import os
import sys

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入预处理和预测函数
exec(open('../../models/ensemble_predict_func.py').read())

app = Flask(__name__)

# 加载模型和相关组件
tfidf_pipeline = pickle.load(open('../../models/best_tfidf_model.pkl', 'rb'))
lstm_model = tf.keras.models.load_model('../../models/lstm_model')
tokenizer = pickle.load(open('../../models/tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('../../models/label_encoder.pkl', 'rb'))

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>新闻分类器</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        textarea {
            width: 100%;
            height: 200px;
            margin-bottom: 10px;
            padding: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .category {
            font-weight: bold;
            font-size: 1.2em;
            color: #333;
        }
        .probability {
            margin-top: 10px;
        }
        .bar {
            height: 20px;
            background-color: #2196F3;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <h1>新闻分类器</h1>
    <p>请输入新闻文本:</p>
    <form method="POST">
        <textarea name="news_text" required>{{ text }}</textarea>
        <button type="submit">分类</button>
    </form>
    
    {% if category %}
    <div class="result">
        <p>分类结果: <span class="category">{{ category }}</span></p>
        <div class="probability">
            {% for cat, prob in probabilities %}
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 100px;">{{ cat }}:</div>
                <div style="flex-grow: 1; display: flex; align-items: center;">
                    <div class="bar" style="width: {{ prob*100 }}%;"></div>
                    <div style="margin-left: 10px;">{{ "%.2f"|format(prob*100) }}%</div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    category = None
    probabilities = []
    text = ""
    
    if request.method == 'POST':
        text = request.form['news_text']
        category, probs = ensemble_predict(text, tfidf_pipeline, lstm_model, tokenizer, label_encoder)
        
        # 准备概率数据用于显示
        categories = label_encoder.classes_
        probabilities = [(cat, prob) for cat, prob in zip(categories, probs)]
        probabilities.sort(key=lambda x: x[1], reverse=True)
    
    return render_template_string(HTML_TEMPLATE, 
                                 category=category, 
                                 probabilities=probabilities,
                                 text=text)

@app.route('/api/classify', methods=['POST'])
def classify_api():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    category, probs = ensemble_predict(text, tfidf_pipeline, lstm_model, tokenizer, label_encoder)
    
    # 准备响应
    categories = label_encoder.classes_
    probabilities = {cat: float(prob) for cat, prob in zip(categories, probs)}
    
    return jsonify({
        'category': category,
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)