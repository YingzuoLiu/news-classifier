def ensemble_predict(text, tfidf_pipeline, lstm_model, tokenizer, label_encoder, max_len=1000):
    # 预处理文本
    processed_text = preprocess_text(text)
    
    # TF-IDF模型预测
    tfidf_model = tfidf_pipeline.named_steps['clf']
    tfidf_vectorizer = tfidf_pipeline.named_steps['tfidf']
    X_tfidf = tfidf_vectorizer.transform([processed_text])
    
    if hasattr(tfidf_model, 'predict_proba'):
        tfidf_probs = tfidf_model.predict_proba(X_tfidf)[0]
    else:
        decision_values = tfidf_model.decision_function(X_tfidf)[0]
        tfidf_probs = np.exp(decision_values) / np.sum(np.exp(decision_values))
    
    # LSTM模型预测
    sequences = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequences, maxlen=max_len)
    lstm_probs = lstm_model.predict(padded)[0]
    
    # 融合预测
    ensemble_probs = (tfidf_probs + lstm_probs) / 2
    ensemble_pred = np.argmax(ensemble_probs)
    
    # 转换回类别名称
    return label_encoder.inverse_transform([ensemble_pred])[0], ensemble_probs

def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 删除标点和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 简单分词
    words = text.split()
    # 删除停用词和词形还原
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)
