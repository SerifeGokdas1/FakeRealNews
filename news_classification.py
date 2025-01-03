import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import re

# FastText tüm vektörlerini yüklemek için cache
@st.cache_resource
def load_full_fasttext_vectors():
    with open("fasttext_vectors.pkl", "rb") as file:
        full_word_vectors = pickle.load(file)  # Tüm kelime vektörlerini yükle
    return full_word_vectors

# Orijinal FastText vektörleri (2 milyon kelime)
full_word_vectors = load_full_fasttext_vectors()

# PKL dosyasını yüklemek için cache (eğitim verisi)
@st.cache_resource
def load_fasttext_dataset():
    with open("fasttext_vectors.pkl", "rb") as file:
        X, y = pickle.load(file)
    return np.array(X), np.array(y)

# Veri ve etiketleri yükleme
X, y = load_fasttext_dataset()

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d", "", text)  # Rakamları kaldır
    text = re.sub(r"[\'\",\.\?\!\(\)\-]", "", text)  # Noktalama işaretlerini kaldır
    text = re.sub(r"\s+", " ", text).strip()  # Fazladan boşlukları kaldır
    return text

# Haber başlığını vektörleştirme
def vectorize_input(user_input, full_word_vectors, embedding_dim=300):
    tokens = user_input.lower().split()
    vectorized_input = np.zeros(embedding_dim)
    count = 0
    for word in tokens:
        if word in full_word_vectors:
            vectorized_input += full_word_vectors[word]
            count += 1
    if count == 0:
        st.error("Girilen başlık, vektörler içinde eşleşen kelimeler içermiyor. Daha farklı bir başlık giriniz.")
        return None
    return vectorized_input / count if count > 0 else vectorized_input

# Veri setini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımlama ve eğitme
logreg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(random_state=42)
xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
kmeans = KMeans(n_clusters=2, random_state=42)

logreg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
xgboost.fit(X_train, y_train)
kmeans.fit(X_train)

# Streamlit uygulaması
st.title("Doğru-Yanlış Haber Tahmini")
st.write("Kullanıcıdan haber başlığı alınır ve seçilen model ile tahmin yapılır.")

# Kullanıcıdan haber başlığını alma
user_input = st.text_input("Haber başlığını girin:")

# Sınıflandırıcı seçimi
classifier_name = st.selectbox(
    "Bir sınıflandırıcı seçin:",
    ["Logistic Regression", "Random Forest", "XGBoost", "KMeans"]
)

# Tahmin butonu
tahmin_butonu = st.button("Tahmini Yap")

if tahmin_butonu and user_input:
    # Haber başlığını ön işleme
    preprocessed_input = preprocess_text(user_input)

    # Haber başlığını vektörleştir
    embedding_dim = 300  # FastText'in sabit boyutlu vektör boyutu
    vectorized_input = vectorize_input(preprocessed_input, full_word_vectors, embedding_dim)

    if vectorized_input is not None:  # None döndüyse işlem yapma
        vectorized_input = vectorized_input.reshape(1, -1)
    else:
        st.stop()  # Hata durumunda uygulamayı durdur

    # Model seçimi ve tahmin
    models = {
        "Logistic Regression": logreg,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
        "KMeans": kmeans
    }
    predictions = {}
    for name, model in models.items():
        if name == "KMeans":
            pred = model.predict(vectorized_input)
            pred = int(pred[0] == 1)  # KMeans küme etiketini sınıf olarak eşleştiriyor
        else:
            pred = model.predict(vectorized_input)
        predictions[name] = "DOĞRU" if pred[0] == 1 else "YANLIŞ"

    # Seçilen modelin tahmini
    selected_model_prediction = predictions[classifier_name]
    st.subheader(f"Seçilen Model ({classifier_name}) Tahmini: {selected_model_prediction}")

    # Tüm modellerin sonuçları
    st.subheader("Tüm Modellerin Tahminleri")
    comparison_df = pd.DataFrame(predictions.items(), columns=["Model", "Tahmin"])
    st.table(comparison_df)

    # Görselleştirme
    st.subheader("Confusion Matrix ve Karşılaştırma")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (name, model) in enumerate(models.items()):
        if name != "KMeans":
            cm = confusion_matrix(y_test, model.predict(X_test))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f"{name} - Confusion Matrix")
            axes[idx].set_xlabel("Tahmin")
            axes[idx].set_ylabel("Gerçek")
    st.pyplot(fig)
