import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import plotly.express as px

# Modelleri yükleme
with open("logistic_regression.pkl", "rb") as f:
    logistic_model = pickle.load(f)

with open("random_forest.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

with open("xgboost.pkl", "rb") as f:
    xgboost_model = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

deep_learning_model = load_model("deep_learning_model.h5")  # Derin öğrenme modeli

# FastText vektörleri yükleme
with open("fasttext_vectors.pkl", "rb") as file:
    data = pickle.load(file)
    data_vectors, data_titles = data["vectors"], data["titles"]  # X: Vektörler, titles: Başlıklar

# Kullanıcıdan haber başlığına karşılık gelen bir vektör elde etmek için arama fonksiyonu
def get_vector_from_title(title, data_titles, data_vectors):
    """
    Girilen başlığa en yakın başlığı bulur ve onun vektörünü döner.
    """
    index = np.argmin([len(set(title.split()) - set(t.split())) for t in data_titles])
    return data_vectors[index]

# Uygulama başlığı
st.title("Haber Sınıflandırma Uygulaması")
st.write("Haber başlıklarını farklı modellerle sınıflandırabilirsiniz.")

# Kullanıcıdan haber başlığını alma
headline = st.text_input("Haber Başlığını Giriniz:")
model_choice = st.selectbox(
    "Bir model seçiniz:",
    ["Logistic Regression", "KMeans", "Random Forest", "XGBoost", "Deep Learning"]
)

if headline:
    # Girilen başlık için vektör bulma
    vectorized_headline = get_vector_from_title(headline, data_titles, data_vectors).reshape(1, -1)
    prediction, probabilities = None, None

    # Model seçimine göre tahmin
    if model_choice == "Logistic Regression":
        prediction = logistic_model.predict(vectorized_headline)
        probabilities = logistic_model.predict_proba(vectorized_headline)

    elif model_choice == "KMeans":
        prediction = kmeans_model.predict(vectorized_headline)
        probabilities = None  # KMeans predict_proba desteği yok

    elif model_choice == "Random Forest":
        prediction = random_forest_model.predict(vectorized_headline)
        probabilities = random_forest_model.predict_proba(vectorized_headline)

    elif model_choice == "XGBoost":
        prediction = xgboost_model.predict(vectorized_headline)
        probabilities = xgboost_model.predict_proba(vectorized_headline)

    elif model_choice == "Deep Learning":
        probabilities = deep_learning_model.predict(vectorized_headline)
        prediction = np.argmax(probabilities, axis=1)

    # Tahmini ve olasılıkları gösterme
    st.write(f"Seçilen model: **{model_choice}**")
    st.write(f"Modelin tahmini: **Class {prediction[0]}**")
    if probabilities is not None:
        st.write("Tahmin olasılıkları:")
        prob_df = pd.DataFrame({"Class": [f"Class {i}" for i in range(probabilities.shape[1])],
                                "Probability": probabilities[0]})
        st.dataframe(prob_df)

        # Olasılık görselleştirme
        fig = px.bar(prob_df, x="Class", y="Probability", title="Prediction Probabilities", color="Class")
        st.plotly_chart(fig)
else:
    st.warning("Lütfen bir haber başlığı giriniz.")

# Tüm modellerin karşılaştırması
if st.button("Tüm Modellerin Karşılaştırması"):
    if headline:
        vectorized_headline = get_vector_from_title(headline, data_titles, data_vectors).reshape(1, -1)
        results = {}

        results["Logistic Regression"] = logistic_model.predict(vectorized_headline)[0]
        results["KMeans"] = kmeans_model.predict(vectorized_headline)[0]
        results["Random Forest"] = random_forest_model.predict(vectorized_headline)[0]
        results["XGBoost"] = xgboost_model.predict(vectorized_headline)[0]
        deep_pred = deep_learning_model.predict(vectorized_headline)
        results["Deep Learning"] = np.argmax(deep_pred, axis=1)[0]

        st.write("Tüm modellerin sonuçları:")
        st.table(pd.DataFrame(results.items(), columns=["Model", "Prediction"]))
    else:
        st.warning("Lütfen bir haber başlığı giriniz.")