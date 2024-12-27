import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Fungsi untuk membuat navigasi sidebar
def sidebar_navigation():
    st.sidebar.title("Navigasi")
    sections = ["🏠 Beranda", "🔧 Preprocessing", "📈 Hasil Evaluasi Model"]
    selected_section = st.sidebar.radio("Pilih Bagian:", sections)
    return selected_section

# Fungsi Neural Network
def create_neural_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse', metrics=['mse'])
    return model

# Fungsi evaluasi model
def evaluate_model(model, X_train, y_train, X_test, y_test, is_neural_network=False):
    if is_neural_network:
        model.fit(X_train, y_train, epochs=100)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred) * 100
    mse = mean_squared_error(y_test, y_pred) * 100
    return r2, mse

def evaluate_model2(model, X_train, y_train, is_neural_network=False):
    if is_neural_network:
        model.fit(X_train, y_train, epochs=100)
        y_pred = model.predict(X_train)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred) * 100
    mse = mean_squared_error(y_train, y_pred) * 100
    return r2, mse

# Evaluasi semua model
def evaluate_all_models(X_train, X_test, y_train, y_test):
    results = []
    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(),
        "Neural Network": create_neural_network()
    }
    for name, model in models.items():
        is_neural_network = name == "Neural Network"
        r2, mse = evaluate_model(model, X_train, y_train, X_test, y_test, is_neural_network)
        results.append({"Model": name, "R2 Score (%)": r2, "MSE (%)": mse})
    return pd.DataFrame(results)

def evaluate_all_models2(X_train, y_train):
    results = []
    models = {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(),
        "Neural Network": create_neural_network()
    }
    for name, model in models.items():
        is_neural_network = name == "Neural Network"
        r2, mse = evaluate_model2(model, X_train, y_train, is_neural_network)
        results.append({"Model": name, "R2 Score (%)": r2, "MSE (%)": mse})
    return pd.DataFrame(results)

# Sidebar navigation
section = sidebar_navigation()

# Beranda
if section == "🏠 Beranda":
    st.title("Prediksi Performa Pelajar")
    st.markdown("Aplikasi ini digunakan untuk memprediksi performa pelajar berdasarkan fitur-fitur yang ada.")
    st.header("📊 Data Mentah")
    data = pd.read_csv("Performa-Pelajar-Dataset.csv")
    st.dataframe(data.head(10))

# Preprocessing
elif section == "🔧 Preprocessing":
    st.header("🔧 Preprocessing Data")
    data = pd.read_csv("Performa-Pelajar-Dataset.csv")
    data = pd.get_dummies(data, columns=['Extracurricular Activities'], dtype=int)
    st.subheader("Encoding Data")
    st.markdown("Dilakukan transformasi data agar fitur yang bertipe kategori diubah menjadi numerik supaya tidak menimbulkan masalah pada algoritma metode yang akan dibangun, berikut data yang telah diencoding")
    st.dataframe(data.head(10))
    X = data.drop(columns=['Performance Index'])
    y = data['Performance Index']

    st.subheader("Standarisasi Data")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    st.markdown("Data Setelah Standarisasi:")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head(10))

# Hasil Evaluasi Model
elif section == "📈 Hasil Evaluasi Model":
    st.header("📈 Hasil Evaluasi Model")
    data = pd.read_csv("Performa-Pelajar-Dataset.csv")
    data = pd.get_dummies(data, columns=['Extracurricular Activities'], dtype=int)
    X = data.drop(columns=['Performance Index'])
    y = data['Performance Index']

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    ratios = [0.1, 0.2, 0.3]
    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=ratio, random_state=42)
        results_train = evaluate_all_models2(X_train, y_train)
        results_test = evaluate_all_models(X_train, X_test, y_train, y_test)
        st.subheader(f"Pembagian Data {int((1-ratio)*100)}:{int(ratio*100)}")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Tabel Evaluasi Data Train")
            st.dataframe(results_train)
        with col2:
            st.write("Tabel Evaluasi Data Test")
            st.dataframe(results_test)