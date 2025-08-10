import streamlit as st
import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
import joblib
import os

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Predictor S&P 500",
    page_icon="📈",
    layout="wide"
)

# --- Funciones Cacheadas para Eficiencia ---

@st.cache_data(ttl=3600) # La caché de datos expira cada hora
def load_and_prepare_data():
    """Descarga y prepara todos los datos necesarios para el modelo."""
    # S&P 500
    sp500 = yf.Ticker("^GSPC").history(period="max")
    sp500.index = sp500.index.tz_localize(None)
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()

    # Nikkei 225
    nikkei = yf.Ticker("^N225").history(period="max")
    nikkei.index = nikkei.index.tz_localize(None)
    sp500 = sp500.join(nikkei[['Close']].rename(columns={'Close': 'Nikkei_Close'}), how='left')
    
    # Tasa de Interés (Bono a 10 años)
    tnx = yf.Ticker("^TNX").history(period="max")
    tnx.index = tnx.index.tz_localize(None)
    sp500 = sp500.join(tnx[['Close']].rename(columns={'Close': 'Interest_Rate'}), how='left')

    # Rellenar y limpiar NaN
    sp500.ffill(inplace=True)
    
    # Crear predictores
    horizons = [2, 5, 60, 250, 1000]
    predictors = []
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        predictors += [ratio_column, trend_column]
        
    sp500['Nikkei_Ratio_5'] = sp500['Nikkei_Close'] / sp500['Nikkei_Close'].rolling(5).mean()
    predictors.append('Nikkei_Ratio_5')
    predictors.append('Interest_Rate')
    
    sp500.dropna(inplace=True)
    return sp500, predictors

@st.cache_resource
def train_and_save_model(data, predictors):
    """Entrena el modelo XGBoost con todos los datos y lo guarda."""
    model_path = "xgboost_model.joblib"
    
    model = XGBClassifier(random_state=1, n_estimators=200, learning_rate=0.05, eval_metric='logloss')
    # Entrenamos con todos los datos disponibles para tener el modelo más actualizado
    model.fit(data[predictors], data["Target"])
    joblib.dump(model, model_path)
    return model

# --- Construcción de la Interfaz de Usuario ---

st.title("🤖 Predictor del S&P 500 con Machine Learning")
st.write("""
Esta aplicación utiliza un modelo **XGBoost** para predecir si el precio del índice S&P 500 subirá o bajará al día siguiente. 
El modelo se ha entrenado con datos históricos desde 1990 y se ha enriquecido con indicadores del mercado japonés y las tasas de interés de EE. UU.
""")

# Cargar datos y modelo
try:
    with st.spinner('Cargando datos y entrenando el modelo...'):
        data, predictors = load_and_prepare_data()
        model = train_and_save_model(data, predictors)

    # --- Mostrar la Predicción para Mañana ---
    st.header("🔮 Predicción para el Próximo Día de Mercado")

    last_data_point = data.iloc[-1:][predictors]
    prediction_proba = model.predict_proba(last_data_point)[0, 1]
    prediction = 1 if prediction_proba >= 0.6 else 0

    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.metric(label="Predicción", value="SUBIDA 📈", delta="Positivo")
        else:
            st.metric(label="Predicción", value="BAJADA 📉", delta="Negativo", delta_color="inverse")
    with col2:
        st.metric(label="Confianza del Modelo", value=f"{prediction_proba*100:.2f}%")

    st.info("Nota: La predicción es 'SUBIDA' solo si la confianza del modelo es superior al 60%.", icon="ℹ️")


    # --- Mostrar Rendimiento del Modelo y Datos Históricos ---
    st.header("📊 Rendimiento y Datos")

    # Gráfico de precios históricos
    st.subheader("Evolución Histórica del S&P 500")
    st.line_chart(data["Close"], use_container_width=True)

    # Mostrar los datos más recientes
    with st.expander("Ver los datos más recientes utilizados para la predicción"):
        st.dataframe(data.tail(10))

except Exception as e:
    st.error(f"Ha ocurrido un error al cargar los datos o el modelo: {e}")
    st.warning("Por favor, asegúrate de tener conexión a internet.")
