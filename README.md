# 📈 Predictor del Mercado de Valores con Machine Learning y Streamlit

## Descripción del Proyecto
Este proyecto utiliza un modelo de Machine Learning (XGBoost) para predecir la dirección diaria (subida o bajada) del índice S&P 500. El modelo se entrena con datos históricos de precios y se enriquece con predictores externos como el comportamiento del índice Nikkei 225 y las tasas de interés de EE. UU.

Además del análisis en el Jupyter Notebook, se ha desarrollado una **aplicación web interactiva con Streamlit** que permite visualizar la predicción más reciente del modelo.


*(Aquí puedes añadir una captura de pantalla de tu app una vez esté funcionando)*

## 🚀 Progreso del Modelo
El proyecto siguió un proceso iterativo de mejora, añadiendo características y probando diferentes algoritmos.

| Modelo / Característica Añadida | Precisión Obtenida |
| --------------------------------- | ------------------- |
| Random Forest (Base)              | 54.2%               |
| XGBoost                           | 54.5%               |
| + Datos del Nikkei 225            | 54.8%               |
| + Tasa de Interés (Bono 10 años)  | **55.08%** |

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.10
* **Librerías Principales:**
    * **Análisis de Datos:** Pandas, yfinance
    * **Machine Learning:** Scikit-learn, XGBoost, joblib, Random Forest
    * **Frontend:** Streamlit
    * **Entorno:** Jupyter Notebook

## ⚙️ Cómo Ejecutar el Proyecto

Tienes dos formas de explorar este proyecto:

### 1. Análisis en Jupyter Notebook
Para explorar el proceso de construcción y evaluación del modelo:

1.  Clona este repositorio.
2.  Crea un entorno virtual e instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
3.  Abre el Jupyter Notebook:
    ```bash
    jupyter notebook main_presentable.ipynb
    ```

### 2. Ejecutar la Aplicación Web Interactiva
Para ver el modelo en acción y obtener la última predicción:

1.  Asegúrate de tener las dependencias instaladas (`pip install -r requirements.txt`).
2.  Ejecuta la aplicación desde tu terminal:
    ```bash
    streamlit run app.py
    ```

## 📄 Licencia
Distribuido bajo la licencia MIT. Ver `LICENSE` para más información.