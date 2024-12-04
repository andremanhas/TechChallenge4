import streamlit as st
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Carregar o modelo treinado
model_fit = joblib.load('sarimax_model.pkl')

# Título do app
st.title("Previsão de Série Temporal com SARIMAX")

# Carregar os dados
uploaded_file = st.file_uploader("Faça o upload de um arquivo CSV", type=["csv"])

if uploaded_file:
    # Ler o arquivo CSV
    data = pd.read_csv(uploaded_file, parse_dates=['data'], index_col='data')
    st.write("Dados carregados com sucesso!")
    st.write(data.head())
    
    # Exibir previsões
    steps = st.number_input("Número de períodos para prever:", min_value=1, max_value=10, value=5)
    
    if st.button("Fazer Previsões"):
        # Fazer previsões
        predictions = model_fit.forecast(steps=steps)
        st.write("Previsões:")
        st.write(predictions)
        
        # Gráfico de previsões
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['preco'], label='Histórico', color='blue')
        future_dates = pd.date_range(start=data.index[-1], periods=steps + 1, freq='D')[1:]
        plt.plot(future_dates, predictions, label='Previsões', color='red')
        plt.legend()
        st.pyplot(plt)
