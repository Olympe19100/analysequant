# app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from tqdm import tqdm
import datetime
import plotly.express as px
import plotly.graph_objects as go

# Désactiver les avertissements de pandas
import warnings
warnings.filterwarnings("ignore")

def obtenir_top_cryptomonnaies(n=20):
    """
    Récupère les symboles des n plus grandes cryptomonnaies.
    """
    top_cryptos = ['BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'DOGE', 'TRX', 'DOT', 'MATIC',
                   'LTC', 'SHIB', 'AVAX', 'LINK', 'UNI', 'ATOM', 'XMR', 'ETC', 'XLM', 'FIL']
    return top_cryptos[:n]

def telecharger_donnees_crypto(symboles, date_debut, date_fin):
    """
    Télécharge les données historiques pour les symboles de cryptomonnaies donnés depuis Yahoo Finance.
    """
    donnees = {}
    for symbole in tqdm(symboles, desc='Téléchargement des données de cryptomonnaies'):
        symbole_crypto = symbole + '-USD'
        df = yf.download(symbole_crypto, start=date_debut, end=date_fin)
        if not df.empty:
            donnees[symbole] = df['Close']
        else:
            st.warning(f"Aucune donnée pour {symbole}")
    return pd.DataFrame(donnees)

def zscore(serie):
    return (serie - serie.mean()) / np.std(serie)

def calculer_hurst(ts):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst_exponent = poly[0] * 2.0
    return hurst_exponent

def strategie_trading_paires(S1, S2, window1=5, window2=60, entry_z=1.5, exit_z=0.5):
    """
    Stratégie simple de trading par paires.
    """
    ratios = S1 / S2
    ma1 = ratios.rolling(window=window1).mean()
    ma2 = ratios.rolling(window=window2).mean()
    std = ratios.rolling(window=window2).std()
    zscore_series = (ma1 - ma2) / std

    # Signaux de trading
    longs = zscore_series < -entry_z
    shorts = zscore_series > entry_z
    exits = abs(zscore_series) < exit_z

    positions = pd.DataFrame(index=ratios.index)
    positions['S1'] = 0
    positions['S2'] = 0

    positions['S1'][longs] = 1
    positions['S1'][shorts] = -1
    positions['S1'][exits] = 0

    positions['S2'] = -positions['S1']

    positions['S1'] = positions['S1'].fillna(method='ffill')
    positions['S2'] = positions['S2'].fillna(method='ffill')

    rendements = positions.shift(1).multiply(S1.pct_change(), axis=0)['S1'] + \
                 positions.shift(1).multiply(S2.pct_change(), axis=0)['S2']

    rendements = rendements.fillna(0)
    rendements_cumules = (1 + rendements).cumprod() - 1

    return rendements_cumules, positions, zscore_series

def main():
    st.title("Analyse de Trading par Paires sur les Cryptomonnaies")
    st.markdown("""
    Cette application effectue une analyse de cointégration et applique une stratégie de trading par paires
    sur les 20 plus grandes cryptomonnaies associées à l'USDC.
    """)

    # Sélection de la période
    date_debut = st.date_input("Date de début", datetime.date(2020, 1, 1))
    date_fin = st.date_input("Date de fin", datetime.date.today())

    if date_debut >= date_fin:
        st.error("La date de début doit être antérieure à la date de fin.")
        return

    # Obtenir les 20 premières cryptomonnaies
    top_cryptos = obtenir_top_cryptomonnaies(20)

    # Sélection de la cryptomonnaie
    crypto_selectionnee = st.selectbox("Sélectionnez une cryptomonnaie", top_cryptos)

    # Paramètres de la stratégie
    st.sidebar.subheader("Paramètres de la Stratégie")
    window1 = st.sidebar.slider("Fenêtre MA1 (courte)", min_value=1, max_value=20, value=5)
    window2 = st.sidebar.slider("Fenêtre MA2 (longue)", min_value=20, max_value=120, value=60)
    entry_z = st.sidebar.slider("Seuil d'entrée Z-score", min_value=0.5, max_value=3.0, value=1.5)
    exit_z = st.sidebar.slider("Seuil de sortie Z-score", min_value=0.1, max_value=1.0, value=0.5)

    if st.button("Lancer l'analyse"):
        with st.spinner('Téléchargement des données...'):
            symboles = [crypto_selectionnee, 'USDC']
            data = telecharger_donnees_crypto(symboles, date_debut, date_fin)
            data = data.dropna()

        if data.empty or 'USDC' not in data.columns or crypto_selectionnee not in data.columns:
            st.error("Données insuffisantes pour effectuer l'analyse.")
            return

        S1 = data[crypto_selectionnee]
        S2 = data['USDC']

        # Calcul du z-score du prix
        zscores = zscore(S1)
        st.subheader(f"Z-score du prix de {crypto_selectionnee}")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=zscores.index, y=zscores, mode='lines', name='Z-score'))
        fig1.add_hline(y=0, line_dash="dash", line_color="black")
        fig1.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig1.add_hline(y=-1.0, line_dash="dash", line_color="green")
        fig1.update_layout(title=f"Z-score du prix de {crypto_selectionnee}", xaxis_title="Date", yaxis_title="Z-score")
        st.plotly_chart(fig1, use_container_width=True)

        # Calcul de l'exposant de Hurst
        h = calculer_hurst(S1.dropna())
        st.write(f"**Exposant de Hurst pour {crypto_selectionnee} :** {h:.4f}")

        # Application de la stratégie de trading par paires
        rendements_cumules, positions, zscore_series = strategie_trading_paires(
            S1, S2, window1=window1, window2=window2, entry_z=entry_z, exit_z=exit_z)

        st.subheader("Rendements cumulés de la stratégie")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rendements_cumules.index, y=rendements_cumules, mode='lines', name='Rendements cumulés'))
        fig2.update_layout(title="Rendements cumulés de la stratégie", xaxis_title="Date", yaxis_title="Rendements cumulés")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Positions prises")
        fig_positions = go.Figure()
        fig_positions.add_trace(go.Scatter(x=positions.index, y=positions['S1'], mode='lines', name=f"Position {crypto_selectionnee}"))
        fig_positions.add_trace(go.Scatter(x=positions.index, y=positions['S2'], mode='lines', name="Position USDC"))
        fig_positions.update_layout(title="Positions prises", xaxis_title="Date", yaxis_title="Position")
        st.plotly_chart(fig_positions, use_container_width=True)

        st.subheader("Z-score du ratio des prix")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=zscore_series.index, y=zscore_series, mode='lines', name='Z-score du ratio'))
        fig3.add_hline(y=0, line_dash="dash", line_color="black")
        fig3.add_hline(y=entry_z, line_dash="dash", line_color="red")
        fig3.add_hline(y=-entry_z, line_dash="dash", line_color="green")
        fig3.update_layout(title="Z-score du ratio des prix", xaxis_title="Date", yaxis_title="Z-score du ratio")
        st.plotly_chart(fig3, use_container_width=True)

        st.success("Analyse terminée.")

if __name__ == '__main__':
    main()
