# app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from tqdm import tqdm
import datetime

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
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(zscores.index, zscores)
        ax1.axhline(0, color='black')
        ax1.axhline(1.0, color='red', linestyle='--')
        ax1.axhline(-1.0, color='green', linestyle='--')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Z-score')
        st.pyplot(fig1)

        # Calcul de l'exposant de Hurst
        h = calculer_hurst(S1.dropna())
        st.write(f"**Exposant de Hurst pour {crypto_selectionnee} :** {h:.4f}")

        # Application de la stratégie de trading par paires
        rendements_cumules, positions, zscore_series = strategie_trading_paires(
            S1, S2, window1=window1, window2=window2, entry_z=entry_z, exit_z=exit_z)

        st.subheader("Rendements cumulés de la stratégie")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(rendements_cumules.index, rendements_cumules)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Rendements cumulés')
        st.pyplot(fig2)

        st.subheader("Positions prises")
        st.line_chart(positions)

        st.subheader("Z-score du ratio des prix")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(zscore_series.index, zscore_series)
        ax3.axhline(0, color='black')
        ax3.axhline(entry_z, color='red', linestyle='--')
        ax3.axhline(-entry_z, color='green', linestyle='--')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Z-score du ratio')
        st.pyplot(fig3)

        st.success("Analyse terminée.")

if __name__ == '__main__':
    main()
