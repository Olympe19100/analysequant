import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import scipy.stats as stats

class CointegrationStrategies:
    def __init__(self, data):
        self.data = data

    def find_cointegrated_pairs(self):
        """
        Identification des paires coïntégrées dans les données.
        """
        n = self.data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = self.data.columns
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                S1 = self.data[keys[i]]
                S2 = self.data[keys[j]]
                result = coint(S1, S2)
                score_matrix[i, j] = result[0]
                pvalue_matrix[i, j] = result[1]
                if result[1] < 0.05:
                    pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs

    def backtest_strategy(self, pair):
        """
        Backtesting de la stratégie de coïntégration sur une paire donnée.
        """
        S1 = self.data[pair[0]]
        S2 = self.data[pair[1]]
        spread = S1 - S2
        zscore = (spread - spread.mean()) / spread.std()

        # Génération des signaux de trading
        buy_signal = zscore < -1
        sell_signal = zscore > 1

        # Calcul des rendements cumulatifs
        returns = S1.pct_change().fillna(0) - S2.pct_change().fillna(0)
        cumulative_returns = (1 + returns).cumprod()

        # Calcul du ratio de Sharpe
        sharpe_ratio = returns.mean() / returns.std()

        return cumulative_returns, sharpe_ratio

class ComprehensiveCryptoCommoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.returns = None

    def fetch_data(self):
        """
        Télécharge les données historiques pour les tickers spécifiés.
        """
        st.markdown("### Téléchargement des données historiques 📊")
        data_dict = {}
        missing_tickers = []
        for ticker, name in zip(self.tickers, self.names):
            try:
                data = yf.download(ticker, start=self.start_date)['Adj Close'].dropna()
                data_dict[ticker] = data
            except Exception as e:
                missing_tickers.append(ticker)
        self.data = pd.DataFrame(data_dict)
        self.returns = self.data.pct_change().dropna()

    def prepare_data(self):
        """
        Prépare soigneusement les données :
        - Gestion des valeurs manquantes
        - Vérification de la stationnarité
        - Transformation pour rendre les séries stationnaires
        - Détection et gestion des outliers
        - Standardisation
        """
        st.markdown("### Préparation des Données 🔧")
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()

    def handle_missing_data(self, method='linear'):
        """
        Gère les valeurs manquantes dans les séries temporelles.
        """
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        """
        Effectue un test de Dickey-Fuller Augmenté (ADF) pour vérifier la stationnarité des séries temporelles.
        """
        st.write("**Vérification de la stationnarité des séries temporelles**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire.")
            else:
                st.success(f"La série pour {column} est stationnaire.")

    def make_stationary(self):
        """
        Applique une différenciation première pour rendre les séries temporelles stationnaires.
        """
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        """
        Détection des outliers dans les séries temporelles via la méthode IQR.
        """
        st.write("**Détection et gestion des outliers**")
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        st.write(f"Nombre d'outliers détectés : {outliers.sum().sum()}")

    def scale_data(self):
        """
        Standardisation des données et mise à la même échelle.
        """
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        min_max_scaler = MinMaxScaler()
        self.data = pd.DataFrame(min_max_scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)

    def analyze_cointegration_strategies(self):
        strategy = CointegrationStrategies(self.data)
        score_matrix, pvalue_matrix, cointegrated_pairs = strategy.find_cointegrated_pairs()

        st.write("Paires coïntégrées identifiées :")
        st.write(cointegrated_pairs)

        for pair in cointegrated_pairs:
            cumulative_returns, sharpe_ratio = strategy.backtest_strategy(pair)
            st.write(f"Paire : {pair}")
            st.write(f"Rendements cumulatifs : {cumulative_returns.iloc[-1]:.2f}")
            st.write(f"Ratio de Sharpe : {sharpe_ratio:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns, mode='lines', name='Rendements cumulatifs'))
            fig.update_layout(title=f"Rendements cumulatifs pour {pair[0]}-{pair[1]}", xaxis_title="Date", yaxis_title="Rendements cumulatifs")
            st.plotly_chart(fig)

def main():
    st.title("💡 Analyse des Relations entre Bitcoin et Autres Cryptomonnaies")
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana']
    start_date = '2018-01-01'

    analyzer = ComprehensiveCryptoCommoAnalyzer(tickers, names, start_date)

    st.header('1. Téléchargement et Préparation des Données')
    analyzer.fetch_data()

    st.header('2. Préparation rigoureuse des Données')
    analyzer.prepare_data()

    st.header('3. Stratégies de Trading Basées sur la Coïntégration')
    analyzer.analyze_cointegration_strategies()

if __name__ == "__main__":
    main()
