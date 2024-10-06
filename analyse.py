import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Classe pour les stratégies de co-intégration simplifiées
class CointegrationStrategies:
    def __init__(self, data):
        self.data = data
        self.pairs = []
        self.coverage_ratios = {}
        self.spreads = {}

    def identify_cointegrated_pairs(self):
        """
        Identifie les paires d'actifs ayant une relation de co-intégration.
        """
        st.write("### Identification des paires d'actifs co-intégrées 📊")
        for i in range(len(self.data.columns)):
            for j in range(i + 1, len(self.data.columns)):
                asset1 = self.data.columns[i]
                asset2 = self.data.columns[j]
                _, pvalue, _ = coint(self.data[asset1], self.data[asset2])
                if pvalue < 0.05:
                    self.pairs.append((asset1, asset2))
                    st.write(f"**Paire détectée : {asset1} et {asset2} (p-value={pvalue:.4f})**")

    def calculate_hedge_ratios(self):
        """
        Calcule le ratio de couverture pour chaque paire détectée.
        """
        st.write("### Calcul des ratios de couverture (régression linéaire) 📉")
        for asset1, asset2 in self.pairs:
            model = OLS(self.data[asset1], sm.add_constant(self.data[asset2])).fit()
            self.coverage_ratios[(asset1, asset2)] = model.params[1]
            st.write(f"Ratio de couverture pour {asset1}/{asset2} : {model.params[1]:.4f}")
            
            # Tracé de la droite de régression
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data[asset2], y=self.data[asset1], mode='markers', name='Données'))
            fig.add_trace(go.Scatter(x=self.data[asset2], y=model.fittedvalues, mode='lines', name='Régression Linéaire'))
            fig.update_layout(title=f"Régression Linéaire : {asset1} vs {asset2}", xaxis_title=asset2, yaxis_title=asset1)
            st.plotly_chart(fig)

    def calculate_spreads(self):
        """
        Calcule la différence (spread) entre deux actifs co-intégrés pour détecter des opportunités de trading.
        """
        st.write("### Calcul des spreads (écarts) entre les actifs 📉")
        for (asset1, asset2), ratio in self.coverage_ratios.items():
            self.spreads[(asset1, asset2)] = self.data[asset1] - ratio * self.data[asset2]
            st.write(f"Spread calculé pour la paire {asset1}/{asset2}")

    def generate_signals(self):
        """
        Génère des signaux de trading en fonction du comportement du spread.
        """
        st.write("### Signaux de Trading 🚦")
        for (asset1, asset2), spread in self.spreads.items():
            z_score = (spread - spread.mean()) / spread.std()
            buy_signal = z_score < -2
            sell_signal = z_score > 2

            st.write(f"**Signaux de Trading pour {asset1}/{asset2} :**")
            st.write(f"Signaux d'achat : {buy_signal.sum()}, Signaux de vente : {sell_signal.sum()}")

# Classe principale pour gérer l'analyse et la préparation des données
class SimpleCryptoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.returns = None

    def fetch_data(self):
        """
        Télécharge les données historiques des actifs.
        """
        st.markdown("### Téléchargement des données 📊")
        data_dict = {}
        missing_tickers = []

        for ticker, name in zip(self.tickers, self.names):
            try:
                data = yf.download(ticker, start=self.start_date)['Adj Close'].dropna()
                data_dict[ticker] = data
            except Exception as e:
                missing_tickers.append(ticker)

        if data_dict:
            self.data = pd.DataFrame(data_dict)
            self.returns = self.data.pct_change().dropna()

        if missing_tickers:
            st.error(f"Les tickers suivants n'ont pas pu être téléchargés : {missing_tickers}")
        else:
            st.success("Toutes les données ont été téléchargées avec succès!")

    def prepare_data(self):
        """
        Prépare soigneusement les données :
        - Gestion des valeurs manquantes
        - Vérification de la stationnarité
        - Transformation des données
        - Standardisation
        """
        st.markdown("### Préparation des données 📊")
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.scale_data()
        st.success("Préparation des données terminée.")

    def handle_missing_data(self):
        """
        Gère les valeurs manquantes en utilisant une interpolation linéaire.
        """
        self.data = self.data.interpolate(method='linear').dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        """
        Vérifie si les données sont stationnaires à l'aide du test ADF.
        """
        st.write("**Vérification de la stationnarité**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire.")
            else:
                st.success(f"La série pour {column} est stationnaire.")

    def make_stationary(self):
        """
        Rends les données stationnaires en appliquant la différenciation.
        """
        self.returns = self.data.diff().dropna()

    def scale_data(self):
        """
        Standardisation des données pour les mettre à la même échelle.
        """
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        st.write("**Données standardisées**")

    def random_forest_model(self):
        """
        Crée un modèle prédictif pour Bitcoin basé sur les autres cryptomonnaies.
        """
        st.markdown("### Modèle de prédiction 🌲")
        st.write("Nous utilisons un modèle pour prédire les rendements du Bitcoin.")

        # Préparation des données
        X = self.returns.drop(columns=['BTC-USD'])
        y = self.returns['BTC-USD']

        # Modèle de Forêt Aléatoire
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Prédiction et évaluation
        y_pred = rf_model.predict(X)
        r_squared = r2_score(y, y_pred)
        st.write(f"**R² du modèle Forêt Aléatoire :** {r_squared:.3f}")
        st.write("Plus le R² est proche de 1, plus le modèle est performant.")

        # Importance des variables
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        st.write("### Importance des Variables dans le Modèle")
        st.write(feature_importances.sort_values(ascending=False))

def main():
    st.title("🔍 Analyse Simplifiée des Cryptomonnaies")
    st.write("Cette application permet d'explorer les relations entre Bitcoin et d'autres cryptomonnaies, et de générer des signaux de trading basés sur des stratégies simplifiées.")

    # Liste des tickers et noms réels des cryptomonnaies
    tickers = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'DOT-USD', 'AVAX-USD'
    ]
    
    # Noms réels des actifs financiers correspondants
    names = [
        'Bitcoin (BTC)', 'Ethereum (ETH)', 'Binance Coin (BNB)', 'Cardano (ADA)', 'Solana (SOL)',
        'Ripple (XRP)', 'Dogecoin (DOGE)', 'Polkadot (DOT)', 'Avalanche (AVAX)'
    ]

    start_date = '2018-01-01'

    analyzer = SimpleCryptoAnalyzer(tickers, names, start_date)

    st.header('1. Téléchargement et Préparation des Données')
    analyzer.fetch_data()
    analyzer.prepare_data()

    st.header('2. Modèle Prédictif Bitcoin')
    analyzer.random_forest_model()

    st.header('3. Stratégies de Co-Intégration')
    cointegration = CointegrationStrategies(analyzer.data)
    cointegration.identify_cointegrated_pairs()
    cointegration.calculate_hedge_ratios()
    cointegration.calculate_spreads()
    cointegration.generate_signals()

if __name__ == "__main__":
    main()
