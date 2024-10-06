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
from statsmodels.regression.linear_model import OLS

st.set_page_config(page_title="Analyse Crypto", page_icon="📊", layout="wide")

class CryptoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.returns = None

    def fetch_data(self):
        st.write("## Téléchargement des données 📥")
        data_dict = {}
        for ticker, name in zip(self.tickers, self.names):
            try:
                data = yf.download(ticker, start=self.start_date)['Adj Close'].dropna()
                data_dict[ticker] = data
            except Exception as e:
                st.warning(f"Impossible de télécharger les données pour {name} ({ticker})")
        
        if data_dict:
            self.data = pd.DataFrame(data_dict)
            self.returns = self.data.pct_change().dropna()
            st.success("Données téléchargées avec succès !")
        else:
            st.error("Aucune donnée n'a pu être téléchargée.")

    def prepare_data(self):
        st.write("## Préparation des données 🔧")
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()
        st.success("Données préparées avec succès !")

    def handle_missing_data(self, method='linear'):
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        st.write("**Vérification de la stationnarité**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire.")
            else:
                st.success(f"La série pour {column} est stationnaire.")

    def make_stationary(self):
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        st.write("**Détection des outliers**")
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        st.write(f"Nombre d'outliers détectés : {outliers.sum().sum()}")

    def scale_data(self):
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        
        min_max_scaler = MinMaxScaler()
        self.data = pd.DataFrame(min_max_scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        
        st.write("**Les données ont été standardisées et mises à la même échelle**")

    def analyze_cointegration(self):
        st.write("## Analyse de co-intégration 🔍")
        btc_col = 'BTC-USD'
        cointegrated_pairs = []
        for col in self.data.columns:
            if col != btc_col:
                _, pvalue, _ = coint(self.data[btc_col], self.data[col])
                if pvalue < 0.01:
                    cointegrated_pairs.append((btc_col, col))
                    st.write(f"**{col} est co-intégré avec Bitcoin (p-value={pvalue:.4f})**")
        return cointegrated_pairs

    def generate_trading_signals(self, cointegrated_pairs):
        st.write("## Génération des signaux de trading 🚦")
        for pair in cointegrated_pairs:
            btc_col, other_col = pair
            spread = self.data[btc_col] - self.data[other_col]
            z_score = (spread - spread.mean()) / spread.std()
            
            buy_signal = z_score < -2
            sell_signal = z_score > 2
            
            st.write(f"**Signaux pour {btc_col}/{other_col} :**")
            st.write(f"Nombre de signaux d'achat : {buy_signal.sum()}, Nombre de signaux de vente : {sell_signal.sum()}")
            
            edge_ratio = (sell_signal.sum() - buy_signal.sum()) / (sell_signal.sum() + buy_signal.sum())
            position_size = abs(edge_ratio) * 100
            st.write(f"Taille de position suggérée : {position_size:.2f}%")

    def predict_bitcoin(self):
        st.write("## Prédiction des mouvements de Bitcoin 🔮")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        st.write(f"**Précision du modèle :** {r_squared:.1%}")
        if r_squared > 0.7:
            st.success("🎉 Le modèle est très performant pour prédire les mouvements du Bitcoin !")
        else:
            st.warning("🤔 Le modèle a une capacité limitée à prédire les mouvements du Bitcoin.")

    def visualize_data(self, cointegrated_pairs):
        st.write("## Visualisation des prix 📈")
        for pair in cointegrated_pairs:
            btc_col, other_col = pair
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[btc_col], mode='lines', name='Bitcoin'))
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[other_col], mode='lines', name=self.names[self.tickers.index(other_col)]))
            fig.update_layout(title=f"Bitcoin vs {self.names[self.tickers.index(other_col)]}", xaxis_title="Date", yaxis_title="Prix (normalisé)")
            st.plotly_chart(fig)

def main():
    st.title("📊 Analyse des Relations entre Bitcoin et Autres Cryptomonnaies")

    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    
    analyzer = CryptoAnalyzer(tickers, names, start_date)

    st.sidebar.header("📌 Navigation")
    page = st.sidebar.radio("Choisissez une section :", ["Préparation des données", "Analyse de co-intégration", "Signaux de trading", "Prédictions", "Visualisation"])

    if page == "Préparation des données":
        analyzer.fetch_data()
        analyzer.prepare_data()

    elif page == "Analyse de co-intégration":
        analyzer.fetch_data()
        analyzer.prepare_data()
        cointegrated_pairs = analyzer.analyze_cointegration()

    elif page == "Signaux de trading":
        analyzer.fetch_data()
        analyzer.prepare_data()
        cointegrated_pairs = analyzer.analyze_cointegration()
        analyzer.generate_trading_signals(cointegrated_pairs)

    elif page == "Prédictions":
        analyzer.fetch_data()
        analyzer.prepare_data()
        analyzer.predict_bitcoin()

    elif page == "Visualisation":
        analyzer.fetch_data()
        analyzer.prepare_data()
        cointegrated_pairs = analyzer.analyze_cointegration()
        analyzer.visualize_data(cointegrated_pairs)

if __name__ == "__main__":
    main()
