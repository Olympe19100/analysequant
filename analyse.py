import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st

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
        for ticker in self.tickers:
            try:
                data = yf.download(ticker, start=self.start_date)['Adj Close']
                data_dict[ticker] = data
            except Exception as e:
                st.warning(f"Erreur lors du téléchargement de {ticker}: {e}")
        
        self.data = pd.DataFrame(data_dict).dropna()
        self.returns = self.data.pct_change().dropna()
        st.success(f"Données téléchargées pour {len(self.data.columns)} cryptomonnaies")

    def prepare_data(self):
        st.write("## Préparation des données 🔧")
        
        # Standardisation
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), 
                                    index=self.returns.index, 
                                    columns=self.returns.columns)
        
        # Mise à l'échelle des prix
        min_max_scaler = MinMaxScaler()
        self.data = pd.DataFrame(min_max_scaler.fit_transform(self.data), 
                                 index=self.data.index, 
                                 columns=self.data.columns)
        
        st.success("Données préparées avec succès")

    def analyze_cointegration(self):
        st.write("## Analyse de co-intégration 🔍")
        btc_col = 'BTC-USD'
        cointegrated_pairs = []
        for col in self.data.columns:
            if col != btc_col:
                _, pvalue, _ = coint(self.data[btc_col], self.data[col])
                if pvalue < 0.05:
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
            st.write(f"Signaux d'achat : {buy_signal.sum()}, Signaux de vente : {sell_signal.sum()}")
            
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
        
        st.write(f"**Précision du modèle :** {r_squared:.2%}")
        
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.write("**Importance des caractéristiques :**")
        st.write(feature_importance)

    def visualize_data(self, cointegrated_pairs):
        st.write("## Visualisation des prix 📈")
        for pair in cointegrated_pairs:
            btc_col, other_col = pair
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[btc_col], mode='lines', name='Bitcoin'))
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[other_col], mode='lines', name=other_col))
            fig.update_layout(title=f"Bitcoin vs {other_col}", xaxis_title="Date", yaxis_title="Prix (normalisé)")
            st.plotly_chart(fig)

def main():
    st.title("📊 Analyse des Relations entre Bitcoin et Autres Cryptomonnaies")

    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    
    analyzer = CryptoAnalyzer(tickers, names, start_date)

    analyzer.fetch_data()
    analyzer.prepare_data()
    cointegrated_pairs = analyzer.analyze_cointegration()
    analyzer.generate_trading_signals(cointegrated_pairs)
    analyzer.predict_bitcoin()
    analyzer.visualize_data(cointegrated_pairs)

if __name__ == "__main__":
    main()
