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

st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">📊 Analyse Crypto</p>', unsafe_allow_html=True)

class CryptoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.returns = None

    def fetch_data(self):
        st.write("## Téléchargement des données 📥")
        st.write("Nous collectons les prix historiques des cryptomonnaies.")
        
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
        st.write("Nous nettoyons et préparons les données pour l'analyse.")
        
        # Gestion des valeurs manquantes
        self.data = self.data.interpolate(method='linear').dropna()
        self.returns = self.data.pct_change().dropna()
        
        # Standardisation des données
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        
        st.success("Données préparées avec succès !")

    def analyze_relationships(self):
        st.write("## Analyse des relations 🔍")
        st.write("Nous examinons les liens entre Bitcoin et les autres cryptomonnaies.")
        
        btc_col = 'BTC-USD'
        for col in self.data.columns:
            if col != btc_col:
                correlation = self.data[btc_col].corr(self.data[col])
                st.write(f"**Corrélation entre Bitcoin et {self.names[self.tickers.index(col)]} :** {correlation:.2f}")
        
        st.success("Analyse des relations terminée !")

    def predict_bitcoin(self):
        st.write("## Prédiction des mouvements de Bitcoin 🔮")
        st.write("Nous utilisons un modèle pour prédire les variations du prix du Bitcoin.")
        
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

    def visualize_data(self):
        st.write("## Visualisation des prix 📈")
        st.write("Comparons l'évolution des prix des différentes cryptomonnaies.")
        
        selected_crypto = st.selectbox("Choisissez une cryptomonnaie à comparer avec Bitcoin :", self.names[1:])
        selected_ticker = self.tickers[self.names.index(selected_crypto)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BTC-USD'], mode='lines', name='Bitcoin'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data[selected_ticker], mode='lines', name=selected_crypto))
        fig.update_layout(title=f"Bitcoin vs {selected_crypto}", xaxis_title="Date", yaxis_title="Prix")
        st.plotly_chart(fig)

def main():
    st.sidebar.header("📌 Navigation")
    page = st.sidebar.radio("Choisissez une section :", ["Accueil", "Analyse", "Prédictions", "Visualisation"])

    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    
    analyzer = CryptoAnalyzer(tickers, names, start_date)

    if page == "Accueil":
        st.write("## Bienvenue dans l'analyseur de cryptomonnaies ! 👋")
        st.write("Cet outil vous aide à comprendre les relations entre Bitcoin et d'autres cryptomonnaies.")
        st.info("👈 Utilisez le menu à gauche pour naviguer entre les différentes sections.")

    elif page == "Analyse":
        analyzer.fetch_data()
        analyzer.prepare_data()
        analyzer.analyze_relationships()

    elif page == "Prédictions":
        analyzer.fetch_data()
        analyzer.prepare_data()
        analyzer.predict_bitcoin()

    elif page == "Visualisation":
        analyzer.fetch_data()
        analyzer.visualize_data()

if __name__ == "__main__":
    main()
