import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, coint
import plotly.graph_objects as go
import streamlit as st

class ComprehensiveCryptoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.scaled_data = None
        self.returns = None
        self.latest_prices = None
        self.cointegration_results = {}
        self.pairs = []

    def fetch_data(self):
        """Télécharge les données historiques pour toutes les cryptomonnaies."""
        st.markdown('<p class="subheader">Téléchargement des données historiques</p>', unsafe_allow_html=True)
        data_dict = {}
        missing_tickers = []
        for ticker, name in zip(self.tickers, self.names):
            try:
                data = yf.download(ticker, start=self.start_date)['Adj Close'].dropna()
                data_dict[ticker] = data
                st.write(f"Données téléchargées pour {name}")
            except Exception as e:
                missing_tickers.append(ticker)
                st.error(f"Erreur lors du téléchargement de {name}: {str(e)}")
        
        if data_dict:
            self.data = pd.DataFrame(data_dict)
            self.returns = self.data.pct_change().dropna()
            self.latest_prices = self.data.iloc[-1]
            st.success(f"Données téléchargées avec succès. Shape: {self.data.shape}")
        else:
            st.error("Aucune donnée n'a pu être téléchargée.")

    def prepare_data(self):
        """Prépare les données en gérant les valeurs manquantes, vérifiant la stationnarité, et normalisant les données."""
        st.markdown('<p class="subheader">Préparation des Données</p>', unsafe_allow_html=True)
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()
        st.success("Préparation des données terminée.")

    def handle_missing_data(self, method='linear'):
        """Gère les valeurs manquantes dans les données."""
        st.write("Gestion des valeurs manquantes...")
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()
        st.write(f"Valeurs manquantes traitées. Nouvelle shape: {self.data.shape}")

    def check_stationarity(self):
        """Vérifie la stationnarité des séries temporelles."""
        st.write("Vérification de la stationnarité des séries temporelles")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire (p-value: {result[1]:.4f}).")
            else:
                st.success(f"La série pour {column} est stationnaire (p-value: {result[1]:.4f}).")

    def make_stationary(self):
        """Rend les séries temporelles stationnaires."""
        st.write("Transformation des séries en séries stationnaires...")
        self.returns = self.data.diff().dropna()
        st.write(f"Séries rendues stationnaires. Nouvelle shape: {self.returns.shape}")

    def detect_outliers(self):
        """Détecte les valeurs aberrantes dans les données."""
        st.write("Détection des outliers...")
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        st.write(f"Nombre d'outliers détectés : {outliers.sum().sum()}")

    def scale_data(self):
        """Normalise les données."""
        st.write("Normalisation des données...")
        scaler = StandardScaler()
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.data), 
                                        index=self.data.index, 
                                        columns=self.data.columns)
        st.write("Les données ont été standardisées et mises à la même échelle")

    def test_cointegration(self):
        """Effectue les tests de cointégration entre toutes les paires de cryptomonnaies."""
        st.markdown('<p class="subheader">Tests de Cointégration</p>', unsafe_allow_html=True)
        n = len(self.tickers)
        for i in range(n):
            for j in range(i+1, n):
                ticker1, ticker2 = self.tickers[i], self.tickers[j]
                _, pvalue, _ = coint(self.scaled_data[ticker1], self.scaled_data[ticker2])
                if pvalue < 0.01:  # Utilisation du seuil de 0.01
                    self.pairs.append((ticker1, ticker2))
                    st.info(f"**{ticker1} et {ticker2} sont co-intégrés (p-value={pvalue:.4f})**")
                else:
                    st.write(f"{ticker1} et {ticker2} ne sont pas co-intégrés (p-value={pvalue:.4f})")

    def calculate_spread(self, ticker1, ticker2):
        """Calcule le spread entre deux cryptomonnaies normalisées."""
        return self.scaled_data[ticker1] - self.scaled_data[ticker2]

    def calculate_zscore(self, spread):
        """Calcule le z-score du spread."""
        return (spread - spread.rolling(window=20).mean()) / spread.rolling(window=20).std()

    def generate_trading_signals(self, ticker1, ticker2, investment_amount):
        """Génère des signaux de trading basés sur le z-score du spread."""
        spread = self.calculate_spread(ticker1, ticker2)
        zscore = self.calculate_zscore(spread)
        
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['signal'] = 0.0
        signals['signal'][zscore > 2.0] = -1.0  # Sell signal
        signals['signal'][zscore < -2.0] = 1.0  # Buy signal
        
        # Close positions if z-score between -1 and 1
        signals['signal'][(zscore > -1.0) & (zscore < 1.0)] = 0.0
        
        return signals

    def backtest_pair(self, ticker1, ticker2, investment_amount):
        """Effectue un backtest de la stratégie de trading par paires."""
        signals = self.generate_trading_signals(ticker1, ticker2, investment_amount)
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['positions'] = signals['signal'].diff()
        
        # Calculate returns
        portfolio['ticker1_returns'] = self.data[ticker1].pct_change()
        portfolio['ticker2_returns'] = self.data[ticker2].pct_change()
        portfolio['returns'] = portfolio['positions'].shift(1) * (portfolio['ticker1_returns'] - portfolio['ticker2_returns'])
        
        return portfolio

    def run_analysis(self, investment_amount):
        """Exécute l'analyse complète."""
        st.write("Début de l'analyse...")
        self.fetch_data()
        if self.data is None or self.data.empty:
            st.error("Pas de données à analyser. Arrêt de l'analyse.")
            return
        
        self.prepare_data()
        self.test_cointegration()
        
        st.markdown('<p class="subheader">Résultats du Trading par Paires</p>', unsafe_allow_html=True)
        for ticker1, ticker2 in self.pairs:
            portfolio = self.backtest_pair(ticker1, ticker2, investment_amount)
            cumulative_returns = (1 + portfolio['returns']).cumprod()
            
            st.markdown(f"""
            <div class='info-box'>
                <h3>Analyse pour la paire {ticker1} - {ticker2} :</h3>
                <p><strong>Rendement cumulatif :</strong> {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%</p>
                <p><strong>Nombre de trades :</strong> {portfolio['positions'].abs().sum()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, mode='lines', name='Rendement Cumulatif'))
            fig.update_layout(title=f"Rendement Cumulatif pour la paire {ticker1} - {ticker2}",
                              xaxis_title="Date",
                              yaxis_title="Rendement Cumulatif")
            st.plotly_chart(fig)

def main():
    st.markdown('<p class="big-font">Analyse Crypto avec Trading par Paires</p>', unsafe_allow_html=True)

    # Configuration des paramètres
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    investment_amount = st.sidebar.number_input("Montant d'investissement ($) :", min_value=100, value=10000, step=100)
    
    analyzer = ComprehensiveCryptoAnalyzer(tickers, names, start_date)
    
    if st.button("Lancer l'analyse"):
        try:
            with st.spinner("Analyse en cours... Veuillez patienter."):
                analyzer.run_analysis(investment_amount)
            st.success("Analyse terminée !")
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'analyse : {str(e)}")

if __name__ == "__main__":
    main()
