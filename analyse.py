import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, coint
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from pykalman import KalmanFilter
import statsmodels.api as sm

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse Crypto Avancée", page_icon="📊", layout="wide")

# Styles CSS (inchangés)
st.markdown("""
    <style>
    .big-font {
        font-size: 48px !important;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 28px;
        color: #1e3d59;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #f5f0e1;
        color: #1e3d59;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #ff6e40;
    }
    .explanation {
        background-color: #ffc13b;
        color: #1e3d59;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #ff6e40;
    }
    body {
        color: #1e3d59;
        background-color: #f5f0e1;
    }
    .stButton>button {
        color: #f5f0e1;
        background-color: #ff6e40;
        border-color: #ff6e40;
    }
    .stTextInput>div>div>input {
        color: #1e3d59;
    }
    </style>
    """, unsafe_allow_html=True)

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
        self.hurst_exponents = {}
        self.half_lives = {}
        self.kalman_results = {}

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
        """Normalise les données sur une échelle de 0 à 1."""
        st.write("Normalisation des données...")
        scaler = MinMaxScaler()
        self.scaled_data = pd.DataFrame(scaler.fit_transform(self.data), 
                                        index=self.data.index, 
                                        columns=self.data.columns)
        st.write("Les données ont été mises à l'échelle entre 0 et 1")

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

    def calculate_hurst_exponent(self, time_series):
        """Calcule l'exposant de Hurst pour une série temporelle."""
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def compute_hurst_exponents(self):
        """Calcule l'exposant de Hurst pour chaque crypto-monnaie."""
        for ticker in self.tickers:
            self.hurst_exponents[ticker] = self.calculate_hurst_exponent(self.scaled_data[ticker])
            st.write(f"Exposant de Hurst pour {ticker}: {self.hurst_exponents[ticker]:.4f}")

    def estimate_half_life(self, spread):
        """Estime la demi-vie du retour à la moyenne pour un spread donné."""
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        spread_ret = spread - spread_lag
        spread_ret.iloc[0] = spread_ret.iloc[1]
        spread_lag2 = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, spread_lag2)
        res = model.fit()
        half_life = -np.log(2) / res.params[1]
        return half_life

    def apply_kalman_filter(self, ticker1, ticker2):
        """Applique un filtre de Kalman pour estimer dynamiquement les paramètres du modèle."""
        obs_mat = np.vstack([self.scaled_data[ticker2], np.ones(self.scaled_data[ticker2].shape)]).T[:, np.newaxis]
        transition_matrices = np.array([[1, 0], [0, 1]])
        observation_matrices = obs_mat
        initial_state_mean = np.zeros(2)
        initial_state_covariance = np.ones((2, 2))
        observation_covariance = 1.0
        transition_covariance = 0.01 * np.eye(2)

        kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance
        )

        state_means, _ = kf.filter(self.scaled_data[ticker1].values)
        return state_means

    def calculate_spread(self, ticker1, ticker2):
        """Calcule le spread entre deux cryptomonnaies mises à l'échelle."""
        return self.scaled_data[ticker1] - self.scaled_data[ticker2]

    def generate_trading_signals(self, ticker1, ticker2, investment_amount):
        """Génère des signaux de trading basés sur le z-score du spread, l'exposant de Hurst, et la demi-vie."""
        spread = self.calculate_spread(ticker1, ticker2)
        half_life = self.estimate_half_life(spread)
        hurst_exponent = (self.hurst_exponents[ticker1] + self.hurst_exponents[ticker2]) / 2
        
        # Appliquer le filtre de Kalman
        state_means = self.apply_kalman_filter(ticker1, ticker2)
        kalman_spread = self.scaled_data[ticker1] - state_means[:, 0] * self.scaled_data[ticker2] - state_means[:, 1]
        
        # Calculer le z-score basé sur la demi-vie
        rolling_mean = spread.rolling(window=int(half_life)).mean()
        rolling_std = spread.rolling(window=int(half_life)).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        signals = pd.DataFrame(index=zscore.index)
        signals['zscore'] = zscore
        signals['kalman_zscore'] = (kalman_spread - kalman_spread.mean()) / kalman_spread.std()
        signals['signal'] = 0.0
        
        # Ajuster les seuils en fonction de l'exposant de Hurst
        upper_threshold = 2.0 + (0.5 - hurst_exponent)
        lower_threshold = -2.0 - (0.5 - hurst_exponent)
        
        signals['signal'][(signals['zscore'] > upper_threshold) & (signals['kalman_zscore'] > upper_threshold)] = -1.0  # Sell signal
        signals['signal'][(signals['zscore'] < lower_threshold) & (signals['kalman_zscore'] < lower_threshold)] = 1.0  # Buy signal
        
        # Close positions if z-score between -0.5 and 0.5
        signals['signal'][(signals['zscore'].between(-0.5, 0.5)) & (signals['kalman_zscore'].between(-0.5, 0.5))] = 0.0
        
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
        self.compute_hurst_exponents()
        self.test_cointegration()
        
        st.markdown('<p class="subheader">Résultats du Trading par Paires</p>', unsafe_allow_html=True)
        for ticker1, ticker2 in self.pairs:
            portfolio = self.backtest_pair(ticker1, ticker2, investment_amount)
            cumulative_returns = (1 + portfolio['returns']).cumprod()
            
            st.markdown(f"""
            def run_analysis(self, investment_amount):
        """Exécute l'analyse complète."""
        st.write("Début de l'analyse...")
        self.fetch_data()
        if self.data is None or self.data.empty:
            st.error("Pas de données à analyser. Arrêt de l'analyse.")
            return
        
        self.prepare_data()
        self.compute_hurst_exponents()
        self.test_cointegration()
        
        st.markdown('<p class="subheader">Résultats du Trading par Paires</p>', unsafe_allow_html=True)
        for ticker1, ticker2 in self.pairs:
            portfolio = self.backtest_pair(ticker1, ticker2, investment_amount)
            cumulative_returns = (1 + portfolio['returns']).cumprod()
            
            spread = self.calculate_spread(ticker1, ticker2)
            half_life = self.estimate_half_life(spread)
            
            st.markdown(f"""
            <div class='info-box'>
                <h3>Analyse pour la paire {ticker1} - {ticker2} :</h3>
                <p><strong>Rendement cumulatif :</strong> {(cumulative_returns.iloc[-1] - 1) * 100:.2f}%</p>
                <p><strong>Nombre de trades :</strong> {portfolio['positions'].abs().sum()}</p>
                <p><strong>Exposant de Hurst de {ticker1} :</strong> {self.hurst_exponents[ticker1]:.4f}</p>
                <p><strong>Exposant de Hurst de {ticker2} :</strong> {self.hurst_exponents[ticker2]:.4f}</p>
                <p><strong>Demi-vie estimée du spread :</strong> {half_life:.2f} jours</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, mode='lines', name='Rendement Cumulatif'))
            fig.update_layout(title=f"Rendement Cumulatif pour la paire {ticker1} - {ticker2}",
                              xaxis_title="Date",
                              yaxis_title="Rendement Cumulatif")
            st.plotly_chart(fig)
            
            # Afficher le spread et les signaux de trading
            signals = self.generate_trading_signals(ticker1, ticker2, investment_amount)
            fig_spread = go.Figure()
            fig_spread.add_trace(go.Scatter(x=spread.index, y=spread.values, mode='lines', name='Spread'))
            fig_spread.add_trace(go.Scatter(x=signals.index, y=signals['zscore'], mode='lines', name='Z-Score'))
            fig_spread.add_trace(go.Scatter(x=signals.index, y=signals['kalman_zscore'], mode='lines', name='Kalman Z-Score'))
            
            buy_signals = signals[signals['signal'] == 1.0]
            sell_signals = signals[signals['signal'] == -1.0]
            
            fig_spread.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['zscore'], 
                                            mode='markers', name='Signal d\'achat', 
                                            marker=dict(symbol='triangle-up', size=10, color='green')))
            fig_spread.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['zscore'], 
                                            mode='markers', name='Signal de vente', 
                                            marker=dict(symbol='triangle-down', size=10, color='red')))
            
            fig_spread.update_layout(title=f"Spread et Signaux de Trading pour {ticker1} - {ticker2}",
                                     xaxis_title="Date",
                                     yaxis_title="Valeur")
            st.plotly_chart(fig_spread)

def main():
    st.markdown('<p class="big-font">Analyse Crypto Avancée avec Trading par Paires</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation">
        <h3>Comment utiliser cet outil ?</h3>
        <ol>
            <li>Choisissez la date de début de l'analyse dans le menu latéral.</li>
            <li>Entrez le montant que vous souhaitez investir.</li>
            <li>Cliquez sur "Lancer l'analyse" pour commencer.</li>
            <li>Examinez les résultats de l'analyse pour chaque paire de cryptomonnaies cointégrées.</li>
            <li>Utilisez les recommandations et les graphiques pour prendre des décisions d'investissement éclairées.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

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
