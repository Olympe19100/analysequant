import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, coint
import plotly.graph_objects as go
import streamlit as st
from pykalman import KalmanFilter

# Configuration de la page Streamlit
st.set_page_config(page_title="Analyse Crypto Avancée", page_icon="📊", layout="wide")

# Styles CSS pour une meilleure présentation
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
        self.returns = None
        self.latest_prices = None
        self.cointegration_results = {}
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
            except Exception as e:
                missing_tickers.append(ticker)
        
        if data_dict:
            self.data = pd.DataFrame(data_dict)
            self.returns = self.data.pct_change().dropna()
            self.latest_prices = self.data.iloc[-1]
            if missing_tickers:
                st.error(f"Les tickers suivants n'ont pas pu être téléchargés : {missing_tickers}")
            else:
                st.success("Toutes les données ont été téléchargées avec succès!")

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
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        """Vérifie la stationnarité des séries temporelles."""
        st.write("**Vérification de la stationnarité des séries temporelles**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire.")
            else:
                st.success(f"La série pour {column} est stationnaire.")

    def make_stationary(self):
        """Rend les séries temporelles stationnaires."""
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        """Détecte les valeurs aberrantes dans les données."""
        st.write("**Détection et gestion des outliers**")
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        st.write(f"Nombre d'outliers détectés : {outliers.sum().sum()}")

    def scale_data(self):
        """Normalise les données."""
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        
        min_max_scaler = MinMaxScaler()
        self.data = pd.DataFrame(min_max_scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        
        st.write("**Les données ont été standardisées et mises à la même échelle**")

    def analyze_cointegration(self):
        """Analyse la cointégration entre Bitcoin et les autres cryptomonnaies."""
        st.markdown('<p class="subheader">Analyse de cointégration</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD':
                _, pvalue, _ = coint(self.data['BTC-USD'], self.data[col])
                self.cointegration_results[col] = pvalue
                if pvalue < 0.05:
                    st.info(f"**{col} est co-intégré avec Bitcoin (p-value={pvalue:.4f})**")
                    self.apply_kalman_filter('BTC-USD', col)

    def apply_kalman_filter(self, asset1, asset2):
        """Applique un filtre de Kalman pour estimer dynamiquement alpha et beta."""
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2)
        obs_mat = np.vstack([self.data[asset2], np.ones(self.data[asset2].shape)]).T[:, np.newaxis]

        kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                          initial_state_mean=np.zeros(2),
                          initial_state_covariance=np.ones((2, 2)),
                          transition_matrices=np.eye(2),
                          observation_matrices=obs_mat,
                          observation_covariance=1.0,
                          transition_covariance=trans_cov)

        state_means, state_covs = kf.filter(self.data[asset1].values)

        self.kalman_results[asset2] = {
            'beta': state_means[:, 0],
            'alpha': state_means[:, 1]
        }

    def generate_trading_signals(self, asset1, asset2, investment_amount, threshold=2):
        """Génère des signaux de trading basés sur les résultats du filtre de Kalman."""
        beta = self.kalman_results[asset2]['beta'][-1]
        alpha = self.kalman_results[asset2]['alpha'][-1]
        
        residuals = self.data[asset1] - (alpha + beta * self.data[asset2])
        z_score = (residuals - residuals.mean()) / residuals.std()
        
        last_residual = z_score.iloc[-1]
        
        if abs(last_residual) > threshold:
            action = "Vendre" if last_residual > 0 else "Acheter"
            n = investment_amount / (2 * self.latest_prices[asset1])  # Diviser l'investissement en deux
            
            return {
                'action_asset1': action,
                'quantity_asset1': n,
                'action_asset2': "Acheter" if action == "Vendre" else "Vendre",
                'quantity_asset2': n * beta
            }
        else:
            return None

    def run_analysis(self, investment_amount):
        """Exécute l'analyse complète."""
        self.fetch_data()
        self.prepare_data()
        self.analyze_cointegration()
        
        st.markdown('<p class="subheader">Résumé de l\'Analyse et Recommandations</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD' and self.cointegration_results.get(col, 1) < 0.05:
                signals = self.generate_trading_signals('BTC-USD', col, investment_amount)
                
                if signals:
                    st.markdown(f"""
                    <div class='info-box'>
                        <h3>Analyse pour la paire Bitcoin - {self.names[self.tickers.index(col)]} :</h3>
                        <p><strong>Cointegration :</strong> Oui (p-value={self.cointegration_results[col]:.4f})</p>
                        <p><strong>Recommandation :</strong><br>
                        {signals['action_asset1']} {signals['quantity_asset1']:.4f} unités de Bitcoin<br>
                        {signals['action_asset2']} {signals['quantity_asset2']:.4f} unités de {self.names[self.tickers.index(col)]}</p>
                        <p><strong>Prix actuels :</strong><br>
                        Bitcoin (BTC) : {self.latest_prices['BTC-USD']:.2f}$<br>
                        {self.names[self.tickers.index(col)]} : {self.latest_prices[col]:.2f}$</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"Pas de signal de trading pour la paire Bitcoin - {self.names[self.tickers.index(col)]} actuellement.")

        self.plot_results()

    def plot_results(self):
        """Trace les graphiques des prix et des résidus."""
        st.markdown('<p class="subheader">Visualisation des Résultats</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD' and col in self.kalman_results:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BTC-USD'], mode='lines', name='Bitcoin'))
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data[col], mode='lines', name=col))
                
                residuals = self.data['BTC-USD'] - (self.kalman_results[col]['alpha'] + self.kalman_results[col]['beta'] * self.data[col])
                fig.add_trace(go.Scatter(x=self.data.index, y=residuals, mode='lines', name='Residuals', yaxis="y2"))
                
                fig.update_layout(title=f"Prix et Résidus pour Bitcoin et {col}",
                                  xaxis_title="Date",
                                  yaxis_title="Prix",
                                  yaxis2=dict(title="Résidus", overlaying="y", side="right"))
                
                st.plotly_chart(fig)

def main():
    st.markdown('<p class="big-font">Analyse Crypto Avancée avec Cointégration</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explanation">
        <h3>Comment utiliser cet outil ?</h3>
        <ol>
            <li>Choisissez la date de début de l'analyse dans le menu latéral.</li>
            <li>Entrez le montant que vous souhaitez investir.</li>
            <li>Examinez les résultats de l'analyse pour chaque paire de cryptomonnaies.</li>
            <li>Utilisez les recommandations et les graphiques pour prendre des décisions d'investissement éclairées.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Configuration des paramètres
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    investment_amount = st.sidebar.number_input("Montant d'investissement ($) :", min_value=100, value=10000, step=100)
    
    # Création de l'analyseur
    analyzer = ComprehensiveCryptoAnalyzer(tickers, names, start_date)
    
    # Exécution de l'analyse
    if st.button("Lancer l'analyse"):
        with st.spinner("Analyse en cours... Veuillez patienter."):
            analyzer.run_analysis(investment_amount)
        
        st.success("Analyse terminée !")
        
        st.markdown("""
        <div class="explanation">
            <h3>Interprétation des résultats</h3>
            <ul>
                <li><strong>Cointégration :</strong> Indique une relation à long terme entre les cryptomonnaies.</li>
                <li><strong>Filtre de Kalman :</strong> Estime dynamiquement les paramètres du modèle (alpha et beta).</li>
                <li><strong>Signaux de trading :</strong> Basés sur les déviations des résidus par rapport à leur moyenne.</li>
                <li><strong>Graphiques :</strong> Montrent l'évolution des prix et des résidus au fil du temps, permettant de visualiser les opportunités de trading.</li>
            </ul>
            <p><em>Note : Ces analyses sont basées sur des données historiques et ne garantissent pas les performances futures. Utilisez ces informations en conjonction avec d'autres outils et votre propre jugement pour prendre des décisions d'investissement.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
