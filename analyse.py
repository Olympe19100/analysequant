import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.regression.linear_model import OLS

st.set_page_config(page_title="Analyse Crypto", page_icon="📈", layout="wide")

# Custom color scheme for Olympe Financial Group
theme_colors = {
    'primary': '#003366',  # Dark blue
    'secondary': '#006699',  # Lighter blue
    'background': '#F0F8FF',  # Light background
    'highlight': '#FFD700',  # Gold for highlights
    'text': '#000000'  # Black for general text
}

st.markdown(f"""
    <style>
    .big-font {{
        font-size:50px !important;
        color: {theme_colors['primary']};
        text-align: center;
    }}
    .subheader {{
        font-size:30px;
        color: {theme_colors['secondary']};
    }}
    .info-box {{
        background-color: {theme_colors['background']};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {theme_colors['highlight']};
        color: {theme_colors['text']};
    }}
    .success-message {{
        color: {theme_colors['primary']};
    }}
    .warning-message {{
        color: #FF4500;  /* Bright red for warnings */
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">📈 Analyse Crypto Avancée</p>', unsafe_allow_html=True)

# Classe pour les stratégies de co-intégration
class CointegrationStrategies:
    def __init__(self, data):
        self.data = data
        self.pairs = []
        self.coverage_ratios = {}
        self.spreads = {}

    def identify_cointegrated_pairs(self):
        st.markdown('<p class="subheader">Identification des paires co-intégrées 🔍</p>', unsafe_allow_html=True)
        for i in range(len(self.data.columns)):
            for j in range(i + 1, len(self.data.columns)):
                asset1 = self.data.columns[i]
                asset2 = self.data.columns[j]
                _, pvalue, _ = coint(self.data[asset1], self.data[asset2])
                if pvalue < 0.05:
                    self.pairs.append((asset1, asset2))
                    st.markdown(f"<div class='info-box'><strong>Paire co-intégrée : {asset1} et {asset2} (p-value={pvalue:.4f})</strong></div>", unsafe_allow_html=True)

    def calculate_hedge_ratios(self):
        st.markdown('<p class="subheader">Calcul des ratios de couverture 📈</p>', unsafe_allow_html=True)
        for asset1, asset2 in self.pairs:
            model = OLS(self.data[asset1], sm.add_constant(self.data[asset2])).fit()
            self.coverage_ratios[(asset1, asset2)] = model.params[1]
            st.markdown(f"<div class='info-box'>Ratio de couverture pour {asset1}/{asset2} : {model.params[1]:.4f}</div>", unsafe_allow_html=True)

    def calculate_spreads(self):
        st.markdown('<p class="subheader">Calcul des spreads 📉</p>', unsafe_allow_html=True)
        for (asset1, asset2), ratio in self.coverage_ratios.items():
            self.spreads[(asset1, asset2)] = self.data[asset1] - ratio * self.data[asset2]
            st.markdown(f"<div class='info-box'>Spread calculé pour la paire {asset1}/{asset2}</div>", unsafe_allow_html=True)

    def generate_signals(self):
        st.markdown('<p class="subheader">Génération des signaux de trading 🚦</p>', unsafe_allow_html=True)
        for (asset1, asset2), spread in self.spreads.items():
            z_score = (spread - spread.mean()) / spread.std()
            buy_signal = z_score < -2
            sell_signal = z_score > 2
            st.markdown(f"<div class='info-box'><h3>Signaux pour {asset1}/{asset2} :</h3>"
                        f"Nombre de signaux d'achat : {buy_signal.sum()}<br>"
                        f"Nombre de signaux de vente : {sell_signal.sum()}<br>"
                        f"Taille de position suggérée : {((sell_signal.sum() - buy_signal.sum()) / (sell_signal.sum() + buy_signal.sum()) * 100):.2f}%"
                        "</div>", unsafe_allow_html=True)

# Classe principale d'analyse
class ComprehensiveCryptoCommoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.returns = None
        self.significant_vars = []
        self.cointegration = None

    def fetch_data(self):
        st.markdown('<p class="subheader">Téléchargement des données historiques 📈</p>', unsafe_allow_html=True)
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
                st.markdown(f"<span class='warning-message'>Les tickers suivants n'ont pas pu être téléchargés : {missing_tickers}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='success-message'>Toutes les données ont été téléchargées avec succès!</span>", unsafe_allow_html=True)

    def prepare_data(self):
        st.markdown('<p class="subheader">Préparation des Données 🔧</p>', unsafe_allow_html=True)
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()
        st.markdown(f"<span class='success-message'>Préparation des données terminée.</span>", unsafe_allow_html=True)

    def handle_missing_data(self, method='linear'):
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        st.write("**Vérification de la stationnarité des séries temporelles**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.markdown(f"<div class='warning-message'>La série pour {column} n'est pas stationnaire.</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='success-message'>La série pour {column} est stationnaire.</div>", unsafe_allow_html=True)

    def make_stationary(self):
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        st.write("**Détection et gestion des outliers**")
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        st.markdown(f"<div class='info-box'>Nombre d'outliers détectés : {outliers.sum().sum()}</div>", unsafe_allow_html=True)

    def scale_data(self):
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        
        min_max_scaler = MinMaxScaler()
        self.data = pd.DataFrame(min_max_scaler.fit_transform(self.data), index=self.data.index, columns=self.data.columns)
        
        st.write("**Les données ont été standardisées et mises à la même échelle**")

    def random_forest_model(self):
        st.markdown('<p class="subheader">Modèle Forêt Aléatoire 🌳</p>', unsafe_allow_html=True)
        
        best_features = {}
        for col in self.returns.columns:
            if col != 'BTC-USD':
                first_derivative = np.gradient(self.returns[col])
                second_derivative = np.gradient(first_derivative)
                inverse_returns = self.returns[col].apply(lambda x: 1/x if x != 0 else 0)
                best_features[f"Première Dérivée de {col}"] = first_derivative
                best_features[f"Seconde Dérivée de {col}"] = second_derivative
                best_features[f"Inverse des Rendements de {col}"] = inverse_returns
        
        X = pd.DataFrame(best_features, index=self.returns.index)
        y = self.returns['BTC-USD']
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt')
        rf_model.fit(X, y)
        
        y_pred = rf_model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        st.markdown(f"<div class='info-box'><h3>Résultats du modèle :</h3>"
                    f"R2 du modèle Forêt Aléatoire : {r_squared:.3f}<br>"
                    f"{'Le modèle explique une grande partie de la variabilité des rendements de Bitcoin.' if r_squared > 0.7 else 'Le modèle a une capacité limitée à expliquer la variabilité des rendements de Bitcoin.'}"
                    "</div>", unsafe_allow_html=True)
        
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        
        st.write("### Importance des Variables dans le Modèle Forêt Aléatoire")
        st.write(feature_importances.head(10))

    def plot_significant_relationships(self):
        st.markdown('<p class="subheader">Visualisation des Relations Statistiquement Significatives 📈</p>', unsafe_allow_html=True)
        
        significant_assets = []
        
        for col in self.returns.columns:
            if col != 'BTC-USD':
                try:
                    test_result = grangercausalitytests(self.returns[['BTC-USD', col]], maxlag=5, verbose=False)
                    min_p_value = min(result[0]['ssr_ftest'][1] for result in test_result.values())
                    if min_p_value < 0.05:
                        significant_assets.append(col)
                        st.markdown(f"<div class='info-box'><strong>{col} a une relation de causalité de Granger significative avec Bitcoin (p-value={min_p_value:.4f})</strong></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f"<div class='warning-message'>Problème avec le test de Granger pour {col} : {e}</div>", unsafe_allow_html=True)
        
        btc_prices = self.data['BTC-USD']
        for col in self.data.columns:
            if col != 'BTC-USD':
                _, pvalue, _ = coint(btc_prices, self.data[col])
                if pvalue < 0.01:
                    significant_assets.append(col)
                    st.markdown(f"<div class='info-box'><strong>{col} est co-intégré avec Bitcoin (p-value={pvalue:.4f})</strong></div>", unsafe_allow_html=True)
        
        significant_assets = list(set(significant_assets))
        
        for col in significant_assets:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BTC-USD'], mode='lines', name='Bitcoin (BTC)', line=dict(color=theme_colors['primary'])))
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[col], mode='lines', name=self.names[self.tickers.index(col)], line=dict(color=theme_colors['secondary'])))
            fig.update_layout(title=f"Relation entre Bitcoin et {self.names[self.tickers.index(col)]}",
                              xaxis_title="Date", yaxis_title="Prix (mis à la même échelle)",
                              autosize=False, width=800, height=400,
                              plot_bgcolor=theme_colors['background'],
                              paper_bgcolor=theme_colors['background'],
                              font=dict(color=theme_colors['text']))
            st.plotly_chart(fig)
            
            signal = self.data['BTC-USD'] - self.data[col]
            buy_signal = signal < signal.quantile(0.25)
            sell_signal = signal > signal.quantile(0.75)
            
            st.markdown(f"<div class='info-box'><h3>Signaux de trading pour la paire Bitcoin - {self.names[self.tickers.index(col)]} :</h3>"
                        f"Nombre de signaux d'achat : {buy_signal.sum()}<br>"
                        f"Nombre de signaux de vente : {sell_signal.sum()}<br>"
                        f"Taille de position suggérée : {((sell_signal.sum() - buy_signal.sum()) / (sell_signal.sum() + buy_signal.sum()) * 100):.2f}%"
                        "</div>", unsafe_allow_html=True)

# Fonction principale Streamlit
def main():
    st.sidebar.header("📍 Navigation")
    page = st.sidebar.radio("Choisissez une section :", ["Accueil", "Analyse des données", "Modélisation", "Visualisation"])

    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    
    analyzer = ComprehensiveCryptoCommoAnalyzer(tickers, names, start_date)

    if page == "Accueil":
        st.write("## Bienvenue dans l'analyseur de cryptomonnaies ! 👋")
        st.write("Cet outil vous aide à comprendre les relations entre Bitcoin et d'autres cryptomonnaies.")
        st.info("👏 Utilisez le menu à gauche pour naviguer entre les différentes sections.")

    elif page == "Analyse des données":
        analyzer.fetch_data()
        analyzer.prepare_data()

    elif page == "Modélisation":
        analyzer.fetch_data()
        analyzer.prepare_data()
        analyzer.random_forest_model()

    elif page == "Visualisation":
        analyzer.fetch_data()
        analyzer.prepare_data()
        analyzer.plot_significant_relationships()

if __name__ == "__main__":
    main()
