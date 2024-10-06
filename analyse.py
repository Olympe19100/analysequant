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

# Styles CSS améliorés
st.markdown("""
    <style>
    .big-font {
        font-size: 48px !important;
        color: #3366cc;
        text-align: center;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 28px;
        color: #1a53ff;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #b3d9ff;
    }
    .explanation {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 5px solid #4d94ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">📊 Analyse Crypto Avancée</p>', unsafe_allow_html=True)

# Explication simplifiée de la cointégration
st.markdown("""
    <div class="explanation">
        <h3>Qu'est-ce que la cointégration ? 🤔</h3>
        <p>Imaginez deux amis marchant ensemble. Parfois, l'un va plus vite, parfois c'est l'autre, 
        mais ils restent toujours proches l'un de l'autre. C'est comme ça que fonctionnent deux actifs cointégrés !</p>
        <p>En termes financiers, cela signifie que même si les prix de deux cryptomonnaies fluctuent, 
        ils ont tendance à suivre une trajectoire similaire sur le long terme. Cette relation peut être 
        utilisée pour prédire les mouvements futurs et créer des stratégies de trading.</p>
    </div>
""", unsafe_allow_html=True)

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
                    st.info(f"**Paire co-intégrée : {asset1} et {asset2} (p-value={pvalue:.4f})**")

    def calculate_hedge_ratios(self):
        st.markdown('<p class="subheader">Calcul des ratios de couverture 📊</p>', unsafe_allow_html=True)
        for asset1, asset2 in self.pairs:
            model = OLS(self.data[asset1], sm.add_constant(self.data[asset2])).fit()
            self.coverage_ratios[(asset1, asset2)] = model.params[1]
            st.write(f"Ratio de couverture pour {asset1}/{asset2} : {model.params[1]:.4f}")

    def calculate_spreads(self):
        st.markdown('<p class="subheader">Calcul des spreads 📉</p>', unsafe_allow_html=True)
        for (asset1, asset2), ratio in self.coverage_ratios.items():
            self.spreads[(asset1, asset2)] = self.data[asset1] - ratio * self.data[asset2]
            st.write(f"Spread calculé pour la paire {asset1}/{asset2}")

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
        st.markdown('<p class="subheader">Téléchargement des données historiques 📊</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="subheader">Préparation des Données 🔧</p>', unsafe_allow_html=True)
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()
        st.success("Préparation des données terminée.")

    def handle_missing_data(self, method='linear'):
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        st.write("**Vérification de la stationnarité des séries temporelles**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire.")
            else:
                st.success(f"La série pour {column} est stationnaire.")

    def make_stationary(self):
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        st.write("**Détection et gestion des outliers**")
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

    def random_forest_model(self):
        st.markdown('<p class="subheader">Modèle Forêt Aléatoire 🌲</p>', unsafe_allow_html=True)
        
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
        st.markdown('<p class="subheader">Visualisation des Relations Importantes 📊</p>', unsafe_allow_html=True)
        
        significant_assets = []
        
        for col in self.returns.columns:
            if col != 'BTC-USD':
                try:
                    test_result = grangercausalitytests(self.returns[['BTC-USD', col]], maxlag=5, verbose=False)
                    min_p_value = min(result[0]['ssr_ftest'][1] for result in test_result.values())
                    if min_p_value < 0.05:
                        significant_assets.append(col)
                        st.info(f"**{col} a une relation de causalité de Granger significative avec Bitcoin (p-value={min_p_value:.4f})**")
                except Exception as e:
                    st.warning(f"Problème avec le test de Granger pour {col} : {e}")
        
        btc_prices = self.data['BTC-USD']
        for col in self.data.columns:
            if col != 'BTC-USD':
                _, pvalue, _ = coint(btc_prices, self.data[col])
                if pvalue < 0.01:
                    significant_assets.append(col)
                    st.info(f"**{col} est co-intégré avec Bitcoin (p-value={pvalue:.4f})**")
        
        significant_assets = list(set(significant_assets))
        
        for col in significant_assets:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BTC-USD'], mode='lines', name='Bitcoin (BTC)'))
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[col], mode='lines', name=self.names[self.tickers.index(col)]))
            fig.update_layout(
                title=f"Relation entre Bitcoin et {self.names[self.tickers.index(col)]}", 
                xaxis_title="Date", 
                yaxis_title="Prix (mis à la même échelle)", 
                autosize=False, 
                width=800, 
                height=400,
                template="plotly_white"  # Utilisation d'un template plus esthétique
            )
            st.plotly_chart(fig)
            
            signal = self.data['BTC-USD'] - self.data[col]
            buy_signal = signal < signal.quantile(0.25)
            sell_signal = signal > signal.quantile(0.75)
            
            st.markdown(f"<div class='info-box'><h3>Signaux de trading pour la paire Bitcoin - {self.names[self.tickers.index(col)]} :</h3>"
                        f"Nombre de signaux d'achat : {buy_signal.sum()}<br>"
                        f"Nombre de signaux de vente : {sell_signal.sum()}<br>"
                        f"Taille de position suggérée : {((sell_signal.sum() - buy_signal.sum()) / (sell_signal.sum() + buy_signal.sum()) * 100):.2f}%"
                        "</div>", unsafe_allow_html=True)

# Fonction principale Streamlit avec une navigation simplifiée
def main():
    st.sidebar.header("📌 Navigation")
    page = st.sidebar.radio("Choisissez une section :", ["Accueil", "Analyse des cryptomonnaies"])

    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD']
    names = ['Bitcoin', 'Ethereum', 'Binance Coin', 'Cardano', 'Solana', 'Ripple', 'Dogecoin']
    
    start_date = st.sidebar.date_input("Date de début de l'analyse :", value=pd.to_datetime("2020-01-01"))
    
    analyzer = ComprehensiveCryptoCommoAnalyzer(tickers, names, start_date)

    if page == "Accueil":
        st.write("## Bienvenue dans l'analyseur de cryptomonnaies ! 👋")
        st.write("Cet outil vous aide à comprendre les relations entre Bitcoin et d'autres cryptomonnaies.")
        st.info("👈 Utilisez le menu à gauche pour commencer l'analyse.")
        
        st.markdown("""
        <div class="explanation">
            <h3>Comment utiliser cet outil ? 🛠️</h3>
            <ol>
                <li>Choisissez la date de début de l'analyse dans le menu latéral.</li>
                <li>Naviguez vers la section "Analyse des cryptomonnaies".</li>
                <li>Explorez les résultats de l'analyse, y compris les relations de cointégration et les signaux de trading.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Analyse des cryptomonnaies":
        st.write("## Analyse des cryptomonnaies 🚀")
        
        with st.spinner("Chargement et préparation des données..."):
            analyzer.fetch_data()
            analyzer.prepare_data()
        
        st.success("Données préparées avec succès!")
        
        st.markdown("""
        <div class="explanation">
            <h3>Que signifient ces résultats ? 🤔</h3>
            <p>L'analyse montre comment les différentes cryptomonnaies sont liées à Bitcoin. 
            Les relations significatives peuvent indiquer des opportunités de trading ou des tendances du marché.</p>
        </div>
        """, unsafe_allow_html=True)
        
        analyzer.random_forest_model()
        analyzer.plot_significant_relationships()
        
        st.markdown("""
        <div class="info-box">
            <h3>Interprétation des résultats 📈</h3>
            <ul>
                <li>Les paires co-intégrées indiquent des cryptomonnaies qui ont tendance à évoluer ensemble sur le long terme.</li>
                <li>Les signaux d'achat et de vente suggèrent des moments potentiels pour entrer ou sortir du marché.</li>
                <li>Le modèle de forêt aléatoire montre quelles cryptomonnaies ont le plus d'impact sur le prix du Bitcoin.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
