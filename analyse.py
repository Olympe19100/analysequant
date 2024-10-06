import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, coint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st
from statsmodels.regression.linear_model import OLS

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
        self.ratios = {}
        self.signals = {}
        self.cointegration_results = {}
        self.hedging_ratios = {}

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

    def calculate_ratios(self):
        """Calcule les ratios entre Bitcoin et les autres cryptomonnaies."""
        st.markdown('<p class="subheader">Calcul des ratios</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD':
                self.ratios[col] = self.data['BTC-USD'] / self.data[col]
                st.write(f"Ratio calculé pour BTC/{col}")

    def generate_signals(self):
        """Génère des signaux basés sur les ratios et leurs moyennes mobiles."""
        st.markdown('<p class="subheader">Génération des signaux de trading</p>', unsafe_allow_html=True)
        for col, ratio in self.ratios.items():
            short_ma = ratio.rolling(window=10).mean()
            long_ma = ratio.rolling(window=30).mean()
            
            buy_signal = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
            sell_signal = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
            
            self.signals[col] = pd.DataFrame({'Buy': buy_signal, 'Sell': sell_signal})
            
            st.write(f"Signaux générés pour BTC/{col}")

    def analyze_cointegration(self):
        """Analyse la cointégration entre Bitcoin et les autres cryptomonnaies."""
        st.markdown('<p class="subheader">Analyse de cointégration</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD':
                _, pvalue, _ = coint(self.data['BTC-USD'], self.data[col])
                self.cointegration_results[col] = pvalue
                if pvalue < 0.05:
                    st.info(f"**{col} est co-intégré avec Bitcoin (p-value={pvalue:.4f})**")
                    self.calculate_hedging_ratio('BTC-USD', col)

    def calculate_hedging_ratio(self, asset1, asset2):
        """Calcule le ratio de couverture entre deux actifs."""
        model = OLS(self.data[asset1], self.data[asset2]).fit()
        self.hedging_ratios[asset2] = model.params[0]
        st.write(f"Ratio de couverture pour {asset1}/{asset2} : {model.params[0]:.4f}")

    def random_forest_model(self):
        """Crée un modèle de forêt aléatoire pour prédire les rendements de Bitcoin."""
        st.markdown('<p class="subheader">Modèle Forêt Aléatoire</p>', unsafe_allow_html=True)
        
        features = self.returns.drop('BTC-USD', axis=1)
        target = self.returns['BTC-USD']
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(features, target)
        
        predictions = rf_model.predict(features)
        r2 = r2_score(target, predictions)
        
        st.write(f"R2 score du modèle : {r2:.4f}")
        
        feature_importance = pd.Series(rf_model.feature_importances_, index=features.columns).sort_values(ascending=False)
        st.write("Importance des caractéristiques :")
        st.write(feature_importance)

    def plot_results(self):
        """Trace les graphiques des prix, ratios et signaux."""
        st.markdown('<p class="subheader">Visualisation des Résultats</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD':
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BTC-USD'], mode='lines', name='Bitcoin'))
                fig.add_trace(go.Scatter(x=self.data.index, y=self.data[col], mode='lines', name=col))
                fig.add_trace(go.Scatter(x=self.ratios[col].index, y=self.ratios[col], mode='lines', name=f'Ratio BTC/{col}', yaxis="y2"))
                
                # Ajout des signaux d'achat et de vente
                buy_signals = self.signals[col]['Buy']
                sell_signals = self.signals[col]['Sell']
                fig.add_trace(go.Scatter(x=buy_signals[buy_signals].index, y=self.data.loc[buy_signals[buy_signals].index, 'BTC-USD'],
                                         mode='markers', name='Signal d\'achat', marker=dict(symbol='triangle-up', size=10, color='green')))
                fig.add_trace(go.Scatter(x=sell_signals[sell_signals].index, y=self.data.loc[sell_signals[sell_signals].index, 'BTC-USD'],
                                         mode='markers', name='Signal de vente', marker=dict(symbol='triangle-down', size=10, color='red')))
                
                fig.update_layout(title=f"Prix, Ratio et Signaux pour BTC/{col}",
                                  xaxis_title="Date",
                                  yaxis_title="Prix",
                                  yaxis2=dict(title="Ratio", overlaying="y", side="right"))
                
                st.plotly_chart(fig)

    def run_analysis(self, investment_amount):
        """Exécute l'analyse complète."""
        self.fetch_data()
        self.prepare_data()
        self.calculate_ratios()
        self.generate_signals()
        self.analyze_cointegration()
        self.random_forest_model()
        self.plot_results()
        
        st.markdown('<p class="subheader">Résumé de l\'Analyse et Recommandations</p>', unsafe_allow_html=True)
        for col in self.data.columns:
            if col != 'BTC-USD':
                latest_signal = self.signals[col].iloc[-1]
                action = "Acheter" if latest_signal['Buy'] else "Vendre" if latest_signal['Sell'] else "Conserver"
                
                cointegrated = self.cointegration_results.get(col, 1) < 0.05
                hedging_ratio = self.hedging_ratios.get(col, None)
                
                st.markdown(f"""
                <div class='info-box'>
                    <h3>Analyse pour la paire Bitcoin - {self.names[self.tickers.index(col)]} :</h3>
                    <p><strong>Signal actuel :</strong> {action} Bitcoin</p>
                    <p><strong>Cointegration :</strong> {'Oui' if cointegrated else 'Non'}</p>
                    {f"<p><strong>Ratio de couverture :</strong> {hedging_ratio:.4f}</p>" if hedging_ratio else ""}
                    <p><strong>Recommandation :</strong><br>
                    {action} Bitcoin pour un montant de {investment_amount/2:.2f}$<br>
                    {"Acheter" if action == "Vendre" else "Vendre"} {self.names[self.tickers.index(col)]} pour un montant de {investment_amount/2:.2f}$</p>
                    <p><strong>Prix actuels :</strong><br>
                    Bitcoin (BTC) : {self.latest_prices['BTC-USD']:.2f}$<br>
                    {self.names[self.tickers.index(col)]} : {self.latest_prices[col]:.2f}$</p>
                </div>
                """, unsafe_allow_html=True)

def main():
    st.markdown('<p class="big-font">Analyse Crypto Avancée</p>', unsafe_allow_html=True)

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
                <li><strong>Signaux de trading :</strong> Basés sur les croisements des moyennes mobiles des ratios de prix.</li>
                <li><strong>Cointégration :</strong> Indique une relation à long terme entre les cryptomonnaies.</li>
                <li><strong>Ratio de couverture :</strong> Suggère la proportion optimale pour une stratégie de trading par paires.</li>
                <li><strong>Modèle Forêt Aléatoire :</strong> Montre l'importance relative de chaque cryptomonnaie dans la prédiction des rendements de Bitcoin.</li>
            </ul>
            <p><em>Note : Ces analyses sont basées sur des données historiques et ne garantissent pas les performances futures. Utilisez ces informations en conjonction avec d'autres outils et votre propre jugement pour prendre des décisions d'investissement.</em></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
