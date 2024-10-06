import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.stattools import adfuller, coint
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

    def fetch_data(self):
        """Télécharge les données historiques pour toutes les cryptomonnaies."""
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
            self.latest_prices = self.data.iloc[-1]
            if missing_tickers:
                st.error(f"Les tickers suivants n'ont pas pu être téléchargés : {missing_tickers}")
            else:
                st.success("Toutes les données ont été téléchargées avec succès!")

    def prepare_data(self):
        """Prépare les données en gérant les valeurs manquantes, vérifiant la stationnarité, et normalisant les données."""
        st.markdown('<p class="subheader">Préparation des Données 🔧</p>', unsafe_allow_html=True)
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

    def calculate_hedging_ratio(self, asset1, asset2):
        """Calcule le ratio de couverture entre deux actifs."""
        model = OLS(self.data[asset1], self.data[asset2]).fit()
        return model.params[0]

    def analyze_relationships(self, investment_amount):
        """Analyse les relations entre Bitcoin et les autres cryptomonnaies."""
        st.markdown('<p class="subheader">Analyse des relations entre cryptomonnaies 📊</p>', unsafe_allow_html=True)
        
        for col in self.data.columns:
            if col != 'BTC-USD':
                # Test de cointégration
                _, pvalue, _ = coint(self.data['BTC-USD'], self.data[col])
                
                if pvalue < 0.05:
                    st.info(f"**{col} est co-intégré avec Bitcoin (p-value={pvalue:.4f})**")
                    
                    # Calcul du hedging ratio
                    hedging_ratio = self.calculate_hedging_ratio('BTC-USD', col)
                    
                    # Calcul du spread
                    spread = self.data['BTC-USD'] - hedging_ratio * self.data[col]
                    
                    # Calcul des signaux basés sur le spread
                    mean_spread = spread.mean()
                    std_spread = spread.std()
                    
                    current_spread = self.latest_prices['BTC-USD'] - hedging_ratio * self.latest_prices[col]
                    
                    # Détermination de l'action à prendre
                    if current_spread > mean_spread + std_spread:
                        action_btc = "Vendre"
                        action_other = "Acheter"
                    elif current_spread < mean_spread - std_spread:
                        action_btc = "Acheter"
                        action_other = "Vendre"
                    else:
                        action_btc = "Conserver"
                        action_other = "Conserver"
                    
                    # Calcul des unités à acheter/vendre
                    btc_price = self.latest_prices['BTC-USD']
                    other_price = self.latest_prices[col]
                    
                    if action_btc != "Conserver":
                        btc_units = (investment_amount / 2) / btc_price
                        other_units = (investment_amount / 2) / other_price
                    else:
                        btc_units = other_units = 0
                    
                    # Affichage des résultats
                    st.markdown(f"""
                    <div class='info-box'>
                        <h3>Analyse pour la paire Bitcoin - {self.names[self.tickers.index(col)]} :</h3>
                        <p><strong>Hedging Ratio :</strong> {hedging_ratio:.4f}<br>
                        <em>Interprétation : Pour chaque unité de Bitcoin, vous devriez détenir {hedging_ratio:.4f} unités de {self.names[self.tickers.index(col)]} pour une couverture optimale.</em></p>
                        <p><strong>Spread actuel :</strong> {current_spread:.4f}<br>
                        <strong>Spread moyen :</strong> {mean_spread:.4f}<br>
                        <strong>Écart-type du spread :</strong> {std_spread:.4f}<br>
                        <em>Interprétation : Le spread actuel est {'au-dessus' if current_spread > mean_spread else 'en-dessous'} de la moyenne, 
                        indiquant que Bitcoin est potentiellement {'surévalué' if current_spread > mean_spread else 'sous-évalué'} par rapport à {self.names[self.tickers.index(col)]}.</em></p>
                        <p><strong>Recommandation :</strong><br>
                        - {action_btc} Bitcoin (BTC)<br>
                        - {action_other} {self.names[self.tickers.index(col)]}</p>
                        <p><strong>Pour un investissement de {investment_amount:.2f}$ :</strong><br>
                        - {btc_units:.4f} unités de Bitcoin (BTC)<br>
                        - {other_units:.4f} unités de {self.names[self.tickers.index(col)]}</p>
                        <p><strong>Prix actuels :</strong><br>
                        - Bitcoin (BTC) : {btc_price:.2f}$<br>
                        - {self.names[self.tickers.index(col)]} : {other_price:.2f}$</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Graphique
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=self.data.index, y=self.data['BTC-USD'], mode='lines', name='Bitcoin (BTC)'))
                    fig.add_trace(go.Scatter(x=self.data.index, y=self.data[col], mode='lines', name=self.names[self.tickers.index(col)]))
                    fig.add_trace(go.Scatter(x=self.data.index, y=spread, mode='lines', name='Spread'))
                    fig.update_layout(title=f"Prix et Spread pour Bitcoin et {self.names[self.tickers.index(col)]}", 
                                      xaxis_title="Date", yaxis_title="Prix / Spread")
                    st.plotly_chart(fig)

def main():
    st.markdown('<p class="big-font">📊 Analyse Crypto Avancée avec Hedging Ratio</p>', unsafe_allow_html=True)

    # Explication de l'outil
    st.markdown("""
    <div class="explanation">
        <h3>Comment utiliser cet outil ? 🛠️</h3>
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
    
    # Création et utilisation de l'analyseur
    analyzer = ComprehensiveCryptoAnalyzer(tickers, names, start_date)
    analyzer.fetch_data()
    analyzer.prepare_data()  # Ajout de l'étape de préparation des données
    analyzer.analyze_relationships(investment_amount)

    # Explication des résultats
    st.markdown("""
    <div class="explanation">
        <h3>Interprétation des résultats 📈</h3>
        <ul>
            <li><strong>Prétraitement des données :</strong> Les données sont nettoyées, rendues stationnaires et normalisées pour une analyse plus précise.</li>
            <li><strong>Hedging Ratio :</strong> Indique combien d'unités de l'autre crypto sont nécessaires pour couvrir une unité de Bitcoin.</li>
            <li><strong>Spread :</strong> Représente la différence entre le prix du Bitcoin et le prix ajusté (par le hedging ratio) de l'autre crypto.</li>
            <li><strong>Signaux d'achat/vente :</strong> Basés sur la position du spread par rapport à sa moyenne historique.</li>
            <li><strong>Recommandations :</strong> Suggèrent d'acheter lorsque le spread est significativement en dessous de sa moyenne (Bitcoin potentiellement sous-évalué) et de vendre lorsqu'il est au-dessus (Bitcoin potentiellement surévalué).</li>
        </ul>
        <p><em>Note : Ces analyses sont basées sur des données historiques et ne garantissent pas les performances futures. Utilisez ces informations en conjonction avec d'autres outils et votre propre jugement pour prendre des décisions d'investissement.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
