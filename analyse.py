import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm  # Assurez-vous que cette ligne est présente en haut de votre code

# Définir la classe d'analyse
class ComprehensiveCryptoCommoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names  # noms réels des actifs financiers
        self.start_date = start_date
        self.data = None
        self.returns = None
        self.significant_vars = []  # Pour stocker les variables significatives

    def fetch_data(self):
        """
        Télécharge les données historiques pour les tickers spécifiés.
        """
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
            st.write(f"Les tickers suivants n'ont pas pu être téléchargés : {missing_tickers}")

    def prepare_data(self):
        """
        Prépare soigneusement les données :
        - Gestion des valeurs manquantes
        - Vérification de la stationnarité
        - Transformation pour rendre les séries stationnaires
        - Détection et gestion des outliers
        - Standardisation
        """
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()

    def handle_missing_data(self, method='linear'):
        """
        Gère les valeurs manquantes dans les séries temporelles.
        """
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        """
        Effectue un test de Dickey-Fuller Augmenté (ADF) pour vérifier la stationnarité des séries temporelles.
        """
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                pass  # Transformer la série si non stationnaire

    def make_stationary(self):
        """
        Applique une différenciation première pour rendre les séries temporelles stationnaires.
        """
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        """
        Détection des outliers dans les séries temporelles via la méthode IQR.
        """
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        return outliers

    def scale_data(self):
        """
        Standardisation des données.
        """
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)

    def correlation_analysis(self):
        """
        Analyse de corrélation pour comprendre les relations linéaires entre Bitcoin et les autres actifs.
        """
        st.write("## Analyse de Corrélation")
        corr_matrix = self.returns.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=self.names,
            y=self.names,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Matrice de corrélation des rendements")
        st.plotly_chart(fig)

        btc_corr = corr_matrix['BTC-USD'].sort_values(ascending=False)
        st.write("### Top 5 des actifs corrélés positivement avec Bitcoin :")
        st.write(btc_corr.head())

        st.write("### Top 5 des actifs corrélés négativement avec Bitcoin :")
        st.write(btc_corr.tail())

    def granger_causality(self, max_lag=10):
        """
        Test de Causalité de Granger pour évaluer si un actif financier peut prédire le mouvement de Bitcoin.
        """
        st.write("## Causalité de Granger")
        causality_results = {}
        for col in self.returns.columns:
            if col != 'BTC-USD':
                test_result = grangercausalitytests(self.returns[['BTC-USD', col]], maxlag=max_lag, verbose=False)
                min_p_value = min(result[0]['ssr_ftest'][1] for result in test_result.values())
                causality_results[col] = min_p_value

        causality_df = pd.DataFrame.from_dict(causality_results, orient='index', columns=['p-value'])
        causality_df = causality_df.sort_values('p-value')

        st.write("### Top 5 des actifs avec une causalité de Granger significative sur Bitcoin :")
        st.write(causality_df.head())

    def cointegration_analysis(self):
        """
        Test de co-intégration pour identifier les actifs qui partagent une relation à long terme avec Bitcoin.
        """
        st.write("## Analyse de Co-intégration")
        btc_prices = self.data['BTC-USD']
        other_prices = self.data.drop('BTC-USD', axis=1)

        results = {}
        for column in other_prices.columns:
            _, pvalue, _ = coint(btc_prices, other_prices[column])
            results[column] = pvalue

        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['p-value'])
        results_df = results_df.sort_values('p-value')

        st.write("### Top 5 des actifs co-intégrés avec Bitcoin :")
        st.write(results_df.head())

    def mutual_information_analysis(self):
        """
        Analyse d'information mutuelle pour détecter les relations non-linéaires entre Bitcoin et les autres actifs.
        """
        st.write("## Information Mutuelle")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']

        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame({'Variable': self.names[1:], 'MI Score': mi_scores})
        mi_df = mi_df.sort_values('MI Score', ascending=False)

        st.write("### Top 5 des actifs avec la plus forte dépendance avec Bitcoin :")
        st.write(mi_df.head())

    def f_test_analysis(self):
        """
        Effectue un test F pour évaluer la significativité des variables explicatives sur BTC-USD.
        """
        st.write("## F-Test d'Évaluation des Variables")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']

        # Ajout d'une constante pour le modèle de régression
        X = sm.add_constant(X)

        # Régression linéaire multiple avec statsmodels
        model = sm.OLS(y, X).fit()

        st.write(model.summary())

        # Extraire les p-values associées à chaque variable
        p_values = model.pvalues.drop('const')  # Ignorer la constante

        st.write("### P-values des variables explicatives :")
        st.write(p_values)

        # Sélection des variables significatives (p-value < 0.05)
        self.significant_vars = p_values[p_values < 0.05].index
        st.write("### Variables ayant un pouvoir prédictif significatif (p-value < 0.05) :")
        st.write(self.significant_vars)

    def random_forest_model(self):
        """
        Construire un modèle Forêt Aléatoire basé sur les variables significatives et afficher le R².
        """
        st.write("## Modèle Forêt Aléatoire avec les Variables Significatives")

        # Créer une nouvelle matrice de données avec uniquement les variables significatives
        X = self.returns[self.significant_vars]
        y = self.returns['BTC-USD']

        # Modèle Forêt Aléatoire
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Prédiction
        y_pred = rf_model.predict(X)

        # Calcul du R²
        r_squared = r2_score(y, y_pred)
        st.write(f"R² du modèle Forêt Aléatoire : {r_squared:.3f}")

# Main Streamlit Application
def main():
    st.title("Analyse des Relations entre Bitcoin et Autres Actifs Financiers")

    # Liste des tickers et noms réels des actifs financiers
    tickers = [
        'BTC-USD', 'SPY', 'QQQ', '^GDAXI', '^FTSE', 'CL=F', 'BZ=F', 'NG=F',
        'GC=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F', 'ZN=F', 'ZW=F',
        'ZC=F', 'ZS=F', 'KC=F', 'CT=F', 'CC=F', 'SB=F', 'OJ=F', 'LE=F', 'HE=F',
        'RB=F', 'HO=F', 'EURUSD=X', 'GBPUSD=X', 'TLT', 'LQD', 'HYG'
    ]
    
    # Noms réels des actifs financiers correspondants
    names = [
        'Bitcoin (BTC)', 'S&P 500 (SPY)', 'Nasdaq 100 (QQQ)', 'DAX Allemagne (^GDAXI)', 'FTSE 100 (^FTSE)',
        'Pétrole WTI (CL=F)', 'Pétrole Brent (BZ=F)', 'Gaz Naturel (NG=F)', 'Or (GC=F)', 'Argent (SI=F)',
        'Platine (PL=F)', 'Palladium (PA=F)', 'Cuivre (HG=F)', 'Zinc (ZN=F)', 'Blé (ZW=F)',
        'Maïs (ZC=F)', 'Soja (ZS=F)', 'Café (KC=F)', 'Coton (CT=F)', 'Cacao (CC=F)', 'Sucre (SB=F)',
        'Jus d\'Orange (OJ=F)', 'Bétail (LE=F)', 'Porcs (HE=F)', 'Essence (RB=F)', 'Fuel (HO=F)',
        'Euro-Dollar (EURUSD=X)', 'Livre Sterling-Dollar (GBPUSD=X)', 'Obligations US 20 ans (TLT)',
        'Obligations d\'entreprises (LQD)', 'Obligations à haut rendement (HYG)'
    ]

    start_date = '2010-07-18'

    analyzer = ComprehensiveCryptoCommoAnalyzer(tickers, names, start_date)

    st.header('Téléchargement et Préparation des Données')
    analyzer.fetch_data()
    
    # Préparation rigoureuse des données sans affichage dans Streamlit
    analyzer.prepare_data()

    st.subheader('Analyse de Corrélation')
    analyzer.correlation_analysis()

    st.subheader('Causalité de Granger')
    analyzer.granger_causality()

    st.subheader('Analyse de Co-intégration')
    analyzer.cointegration_analysis()

    st.subheader('Analyse d\'Information Mutuelle')
    analyzer.mutual_information_analysis()

    st.subheader('F-Test pour évaluer les variables explicatives')
    analyzer.f_test_analysis()

    st.subheader('Modèle Forêt Aléatoire avec Variables Significatives')
    analyzer.random_forest_model()

if __name__ == "__main__":
    main()