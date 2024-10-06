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
import statsmodels.api as sm  # Assurez-vous que cette ligne est présente en Ahaut de votre code
import scipy.stats as stats

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
        st.markdown("### Téléchargement des données historiques 📊")
        st.write("Nous collectons les données historiques des actifs spécifiés depuis Yahoo Finance.")
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
        """
        Prépare soigneusement les données :
        - Gestion des valeurs manquantes
        - Vérification de la stationnarité
        - Transformation pour rendre les séries stationnaires
        - Détection et gestion des outliers
        - Standardisation
        """
        st.markdown("### Préparation des Données 🔧")
        st.write("Nous préparons maintenant les données pour les analyses suivantes, en vérifiant leur qualité et en les rendant utilisables pour les modèles prédictifs.")
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()
        st.success("Préparation des données terminée.")

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
        st.write("**Vérification de la stationnarité des séries temporelles**")
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                st.warning(f"La série pour {column} n'est pas stationnaire.")
            else:
                st.success(f"La série pour {column} est stationnaire.")

    def make_stationary(self):
        """
        Applique une différenciation première pour rendre les séries temporelles stationnaires.
        """
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        """
        Détection des outliers dans les séries temporelles via la méthode IQR.
        """
        st.write("**Détection et gestion des outliers**")
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        st.write(f"Nombre d'outliers détectés : {outliers.sum().sum()}")

    def scale_data(self):
        """
        Standardisation des données.
        """
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)
        st.write("**Les données ont été standardisées**")

    def random_forest_model(self):
        """
        Construire un modèle Forêt Aléatoire basé sur les meilleures dérivées et inverses des actifs, excepté Bitcoin, et afficher le R².
        """
        st.markdown("### Modèle Forêt Aléatoire avec Meilleures Dérivées et Inverses des Actifs 🌲")
        st.write("Nous construisons un modèle de Forêt Aléatoire pour prédire les rendements de Bitcoin en utilisant les meilleures dérivées et inverses des autres actifs, sans utiliser Bitcoin lui-même.")

        # Préparation des features : dérivées premières, secondes et inverses des rendements
        best_features = {}
        for col in self.returns.columns:
            if col != 'BTC-USD':
                first_derivative = np.gradient(self.returns[col])
                second_derivative = np.gradient(first_derivative)
                inverse_returns = self.returns[col].apply(lambda x: 1/x if x != 0 else 0)

                best_features[f"Première Dérivée de {col}"] = first_derivative
                best_features[f"Seconde Dérivée de {col}"] = second_derivative
                best_features[f"Inverse des Rendements de {col}"] = inverse_returns

        # Créer une matrice de données avec les meilleures dérivées et inverses sélectionnées
        X = pd.DataFrame(best_features, index=self.returns.index)
        y = self.returns['BTC-USD']

        # Modèle Forêt Aléatoire
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        # Prédiction
        y_pred = rf_model.predict(X)

        # Calcul du R²
        r_squared = r2_score(y, y_pred)
        st.write(f"**R² du modèle Forêt Aléatoire :** {r_squared:.3f}")
        if r_squared > 0.7:
            st.success("Le modèle explique une grande partie de la variabilité des rendements de Bitcoin.")
        else:
            st.warning("Le modèle a une capacité limitée à expliquer la variabilité des rendements de Bitcoin.")

        # Importance des features
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        st.write("### Importance des Variables dans le Modèle Forêt Aléatoire")
        st.write(feature_importances.head(10))


def main():
    st.title("💡 Analyse des Relations entre Bitcoin et Autres Actifs Financiers")
    st.write("Bienvenue dans cette application d'analyse financière qui explore les liens entre Bitcoin et divers autres actifs.")

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

    st.header('1. Téléchargement et Préparation des Données')
    analyzer.fetch_data()
    
    # Préparation rigoureuse des données
    analyzer.prepare_data()

    st.header('2. Modèle Forêt Aléatoire avec Variables Significatives')
    analyzer.random_forest_model()

if __name__ == "__main__":
    main()
