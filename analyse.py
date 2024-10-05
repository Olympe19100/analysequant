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

    def correlation_analysis(self):
        """
        Analyse de corrélation pour comprendre les relations linéaires entre Bitcoin et les autres actifs.
        """
        st.markdown("### Analyse de Corrélation 📈")
        st.write("Cette analyse permet de comprendre comment les variations de prix de Bitcoin sont associées à celles d'autres actifs financiers.")
        corr_matrix = self.returns.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=self.returns.columns,
            y=self.returns.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1
        ))
        fig.update_layout(title="Matrice de corrélation des rendements", autosize=False, width=800, height=600)
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
        st.markdown("### Causalité de Granger 🔗")
        st.write("Ce test évalue si les rendements d'un actif peuvent aider à prédire ceux de Bitcoin.")
        causality_results = {}
        for col in self.returns.columns:
            if col != 'BTC-USD':
                try:
                    test_result = grangercausalitytests(self.returns[['BTC-USD', col]], maxlag=max_lag, verbose=False)
                    min_p_value = min(result[0]['ssr_ftest'][1] for result in test_result.values())
                    causality_results[col] = min_p_value
                except ValueError:
                    st.warning(f"Problème avec le test de Granger pour {col}, peut-être dû à des données insuffisantes.")

        causality_df = pd.DataFrame.from_dict(causality_results, orient='index', columns=['p-value'])
        causality_df = causality_df.sort_values('p-value')

        st.write("### Top 5 des actifs avec une causalité de Granger significative sur Bitcoin :")
        st.write(causality_df.head())

    def cointegration_analysis(self):
        """
        Test de co-intégration pour identifier les actifs qui partagent une relation à long terme avec Bitcoin.
        """
        st.markdown("### Analyse de Co-intégration 🔗")
        st.write("La co-intégration indique une relation à long terme entre Bitcoin et d'autres actifs.")
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
        st.markdown("### Analyse d'Information Mutuelle 🤝")
        st.write("L'information mutuelle permet de détecter des relations non-linéaires entre les actifs.")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']

        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame({'Variable': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values('MI Score', ascending=False)

        st.write("### Top 5 des actifs avec la plus forte dépendance avec Bitcoin :")
        st.write(mi_df.head())

    def f_test_analysis(self):
        """
        Effectue un test F pour évaluer la significativité des variables explicatives sur BTC-USD.
        """
        st.markdown("### F-Test d'Évaluation des Variables 📊")
        st.write("Ce test permet de savoir si d'autres actifs ont un effet significatif sur Bitcoin.")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']

        # Ajout d'une constante pour le modèle de régression
        X = sm.add_constant(X)

        # Régression linéaire multiple avec statsmodels
        model = sm.OLS(y, X).fit()

        st.write("### Résultats du Test F :")
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
        Construire un modèle Forêt Aléatoire basé sur les meilleures dérivées et inverses des actifs, excepté Bitcoin, et afficher le R².
        """
        st.markdown("### Modèle Forêt Aléatoire avec Meilleures Dérivées et Inverses des Actifs 🌲")
        st.write("Nous construisons un modèle de Forêt Aléatoire pour prédire les rendements de Bitcoin en utilisant les meilleures dérivées et inverses des autres actifs.")

        # Sélectionner les dérivées et inverses les plus corrélés avec BTC-USD
        best_features = []
        for col in self.returns.columns:
            if col != 'BTC-USD':
                first_derivative = np.gradient(self.returns[col])
                second_derivative = np.gradient(first_derivative)
                inverse_returns = self.returns[col].apply(lambda x: 1/x if x != 0 else 0)

                # Calculer la corrélation avec Bitcoin
                correlation_first = np.corrcoef(first_derivative, self.returns['BTC-USD'])[0, 1]
                correlation_second = np.corrcoef(second_derivative, self.returns['BTC-USD'])[0, 1]
                correlation_inverse = np.corrcoef(inverse_returns, self.returns['BTC-USD'])[0, 1]

                # Sélectionner les meilleures dérivées et inverse pour chaque actif
                if abs(correlation_first) > abs(correlation_second) and abs(correlation_first) > abs(correlation_inverse):
                    best_features.append(('Première Dérivée', first_derivative))
                elif abs(correlation_second) > abs(correlation_first) and abs(correlation_second) > abs(correlation_inverse):
                    best_features.append(('Seconde Dérivée', second_derivative))
                else:
                    best_features.append(('Inverse des Rendements', inverse_returns))

        # Créer une matrice de données avec les meilleures dérivées et inverses sélectionnées
        X = pd.DataFrame({f"{feature_name} de {col}": feature_data for (feature_name, feature_data), col in zip(best_features, self.returns.columns) if col != 'BTC-USD'})
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

    def derive_and_inverse_analysis(self):
        """
        Effectue des analyses statistiques en utilisant les dérivées des données du Bitcoin ainsi que leurs inverses, ainsi que des autres actifs pour voir leur pouvoir prédictif sur Bitcoin.
        """
        st.markdown("### Analyse des Dérivées et Inverses des Rendements 🚀")
        st.write("Cette analyse explore les dérivées des rendements de Bitcoin et des autres actifs pour mieux comprendre les variations et comportements.")

        btc_returns = self.returns['BTC-USD']
        all_derivatives = {}
        all_inverse_returns = {}

        for col in self.returns.columns:
            # Calcul des dérivées premières et secondes des rendements
            returns = self.returns[col]
            first_derivative = np.gradient(returns)
            second_derivative = np.gradient(first_derivative)

            # Inverses des rendements (en évitant la division par zéro)
            inverse_returns = returns.apply(lambda x: 1/x if x != 0 else 0)

            all_derivatives[col] = {
                'Première Dérivée': first_derivative,
                'Seconde Dérivée': second_derivative,
                'Inverse des Rendements': inverse_returns
            }
            all_inverse_returns[col] = inverse_returns

            # Calcul des statistiques descriptives
            st.write(f"### Statistiques Descriptives pour {col}")
            descriptive_stats = pd.DataFrame({
                'Première Dérivée': first_derivative,
                'Seconde Dérivée': second_derivative,
                'Inverse des Rendements': inverse_returns
            }).describe()
            st.write(descriptive_stats)

            # Visualisation des dérivées
            st.line_chart(pd.Series(first_derivative, index=self.returns.index), use_container_width=True)
            st.write(f"Graphique de la première dérivée des rendements de {col}")

            st.line_chart(pd.Series(second_derivative, index=self.returns.index), use_container_width=True)
            st.write(f"Graphique de la seconde dérivée des rendements de {col}")

            st.line_chart(pd.Series(inverse_returns, index=self.returns.index), use_container_width=True)
            st.write(f"Graphique des inverses des rendements de {col}")

        # Effectuer des analyses de corrélation et de prédiction entre les dérivées des autres actifs et Bitcoin
        st.markdown("### Analyse de Corrélation et de Prédiction entre les Dérivées des Actifs et Bitcoin 🔗")
        for col in self.returns.columns:
            if col != 'BTC-USD':
                st.write(f"### Analyse pour {col}")
                first_derivative = all_derivatives[col]['Première Dérivée']
                second_derivative = all_derivatives[col]['Seconde Dérivée']
                inverse_returns = all_inverse_returns[col]

                # Calculer la corrélation avec Bitcoin
                correlation_first = np.corrcoef(first_derivative, btc_returns)[0, 1]
                correlation_second = np.corrcoef(second_derivative, btc_returns)[0, 1]
                correlation_inverse = np.corrcoef(inverse_returns, btc_returns)[0, 1]

                st.write(f"Corrélation de la première dérivée avec Bitcoin : {correlation_first:.3f}")
                st.write(f"Corrélation de la seconde dérivée avec Bitcoin : {correlation_second:.3f}")
                st.write(f"Corrélation des inverses avec Bitcoin : {correlation_inverse:.3f}")

                # Effectuer une régression linéaire pour voir le pouvoir prédictif
                X = pd.DataFrame({
                    'Première Dérivée': first_derivative,
                    'Seconde Dérivée': second_derivative,
                    'Inverse des Rendements': inverse_returns
                })
                X = sm.add_constant(X)
                y = btc_returns

                model = sm.OLS(y, X).fit()
                st.write(model.summary())

        st.success("### Analyse terminée pour toutes les dérivées et inverses.")

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

    st.header('2. Analyse des Corrélations')
    analyzer.correlation_analysis()

    st.header('3. Analyse de la Causalité de Granger')
    analyzer.granger_causality()

    st.header('4. Analyse de la Co-intégration')
    analyzer.cointegration_analysis()

    st.header('5. Analyse de l'Information Mutuelle')
    analyzer.mutual_information_analysis()

    st.header('6. F-Test pour évaluer les variables explicatives')
    analyzer.f_test_analysis()

    st.header('7. Modèle Forêt Aléatoire avec Variables Significatives')
    analyzer.random_forest_model()

    st.header('8. Analyse des Dérivées et Inverses des Rendements des Actifs')
    analyzer.derive_and_inverse_analysis()

if __name__ == "__main__":
    main()
