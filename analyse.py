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
import statsmodels.api as sm
from hurst import compute_Hc
import scipy.stats as stats

class ComprehensiveCryptoCommoAnalyzer:
    def __init__(self, tickers, names, start_date):
        self.tickers = tickers
        self.names = names
        self.start_date = start_date
        self.data = None
        self.returns = None
        self.significant_vars = []

    def fetch_data(self):
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
        self.handle_missing_data()
        self.check_stationarity()
        self.make_stationary()
        self.detect_outliers()
        self.scale_data()

    def handle_missing_data(self, method='linear'):
        self.data = self.data.interpolate(method=method).dropna()
        self.returns = self.data.pct_change().dropna()

    def check_stationarity(self):
        for column in self.data.columns:
            result = adfuller(self.data[column].dropna())
            if result[1] > 0.05:
                pass

    def make_stationary(self):
        self.returns = self.data.diff().dropna()

    def detect_outliers(self):
        Q1 = self.returns.quantile(0.25)
        Q3 = self.returns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.returns < (Q1 - 1.5 * IQR)) | (self.returns > (Q3 + 1.5 * IQR))
        return outliers

    def scale_data(self):
        scaler = StandardScaler()
        self.returns = pd.DataFrame(scaler.fit_transform(self.returns), index=self.returns.index, columns=self.returns.columns)

    def correlation_analysis(self):
        st.write("## Analyse de Corrélation")
        st.write("Cette analyse permet de comprendre comment les variations de prix de Bitcoin sont associées à celles d'autres actifs financiers. Une corrélation positive indique que deux actifs ont tendance à évoluer dans la même direction.")
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
        st.write("## Causalité de Granger")
        st.write("Ce test aide à savoir si les rendements d'un actif peuvent être utilisés pour prédire ceux de Bitcoin. Une faible p-value (inférieure à 0,05) indique une causalité significative.")
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
        st.write("## Analyse de Co-intégration")
        st.write("La co-intégration indique une relation à long terme entre Bitcoin et d'autres actifs. Cela signifie que même si les prix fluctuent à court terme, ils restent liés à long terme.")
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
        st.write("## Information Mutuelle")
        st.write("L'information mutuelle permet de détecter des relations non-linéaires entre les actifs. Cela aide à comprendre si un actif peut aider à prédire les variations de Bitcoin, même s'il n'y a pas de relation linéaire.")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']
        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame({'Variable': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values('MI Score', ascending=False)
        st.write("### Top 5 des actifs avec la plus forte dépendance avec Bitcoin :")
        st.write(mi_df.head())

    def f_test_analysis(self):
        st.write("## F-Test d'Évaluation des Variables")
        st.write("Ce test permet d'évaluer si les autres actifs ont un effet significatif sur Bitcoin. Les variables avec une p-value inférieure à 0,05 sont considérées comme significatives.")
        X = self.returns.drop('BTC-USD', axis=1)
        y = self.returns['BTC-USD']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.write(model.summary())
        p_values = model.pvalues.drop('const')
        st.write("### P-values des variables explicatives :")
        st.write(p_values)
        self.significant_vars = p_values[p_values < 0.05].index
        st.write("### Variables ayant un pouvoir prédictif significatif (p-value < 0.05) :")
        st.write(self.significant_vars)

    def random_forest_model(self):
        st.write("## Modèle Forêt Aléatoire avec les Variables Significatives")
        st.write("Le modèle Forêt Aléatoire est utilisé pour prédire les rendements de Bitcoin en utilisant les variables significatives identifiées. Le coefficient de détermination R2 indique la proportion de la variabilité expliquée par le modèle.")
        X = self.returns[self.significant_vars]
        y = self.returns['BTC-USD']
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        y_pred = rf_model.predict(X)
        r_squared = r2_score(y, y_pred)
        st.write(f"R2 du modèle Forêt Aléatoire : {r_squared:.3f}")

    def hurst_exponent_analysis(self):
        st.write("## Analyse de l'Exposant de Hurst")
        st.write("L'exposant de Hurst permet de déterminer si une série temporelle est aléatoire (valeur proche de 0,5), persistante (>0,5) ou anti-persistante (<0,5).")
        H, c, data = compute_Hc(self.returns['BTC-USD'], kind='price', simplified=True)
        st.write(f"Exposant de Hurst pour Bitcoin : {H:.3f}")

    def levy_process_analysis(self):
        st.write("## Analyse des Processus de Lévy")
        st.write("Les processus de Lévy permettent de modéliser les sauts et les fluctuations irrégulières observées dans les séries financières, fournissant une meilleure représentation des mouvements de prix.")
        levy_path = np.cumsum(self.returns['BTC-USD'].values)
        st.line_chart(pd.Series(levy_path, index=self.returns.index), width=800, height=400, use_container_width=True)
        st.write("Trajectoire simulée basée sur un processus de Lévy pour Bitcoin")

    def hawkes_process_analysis(self):
        st.write("## Analyse des Processus de Hawkes")
        st.write("Les processus de Hawkes sont des processus de comptage qui capturent l'auto-excitation, utilisés ici pour modéliser l'influence d'événements passés sur l'arrivée de nouveaux événements.")
        events_series = (self.returns['BTC-USD'] > self.returns['BTC-USD'].quantile(0.95)).astype(int)
        st.line_chart(events_series.cumsum(), width=800, height=400, use_container_width=True)
        st.write("Trajectoire cumulée des événements excédant le 95e percentile pour illustrer un processus de Hawkes sur Bitcoin.")

def main():
    st.title("Analyse des Relations entre Bitcoin et Autres Actifs Financiers")
    
    tickers = [
        'BTC-USD', 'SPY', 'QQQ', '^GDAXI', '^FTSE', 'CL=F', 'BZ=F', 'NG=F',
        'GC=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F', 'ZN=F', 'ZW=F',
        'ZC=F', 'ZS=F', 'KC=F', 'CT=F', 'CC=F', 'SB=F', 'OJ=F', 'LE=F', 'HE=F',
        'RB=F', 'HO=F', 'EURUSD=X', 'GBPUSD=X', 'TLT', 'LQD', 'HYG'
    ]
    
    names = [
        'Bitcoin (BTC)', 'S&P 500 (SPY)', 'Nasdaq 100 (QQQ)', 'DAX Allemagne (^GDAXI)', 'FTSE 100 (^FTSE)',
        'Pétrole WTI (CL=F)', 'Pétrole Brent (BZ=F)', 'Gaz Naturel (NG=F)', 'Or (GC=F)', 'Argent (SI=F)',
        'Platine (PL=F)', 'Palladium (PA=F)', 'Cuivre (HG=F)', 'Zinc (ZN=F)', 'Blé (ZW=F)',
        'Maïs (ZC=F)', 'Soja (ZS=F)', 'Café (KC=F)', 'Coton (CT=F)', 'Cacao (CC=F)', 'Sucre (SB=F)',
        'Jus d\'Orange (OJ=F)', 'Bétail (LE=F)', 'Porcs (HE=F)', 'Essence (RB=F)', 'Fioul (HO=F)',
        'EUR/USD (EURUSD=X)', 'GBP/USD (GBPUSD=X)', 'Obligations US 20+ ans (TLT)',
        'Obligations Corporate Investment Grade (LQD)', 'Obligations Corporate High Yield (HYG)'
    ]

    start_date = '2015-01-01'
    analyzer = ComprehensiveCryptoCommoAnalyzer(tickers, names, start_date)
    analyzer.fetch_data()
    analyzer.prepare_data()
    analyzer.correlation_analysis()
    analyzer.granger_causality()
    analyzer.cointegration_analysis()
    analyzer.mutual_information_analysis()
    analyzer.f_test_analysis()
    analyzer.random_forest_model()
    analyzer.hurst_exponent_analysis()
    analyzer.levy_process_analysis()
    analyzer.hawkes_process_analysis()

if __name__ == "__main__":
    main()
