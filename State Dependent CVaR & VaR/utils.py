import requests
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chisquare, norm, shapiro


# --- CVaR & VaR Class ---
class ValueAtRisk:
    def __init__(self, px_hist, hlds, percentile, alpha, lookback, bins):
        self.px_hist = px_hist
        self.hlds = hlds
        self.percentile = percentile
        self.alpha = alpha
        self.lookback = lookback
        self.bins = bins


    # Portfolio's Daily Returns
    def portfolio_returns(self):

        relevant_px_hist = self.px_hist[self.hlds.keys()].copy()

        for col in relevant_px_hist.columns:
            if col in list(self.hlds.keys()):
                relevant_px_hist.loc[:, col] = self.hlds[col] * relevant_px_hist[col]
            else:
                continue

        relevant_px_hist['Portfolio Value'] = relevant_px_hist.sum(axis=1)
        relevant_px_hist['Portfolio Pct Chg'] = relevant_px_hist['Portfolio Value'].pct_change()

        return relevant_px_hist
    

    # Test for Normality
    def shapiro_wilks(self, relevant_px_hist):
        relevant_px_hist = relevant_px_hist.dropna()
        lookback_relevant_px_hist = relevant_px_hist.tail(self.lookback)
        pf_returns = lookback_relevant_px_hist['Portfolio Pct Chg']
        results = shapiro(pf_returns)
        
        return f"Test for normality p-value of {round(results.pvalue, 4)} for {self.lookback} day lookback"
    
    
    # Test for Goodness of Fit
    def chi_square_test(self, relevant_px_hist):
        relevant_px_hist = relevant_px_hist.dropna()
        lookback_relevant_px_hist = relevant_px_hist.tail(self.lookback)
        
        binned = pd.cut(lookback_relevant_px_hist['Portfolio Pct Chg'], bins=self.bins)
        observed_counts = binned.value_counts().sort_index()

        mu = lookback_relevant_px_hist['Portfolio Pct Chg'].mean()
        sigma = lookback_relevant_px_hist['Portfolio Pct Chg'].std()

        bin_edges = binned.cat.categories

        expected_counts = np.zeros(self.bins)
        
        for i, bin_edge in enumerate(bin_edges):
            lower_cdf = norm.cdf(bin_edge.left, mu, sigma)
            upper_cdf = norm.cdf(bin_edge.right, mu, sigma)
            
            bin_probability = upper_cdf - lower_cdf
            
            expected_counts[i] = bin_probability * len(lookback_relevant_px_hist)

        if np.sum(observed_counts) != np.sum(expected_counts):
            expected_counts = expected_counts * (np.sum(observed_counts) / np.sum(expected_counts))

        chi_square_stat, p_value = chisquare(observed_counts, expected_counts)

        return f"Goodness of fit p-value of {round(p_value, 4)} for {self.lookback} day lookback"


    # Histogram of the Portfolio's Returns
    def histogram(self, relevant_px_hist):

        relevant_px_hist = relevant_px_hist.dropna()
        lookback_relevant_px_hist = relevant_px_hist.dropna().tail(self.lookback)
        
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        sns.histplot(lookback_relevant_px_hist['Portfolio Pct Chg'], bins=self.bins, edgecolor='black')

        plt.title(f"Distribution of Portfolio Daily Returns | Lookback: {self.lookback}", fontsize=20, fontweight='bold', color='black')
        plt.ylabel("Frequency", fontsize=12, fontweight='bold')
        plt.xlabel("Portfolio Pct Chg", fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')

        sns.despine()
        plt.tight_layout()


    # Normal Portfolio VaR
    def parametric_normal(self, relevant_px_hist):
        lookback_relevant_px_hist = relevant_px_hist.dropna().tail(self.lookback)
        pf_value = relevant_px_hist['Portfolio Value'].iloc[-1]
        pf_returns = lookback_relevant_px_hist['Portfolio Pct Chg']
        
        mu = pf_returns.mean()
        sigma = pf_returns.std()
        z_score = norm.ppf(self.alpha)
        percent_VaR = mu + z_score * sigma

        return f"{round(pf_value*percent_VaR, 2)} VaR for {self.lookback} day lookback"

    
    # Percentile Portfolio VaR
    def historic_percentile(self, relevant_px_hist):

        lookback_relevant_px_hist = relevant_px_hist.dropna().tail(self.lookback)
        pf_value = relevant_px_hist['Portfolio Value'].iloc[-1] 
        percent_VaR = np.percentile(lookback_relevant_px_hist['Portfolio Pct Chg'], self.percentile)
        
        return f"{round(pf_value*percent_VaR, 2)} VaR for {self.lookback} day lookback"
    
    # Monte-Carlo Portfolio VaR
    def monte_carlo(self, relevant_px_hist, n=10000):
        lookback_relevant_px_hist = relevant_px_hist.dropna().tail(self.lookback)
        pf_returns = lookback_relevant_px_hist['Portfolio Pct Chg']
        pf_value = relevant_px_hist['Portfolio Value'].iloc[-1] 

        mu = pf_returns.mean()
        sigma = pf_returns.std()

        samples = np.random.normal(loc=mu, scale=sigma, size=n)
        percent_VaR = np.percentile(samples, self.percentile)

        return f"{round(pf_value*percent_VaR, 2)} VaR for {self.lookback} day lookback"


# --- Data Analysis Funcitons ---
def plot_feature_frequency(decile, stats_dict, name, color: str='lightpink'):
    plt.figure(figsize=(10, 8))
    
    features_count = stats_dict['features_count']
    feature_names = list(features_count.keys())
    frequencies = list(features_count.values())
    
    plt.barh(feature_names, frequencies, color=color)
    
    plt.suptitle(f"Feature Frequency of Decile {decile} for {name}", fontsize=18, fontweight='bold')
    plt.title(f"Decile Average Silhouette Score: {round(stats_dict['score_avg'], 4)}")
    plt.xlabel('Frequency')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


# --- Fixed-Width Window Fracdiff for Stationarity ---
def get_weight_ffd(d, thres):
    w, k = [1.], 1
    while True:
        w_ = -w[-1]/k*(d-k+1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1,1)

def frac_diff_ffd(series, d, thres=1e-4):
    w = get_weight_ffd(d, thres)
    width = len(w) - 1
    df = {}
    
    for name in series.columns:
        
        seriesF = series[[name]].ffill().dropna()
        df_ = pd.Series(dtype=float)
        
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_.loc[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1].values)[0,0]
        
        df[name] = df_.copy(deep=True)

    result_df = pd.concat(df, axis=1)

    return result_df

def optimal_frac_diff_ffd(series, step=20, pval=0.05, thres=1e-4):
    for d in np.linspace(0, 1, step):
        df = frac_diff_ffd(series, d, thres)
        if all(adfuller(df[col])[1] < pval for col in df.columns): return df, d


# --- Data Processing Functions ---
def impute(frame: pd.DataFrame, cols_to_impute: list, start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.loc[start_date:, :].copy()
    X = frame[cols_to_impute]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace(['.', '', ' ', 'N/A', 'NA', 'null'], np.nan)

    cs_imputed_data = X.interpolate(method='cubic')
    lin_imputed_data = X.interpolate(method='linear')

    return (
            pd.DataFrame(cs_imputed_data, columns=X.columns, index=X.index), 
            pd.DataFrame(lin_imputed_data, columns=X.columns, index=X.index)
    )

def _remove_outliers(data: pd.DataFrame, sigma=6):
    for col in data.columns:
        st_dev = np.std(data[col])
        mu = np.mean(data[col])
        
        outlier_mask = (data[col] > mu + st_dev * sigma) | (data[col] < mu - st_dev * sigma)
    
        data.loc[outlier_mask, col] = np.nan

def remove_outliers(data: pd.DataFrame, sigma=6) -> pd.DataFrame:
    data_to_process = data.copy()
    _remove_outliers(data_to_process, sigma)
    
    return data_to_process


# --- Macroeconomic Data Loader ---
fred_key = '0e26fed1b95ca710abdb6bbde2ad1a8a'

fred_daily_id_dict = {
    "WTI": "DCOILWTICO",
    "Indeed Job Postings": "IHLIDXNEWUS",
    "CBOE VIX": "VIXCLS",
    "5YR Breakeven Inflation": "T5YIE",
    "10YR Breakeven Inflation": "T10YIE",
    "Treasury 10YR minus 2YR": "T10Y2Y",
    "Treasury 10YR minus 3M": "T10Y3M",
    "HY OAS": "BAMLH0A0HYM2",
}

fred_weekly_id_dict = {
    "Initial Claims SA": "ICSA"
}

fred_monthly_id_dict = {
    "U3 SA": "UNRATE",
    "U6 SA": "U6RATE",
    "PCE SA": "PCE",
    "CPI SA": "CPIAUCSL",
    "Number Unemployed SA": "UNEMPLOY",
    "Job Openings SA": "JTSJOL"
}

def load_econ(data: dict) -> dict:
    econ_data_dict = {}
    for k, v in data.items():
        url = f'https://api.stlouisfed.org/fred/series/observations?series_id={v}&api_key={fred_key}&file_type=json'
        econ_data = requests.get(url).json()
        econ_data_dict[k] = {'date' : [observation['date'] for observation in econ_data['observations']],
                             k : [observation['value'] for observation in econ_data['observations']]}
    
    return econ_data_dict