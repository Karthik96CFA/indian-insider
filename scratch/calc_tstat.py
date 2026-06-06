import json
import numpy as np
import pandas as pd
from pathlib import Path
import ast
import math

cache_file = Path('C:/Users/karth/indian-insider/.state/yfinance_cache.json')
if not cache_file.exists():
    print("No cache file found")
    exit(1)

with open(cache_file, "r") as f:
    data = json.load(f)
    cache = {ast.literal_eval(k): v for k, v in data.items()}

# Extract unique tickers and dates
tickers = list(set(k[0] for k in cache.keys()))

prices = {t: {} for t in tickers}
for (ticker, date_str, _), val in cache.items():
    entry_p, exit_p = val
    prices[ticker][date_str] = entry_p

prices_df = pd.DataFrame(prices).sort_index().ffill().bfill()
returns_df = prices_df.pct_change().dropna()

g1 = ['INFY', 'HDFCBANK', 'ICICIBANK', 'LT', 'RELIANCE', 'TCS']
g3 = ['AGARIND', 'CHEMCON', 'ITDC', 'MOS', 'PARAS', 'RAMCOSYS']

g1_cols = [c for c in g1 if c in returns_df.columns]
g3_cols = [c for c in g3 if c in returns_df.columns]

g1_returns = returns_df[g1_cols].mean(axis=1)
g3_returns = returns_df[g3_cols].mean(axis=1)

diff = g1_returns - g3_returns
mean_diff = diff.mean()
std_diff = diff.std()
n = len(diff)

se = std_diff / math.sqrt(n) if n > 0 else 1.0
t_stat = mean_diff / se if se > 0 else 0.0

# Approximate p-value using normal distribution (two-tailed)
# p = 2 * (1 - cdf(|z|))
# cdf(|z|) approx using standard normal approximation
z = abs(t_stat)
# error function approximation
def erf(x):
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y

p_val = 1.0 - erf(z / math.sqrt(2.0))

print(f"Mean G1: {g1_returns.mean()*100:.4f}%")
print(f"Mean G3: {g3_returns.mean()*100:.4f}%")
print(f"Mean Diff: {mean_diff*100:.4f}%")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.6f}")
