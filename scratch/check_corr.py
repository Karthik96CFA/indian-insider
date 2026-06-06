import sqlite3
import pandas as pd

conn = sqlite3.connect('C:/Users/karth/indian-insider/.state/state.db')
query = """
SELECT 
    ticker,
    fundamental_score as Quality, 
    multibagger_score as Growth, 
    valuation_score as Valuation, 
    event_score as Momentum, 
    canslim_score as Institutional, 
    industry_tailwind_score as Tailwind, 
    credibility_score as Credibility 
FROM company_scores
WHERE ticker IN (SELECT DISTINCT ticker FROM management_promises)
"""
df = pd.read_sql_query(query, conn).fillna(0.0)
print(df.head(10))
