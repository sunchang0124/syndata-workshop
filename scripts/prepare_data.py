"""Build the workshop table from the UCI Adult census data (OpenML id 'adult', v2).

Seven columns, categories collapsed so a DP-CGAN trains in ~2 minutes on a laptop CPU.
Writes data/census_workshop.csv (all cleaned rows) next to this script's parent folder.
"""
import numpy as np, pandas as pd
from sklearn.datasets import fetch_openml

ad = fetch_openml("adult", version=2, as_frame=True).frame
df = pd.DataFrame({
    "age": ad["age"].astype(int),
    "education": ad["education"].map(lambda e: "Bachelor or higher" if e in ["Bachelors", "Masters", "Doctorate", "Prof-school"]
                                     else "Some college" if e in ["Some-college", "Assoc-acdm", "Assoc-voc"]
                                     else "High school" if e == "HS-grad" else "No high school"),
    "marital_status": ad["marital-status"].map(lambda m: "Married" if str(m).startswith("Married")
                                               else "Never married" if m == "Never-married" else "Previously married"),
    "sex": ad["sex"].astype(str),
    "workclass": ad["workclass"].map(lambda w: "Private" if w == "Private"
                                     else "Government" if isinstance(w, str) and "gov" in w
                                     else "Self-employed" if isinstance(w, str) and w.startswith("Self") else np.nan),
    "hours_per_week": ad["hours-per-week"].astype(int),
    "income": ad["class"].map(lambda c: "high" if c == ">50K" else "low"),
}).dropna().reset_index(drop=True)
out = __file__.rsplit("/scripts/", 1)[0] + "/data/census_workshop.csv"
df.to_csv(out, index=False)
print(df.shape, "->", out); print(df.head()); print({c: df[c].nunique() for c in df.columns})
