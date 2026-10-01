"""Generate dpcgans_hands_on.ipynb. Edit this file and re-run it; do not hand-edit the notebook."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
               "colab": {"name": "dpcgans_hands_on.ipynb", "provenance": []}}
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# DP-CGANS hands-on: train your own synthetic data generator

**Unlock Your Sensitive Data: Generate Your Own Synthetic Dataset** · TDCC-SSH workshop · 14 October 2026
Raoul Schram (Utrecht University) and Chang Sun (Maastricht University)

In the next 15 minutes you will

1. look at a small, real, public dataset,
2. train a **DP-CGAN** (a differentially private conditional generative adversarial network) on it and watch it learn,
3. compare synthetic data with the real data on **fidelity, utility and privacy**, using a generator we trained longer in advance,
4. switch differential privacy on, train again, and see what it costs.

Everything runs in this notebook. Run the cells top to bottom with **Shift+Enter**, or use *Runtime → Run all*.
Nothing here is sensitive: the data is the public UCI census income dataset.

> **Tip.** A GPU is not needed. On the free Colab CPU one quick training run takes one to two minutes.
> The longer-trained generators are downloaded, not trained here.
""")

md("## 1. Set up (about 1 minute)")
code(r"""
%%capture
!pip install -q dp-cgans
""")
code(r"""
import os, io, contextlib, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dp_cgans import DP_CGAN

sns.set_theme(style="whitegrid", font_scale=0.9)
PALETTE = {"real": "#4C72B0", "synthetic": "#DD8452", "synthetic (DP)": "#55A868",
           "quick, 40 epochs": "#BBBBBB", "quick DP, 40 epochs": "#999999"}

# Where the workshop files live (data and pre-trained generators)
REPO_RAW = os.environ.get("WORKSHOP_REPO_RAW", "https://raw.githubusercontent.com/sunchang0124/syndata-workshop/main")

# Settings you can play with later
N_QUICK = 2000     # rows used for the quick training run in this notebook
EPOCHS  = 40       # quick-run training rounds; more = better fidelity, slower, and (with DP) more privacy budget
PRETRAINED_ROWS, PRETRAINED_EPOCHS = 5000, 300   # what the generators we trained in advance saw
SEED    = 2026

print("torch", torch.__version__, "| GPU available:", torch.cuda.is_available())
""")

md(r"""
## 2. The real data (about 2 minutes)

We use the **UCI census income** data: 46 012 adults from the 1994 US census. We kept seven columns and grouped some categories so that a generator trains quickly:

| column | type | values |
|---|---|---|
| `age` | numeric | 17 to 90 |
| `education` | categorical | No high school, High school, Some college, Bachelor or higher |
| `marital_status` | categorical | Never married, Married, Previously married |
| `sex` | categorical | Female, Male |
| `workclass` | categorical | Private, Government, Self-employed |
| `hours_per_week` | numeric | 1 to 99 |
| `income` | categorical | low (at most 50k dollars a year), high |

Think of it as a stand-in for a survey or registry extract you are not allowed to share.
""")
code(r"""
def load_census():
    "Load the prepared workshop table; fall back to building it from OpenML if the repo is unreachable."
    try:
        return pd.read_csv(f"{REPO_RAW}/data/census_workshop.csv")
    except Exception as e:
        print("Could not reach the workshop repository, building the table from OpenML instead:", e)
        from sklearn.datasets import fetch_openml
        ad = fetch_openml("adult", version=2, as_frame=True).frame
        edu = lambda x: ("Bachelor or higher" if x in ["Bachelors", "Masters", "Doctorate", "Prof-school"]
                         else "Some college" if x in ["Some-college", "Assoc-acdm", "Assoc-voc"]
                         else "High school" if x == "HS-grad" else "No high school")
        mar = lambda m: "Married" if str(m).startswith("Married") else "Never married" if m == "Never-married" else "Previously married"
        wc  = lambda w: ("Private" if w == "Private" else "Government" if isinstance(w, str) and "gov" in w
                         else "Self-employed" if isinstance(w, str) and w.startswith("Self") else np.nan)
        df = pd.DataFrame({"age": ad["age"].astype(int), "education": ad["education"].map(edu),
                           "marital_status": ad["marital-status"].map(mar), "sex": ad["sex"].astype(str),
                           "workclass": ad["workclass"].map(wc), "hours_per_week": ad["hours-per-week"].astype(int),
                           "income": ad["class"].map(lambda c: "high" if c == ">50K" else "low")})
        return df.dropna().reset_index(drop=True)

census = load_census()
CATEGORICAL = [c for c in census.columns if census[c].dtype == object]
NUMERIC     = [c for c in census.columns if c not in CATEGORICAL]

# `real_train`: the 5 000 rows the pre-trained generators learned from. The quick run below uses 2 000 of them.
# `real_holdout`: rows no generator has ever seen, kept aside to test utility and privacy honestly.
real_train   = census.sample(PRETRAINED_ROWS, random_state=42).reset_index(drop=True)
real_holdout = census.drop(census.sample(PRETRAINED_ROWS, random_state=42).index).sample(2000, random_state=SEED).reset_index(drop=True)
quick_train  = real_train.sample(N_QUICK, random_state=SEED).reset_index(drop=True)

print(f"{len(census):,} rows in total | {len(real_train):,} seen by the pre-trained generators | "
      f"{len(quick_train):,} used for the quick run | {len(real_holdout):,} kept aside")
real_train.head(8)
""")
code(r"""
# A first look: how is each column distributed?
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, col in zip(axes.ravel(), census.columns):
    if col in NUMERIC:
        sns.histplot(real_train[col], bins=30, ax=ax, color=PALETTE["real"])
    else:
        real_train[col].value_counts(normalize=True).sort_index().plot.bar(ax=ax, color=PALETTE["real"])
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(col); ax.set_xlabel(""); ax.set_ylabel("")
axes.ravel()[-1].axis("off")
plt.suptitle("Real training data", y=1.02); plt.tight_layout(); plt.show()

print("Share of high income by education (the kind of relation we want the generator to learn):")
print(pd.crosstab(real_train["education"], real_train["income"], normalize="index").round(2))
""")

md(r"""
## 3. Train a generator and watch it learn (about 3 minutes)

`DP_CGAN` has a lot of knobs. The ones that matter today:

* `epochs`: how many rounds the generator and discriminator play against each other,
* `batch_size`: how many rows they look at per step,
* `private`: `False` now, `True` in step 5 to switch differential privacy on.

Everything else is the published default. Run the cell and watch the progress bar. Forty epochs on 2 000 rows is deliberately short: enough to see the mechanics, not enough for good data. A GAN typically needs a few hundred passes.
""")
code(r"""
def train_dpcgan(train_df, epochs=EPOCHS, private=False, batch_size=500, seed=SEED):
    "Train a DP-CGAN and return (model, seconds, epsilon). epsilon is None when private=False."
    # dp-cgans caches a fitted transformer in the working directory and silently reuses it,
    # which breaks when the data changes. Always start clean.
    if os.path.exists("fitted_transformer.pkl"):
        os.remove("fitted_transformer.pkl")
    torch.manual_seed(seed); np.random.seed(seed)
    log = io.StringIO(); t0 = time.time()
    with contextlib.redirect_stdout(log):          # keep the progress bar, hide the per-epoch chatter
        model = DP_CGAN(epochs=epochs, batch_size=batch_size, private=private,
                        cuda=torch.cuda.is_available(), verbose=False, saved_transformer=None)
        model.fit(train_df)
    seconds = time.time() - t0
    eps_lines = [l for l in log.getvalue().splitlines() if "differential privacy with eps" in l]
    epsilon = float(eps_lines[-1].split("eps = ")[1].split(" ")[0]) if eps_lines else None
    print(f"trained {epochs} epochs on {len(train_df):,} rows in {seconds:.0f} s"
          + (f" | differential privacy: epsilon = {epsilon:.2f}" if epsilon is not None else " | no differential privacy"))
    return model, seconds, epsilon

model_quick, _, _ = train_dpcgan(quick_train, epochs=EPOCHS, private=False)
""")
code(r"""
# Sampling is instant once the generator is trained. Ask for as many rows as you like.
synthetic_quick = model_quick.sample(len(quick_train))
synthetic_quick.to_csv("synthetic_census_quick.csv", index=False)

# A first comparison: the categories are right, the proportions usually are not yet.
pd.concat({"real": quick_train["marital_status"].value_counts(normalize=True),
           "synthetic, 40 epochs": synthetic_quick["marital_status"].value_counts(normalize=True)}, axis=1).round(2)
""")

md(r"""
## 4. Compare real and synthetic (about 4 minutes)

To judge synthetic data fairly we need a generator that has finished learning. We trained the same model for **300 epochs on 5 000 rows** before the workshop (six minutes on a laptop). The next cell downloads it; if the download fails it falls back to the synthetic rows we sampled from it.

Then three questions, three checks:

* **Fidelity**: does each column look the same? Do the columns relate in the same way?
* **Utility**: if I train a classifier on the synthetic data, does it work on real data?
* **Privacy**: how close do synthetic rows come to real rows?
""")
code(r"""
import urllib.request

def load_pretrained(name, n_rows=5000):
    "Download a generator from the workshop repository and sample from it. Falls back to the synthetic CSV sampled from it."
    path = f"dpcgan_{name}.pkl"
    try:
        urllib.request.urlretrieve(f"{REPO_RAW}/models/{path}", path)
        model = DP_CGAN.load(path)
        print(f"loaded {path}")
        return model.sample(n_rows), model
    except Exception as e:
        print(f"could not load the model ({str(e)[:80]}); using the synthetic CSV we sampled from it instead")
        return pd.read_csv(f"{REPO_RAW}/data/synthetic_{name}.csv"), None

synthetic, model_plain = load_pretrained("plain_300ep")
synthetic.to_csv("synthetic_census.csv", index=False)
synthetic.head(5)
""")
code(r"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def _encode(df, columns=None):
    "One-hot categoricals, keep numerics, so that every table has the same columns."
    X = pd.get_dummies(df, columns=[c for c in CATEGORICAL if c in df.columns], dtype=float)
    return X.reindex(columns=columns, fill_value=0.0) if columns is not None else X

def plot_distributions(real, synths):
    "synths: dict name -> dataframe. Draws real vs each synthetic set, one panel per column."
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), real.columns):
        if col in NUMERIC:
            sns.kdeplot(real[col], ax=ax, color=PALETTE["real"], label="real", fill=True, alpha=.3, clip=(real[col].min(), real[col].max()))
            for name, s in synths.items():
                sns.kdeplot(s[col], ax=ax, color=PALETTE.get(name, None), label=name, clip=(real[col].min(), real[col].max()))
        else:
            tab = pd.concat([real[col].value_counts(normalize=True).rename("real")]
                            + [s[col].value_counts(normalize=True).rename(n) for n, s in synths.items()], axis=1).fillna(0)
            tab.plot.bar(ax=ax, color=[PALETTE["real"]] + [PALETTE.get(n) for n in synths], legend=False)
            ax.tick_params(axis="x", rotation=30)
        ax.set_title(col); ax.set_xlabel(""); ax.set_ylabel("")
    axes.ravel()[-1].axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.12))
    plt.tight_layout(); plt.show()

def plot_correlations(real, synth, name="synthetic"):
    "Correlation of every (one-hot encoded) column with every other, real vs synthetic, plus the difference."
    R = _encode(real); S = _encode(synth, R.columns)
    cr, cs = R.corr(), S.corr()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, m, t in zip(axes, [cr, cs, (cr - cs).abs()], ["real", name, "absolute difference"]):
        sns.heatmap(m, ax=ax, vmin=-1 if t != "absolute difference" else 0, vmax=1, cmap="RdBu_r" if t != "absolute difference" else "Reds",
                    cbar=False, xticklabels=False, yticklabels=(ax is axes[0]))
        ax.set_title(t)
    plt.tight_layout(); plt.show()
    return float((cr - cs).abs().values[np.triu_indices_from(cr, 1)].mean())

def utility_tstr(train_on, test_on, target="income"):
    "Train a random forest on `train_on`, test it on `test_on`. Returns (accuracy, balanced accuracy)."
    cols = _encode(real_train.drop(columns=target)).columns
    Xtr, ytr = _encode(train_on.drop(columns=target), cols), train_on[target]
    Xte, yte = _encode(test_on.drop(columns=target), cols), test_on[target]
    clf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred), balanced_accuracy_score(yte, pred)

def privacy_dcr(real_tr, real_ho, synth):
    "Distance from every synthetic row to its closest real training row, compared with real hold-out rows (the honest baseline)."
    cols = _encode(real_tr).columns
    scaler = StandardScaler().fit(_encode(real_tr))
    nn = NearestNeighbors(n_neighbors=1).fit(scaler.transform(_encode(real_tr)))
    d_syn = nn.kneighbors(scaler.transform(_encode(synth, cols)))[0].ravel()
    d_ho  = nn.kneighbors(scaler.transform(_encode(real_ho, cols)))[0].ravel()
    return d_syn, d_ho

def evaluate(synth, name="synthetic", show=True, also=None):
    "Run all three checks and return one row of numbers. `also` = dict of extra synthetic sets to overlay in the distribution plot."
    acc_rr, bacc_rr = utility_tstr(real_train, real_holdout)
    acc_sr, bacc_sr = utility_tstr(synth, real_holdout)
    d_syn, d_ho = privacy_dcr(real_train, real_holdout, synth)
    if show:
        plot_distributions(real_train, {**(also or {}), name: synth})
        corr_gap = plot_correlations(real_train, synth, name)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        sns.kdeplot(d_ho, ax=ax, label="real hold-out rows (baseline)", color=PALETTE["real"], fill=True, alpha=.3)
        sns.kdeplot(d_syn, ax=ax, label=name, color=PALETTE.get(name))
        ax.set_xlabel("distance to the closest real training row"); ax.set_title("Privacy: are synthetic rows suspiciously close to real ones?")
        ax.legend(); plt.tight_layout(); plt.show()
    else:
        R = _encode(real_train); S = _encode(synth, R.columns)
        corr_gap = float((R.corr() - S.corr()).abs().values[np.triu_indices_from(R.corr(), 1)].mean())
    row = pd.Series({
        "fidelity: mean |correlation difference| (0 = identical)": round(corr_gap, 3),
        "utility: accuracy, train on real, test on real": round(acc_rr, 3),
        "utility: accuracy, train on synthetic, test on real": round(acc_sr, 3),
        "utility: balanced accuracy, train on real, test on real": round(bacc_rr, 3),
        "utility: balanced accuracy, train on synthetic, test on real": round(bacc_sr, 3),
        "privacy: median distance to closest real row, synthetic": round(float(np.median(d_syn)), 3),
        "privacy: median distance to closest real row, real hold-out": round(float(np.median(d_ho)), 3),
        "privacy: share of synthetic rows identical to a real training row": round(float((d_syn < 1e-9).mean()), 3),
        "privacy: share of real hold-out rows identical to a real training row": round(float((d_ho < 1e-9).mean()), 3),
    }, name=name)
    return row

print("Evaluation functions ready.")
""")
code(r"""
results = {}
results["quick, 40 epochs"] = evaluate(synthetic_quick, "quick, 40 epochs", show=False)
results["synthetic"] = evaluate(synthetic, "synthetic", also={"quick, 40 epochs": synthetic_quick})
pd.DataFrame(results)
""")
md(r"""
**How to read this**

* *Distributions*: orange (300 epochs) should trace blue (real). Grey is your 40-epoch model: same categories, wrong proportions. Numeric columns with spikes (hours per week = 40) are the hardest.
* *Correlations*: the third heat-map should be pale. Bright cells are relations the generator missed.
* *Utility*: the two accuracies should be close. Look at **balanced** accuracy too: only one in four people has a high income, so a classifier that always says "low" already scores 0.75. A balanced accuracy near 0.5, as for the quick model, means the synthetic data does not carry the relation between income and the other columns.
* *Privacy*: synthetic rows should not sit closer to the training rows than real hold-out rows do. If they do, the generator is copying. Compare the two "identical rows" shares: with seven coarse columns many different people have exactly the same row, so real hold-out rows coincide with training rows too. A synthetic share far above that baseline would be a warning sign.

Write down the accuracy gap. You will compare it with the private model next.
""")

md(r"""
## 5. Switch privacy on and train again (about 4 minutes)

With `private=True`, DP-CGANS clips the discriminator's weights and adds calibrated noise to its gradients while it learns.
This gives a formal **differential privacy** guarantee, summarised by $\varepsilon$ (epsilon): the smaller, the stronger.
The budget is spent epoch by epoch, so $\varepsilon$ grows as training continues. It also grows faster on small training sets, because every row is seen more often per epoch.

Expect a value around 30 after 40 epochs on 2 000 rows. That is a **weak** guarantee: values below 10 are what you would aim for when publishing. Fewer epochs, or more training rows, bring it down. Today the point is to watch the dial move and see what it costs.
""")
code(r"""
model_quick_dp, _, epsilon = train_dpcgan(quick_train, epochs=EPOCHS, private=True)
synthetic_quick_dp = model_quick_dp.sample(len(quick_train))

# and the private generator we trained in advance: 100 epochs on 5 000 rows
synthetic_dp, model_dp = load_pretrained("private_100ep")
synthetic_dp.to_csv("synthetic_census_dp.csv", index=False)
synthetic_dp.head(5)
""")
code(r"""
results["quick DP, 40 epochs"] = evaluate(synthetic_quick_dp, "quick DP, 40 epochs", show=False)
results["synthetic (DP)"] = evaluate(synthetic_dp, "synthetic (DP)", also={"synthetic": synthetic})
pd.DataFrame(results)[["synthetic", "synthetic (DP)", "quick, 40 epochs", "quick DP, 40 epochs"]]
""")
md(r"""
**Questions for the discussion**

1. Compare the two pre-trained columns: how much fidelity and utility did differential privacy cost? Would you accept that trade for a dataset you could publish openly?
2. Which columns suffered most? Why might that be?
3. The non-private model has no guarantee, but its rows were not closer to real rows than the hold-out rows were. Is that enough evidence of privacy? For whom?
""")

md(r"""
## 6. If you have time left

Pick one:

* **Train longer.** Set `EPOCHS = 100` in step 1 and re-run step 3. Does the grey curve move towards the orange one? What happens to $\varepsilon$ in step 5? In our tests, even 200 epochs on 2 000 rows stayed unstable: GANs want data as well as time.
* **Sample more than you had.** `model_plain.sample(50_000)` gives you 50 000 rows from a model that saw 5 000. What stays right and what breaks? Try `plot_distributions(real_train, {"50k": model_plain.sample(50_000)})`.
* **Your own table.** Run the next cell to upload a CSV. **Public or already anonymised data only.** Keep it to about 10 columns with few categories each, or training will be slow.
* **Compare with metasyn.** Export the metasyn output from the previous session, upload it with the next cell, and run `evaluate()` on it. Same columns, different end of the trade-off.
""")
code(r"""
# Optional: upload your own CSV (Colab only). Then: my_model, _, _ = train_dpcgan(my_df, epochs=40)
try:
    from google.colab import files
    uploaded = files.upload()
    my_df = pd.read_csv(next(iter(uploaded)))
    print(my_df.shape); my_df.head()
except ImportError:
    print("Not running in Colab: load your CSV with pd.read_csv(path) instead.")
""")
code(r"""
# Optional: download what you made
try:
    from google.colab import files
    for f in ["synthetic_census_quick.csv", "synthetic_census.csv", "synthetic_census_dp.csv"]: files.download(f)
except ImportError:
    print("Files are in the working directory:", [f for f in os.listdir() if f.endswith(".csv")])
""")

md(r"""
## Learn more

* Package: `pip install dp-cgans` · code and issues: https://github.com/sunchang0124/dp_cgans
* Paper: Sun C, van Soest J, Dumontier M. *Generating synthetic personal health data using conditional generative adversarial networks combining with differential privacy.* Journal of Biomedical Informatics 143 (2023) 104404. https://doi.org/10.1016/j.jbi.2023.104404
* metasyn (the other tool in this workshop): https://metasyn.readthedocs.io
* TDCC-SSH project *Synthetic data: leveraging the potential of sensitive data in SSH research*: Utrecht University, DANS, SURF, Maastricht University.

Questions after the workshop: chang.sun@maastrichtuniversity.nl
""")

nb.cells = C
out = __file__.rsplit("/scripts/", 1)[0] + "/dpcgans_hands_on.ipynb"
nbf.write(nb, out); print("wrote", out, len(C), "cells")
