"""Generate dpcgans_hands_on.ipynb. Edit this file and re-run it; do not hand-edit the notebook.
The notebook follows the usage example on https://pypi.org/project/dp-cgans/ as closely as possible."""
import os
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
               "colab": {"name": "dpcgans_hands_on.ipynb", "provenance": []}}
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))
REPO = os.environ.get("WORKSHOP_REPO_RAW", "https://raw.githubusercontent.com/sunchang0124/syndata-workshop/main")

md(r"""
# Generate your own synthetic data with DP-CGANS

**Unlock Your Sensitive Data** · TDCC-SSH workshop · 14 October 2026 · Raoul Schram (Utrecht University) and Chang Sun (Maastricht University)

Run the cells from top to bottom with **Shift+Enter**. In 15 minutes you will train a generator on a small public dataset, create synthetic rows from it, look at how close they are to the real data, and switch differential privacy on.
""")

md("## 1. Install the package")
code(r"""
%%capture
!pip install dp-cgans
""")

md(r"""
## 2. Load the data

Census income data on 2 000 adults: age, education, marital status, sex, type of employer, working hours and income. Public data, so nothing sensitive here. Think of it as a stand-in for a survey you are not allowed to share.
""")
code(rf"""
import pandas as pd

tabular_data = pd.read_csv("{REPO}/data/census_workshop.csv").sample(2000, random_state=0)
tabular_data.head()
""")

md(r"""
## 3. Train the generator

This is the usage example from the package documentation. Forty rounds of training takes about a minute. Watch the progress bar.
""")
code(r"""
from dp_cgans import DP_CGAN
import time, os

if os.path.exists("fitted_transformer.pkl"):   # the package reuses an old transformer if it finds one; start fresh
    os.remove("fitted_transformer.pkl")

model = DP_CGAN(
    epochs=40,            # number of training rounds
    batch_size=500,       # rows per training step
    private=False,        # differential privacy off for now
    verbose=False,
    saved_transformer=None,
)

start_time = time.time()
model.fit(tabular_data)
print("Training took", round(time.time() - start_time), "seconds")

model.save("generator.pkl")
""")

md("## 4. Generate synthetic rows")
code(r"""
syn_data = model.sample(2000)
syn_data.to_csv("syn_data_file.csv", index=None)
syn_data.head()
""")

md(r"""
## 5. Does it look like the real data?

Blue is real, orange is synthetic. After only 40 rounds the generator knows the categories but usually not yet the right proportions.
""")
code(r"""
import matplotlib.pyplot as plt

def compare(real, synthetic, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), real.columns):
        if real[col].dtype == object:
            pd.DataFrame({"real": real[col].value_counts(normalize=True),
                          "synthetic": synthetic[col].value_counts(normalize=True)}).plot.bar(ax=ax, legend=False)
        else:
            ax.hist([real[col], synthetic[col]], bins=20, density=True, label=["real", "synthetic"])
        ax.set_title(col)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(["real", "synthetic"])
    fig.suptitle(title); plt.tight_layout(); plt.show()

compare(tabular_data, syn_data, "After 40 training rounds")
""")

md(r"""
## 6. Switch differential privacy on

Same model, `private=True`. The package now adds noise while it learns and prints the privacy guarantee **epsilon** after every round. Smaller epsilon means stronger privacy. Watch it grow as training continues.
""")
code(r"""
if os.path.exists("fitted_transformer.pkl"):
    os.remove("fitted_transformer.pkl")

private_model = DP_CGAN(epochs=40, batch_size=500, private=True, verbose=False, saved_transformer=None)
private_model.fit(tabular_data)

private_syn_data = private_model.sample(2000)
compare(tabular_data, private_syn_data, "After 40 training rounds, with differential privacy")
""")

md(r"""
## 7. What more training looks like

A generator usually needs a few hundred rounds. We trained the same model for **300 rounds on 5 000 rows** before the workshop (six minutes on a laptop). Load it and compare.
""")
code(rf"""
import urllib.request

urllib.request.urlretrieve("{REPO}/models/dpcgan_plain_300ep.pkl", "dpcgan_plain_300ep.pkl")
trained_model = DP_CGAN.load("dpcgan_plain_300ep.pkl")

compare(tabular_data, trained_model.sample(2000), "After 300 training rounds")
""")

md(r"""
## 8. If you have time left

* Change `epochs=40` to `epochs=100` in step 3 and run steps 3 to 5 again.
* Ask for more rows than you had: `trained_model.sample(50000)`.
* Upload your own CSV with the cell below, **public or already anonymised data only**, and train on it. Keep it to about 10 columns with few categories each, or training gets slow.
""")
code(r"""
from google.colab import files
uploaded = files.upload()
my_data = pd.read_csv(next(iter(uploaded)))
my_data.head()
""")
code(r"""
# Download your synthetic data
files.download("syn_data_file.csv")
""")

md(r"""
## Learn more

* `pip install dp-cgans` · https://github.com/sunchang0124/dp_cgans · https://pypi.org/project/dp-cgans/
* Sun C, van Soest J, Dumontier M. *Generating synthetic personal health data using conditional generative adversarial networks combining with differential privacy.* Journal of Biomedical Informatics 143 (2023) 104404.
* metasyn, the other tool in this workshop: https://metasyn.readthedocs.io
* Questions: chang.sun@maastrichtuniversity.nl
""")

nb.cells = C
out = __file__.rsplit("/scripts/", 1)[0] + "/dpcgans_hands_on.ipynb"
nbf.write(nb, out); print("wrote", out, len(C), "cells")
