# DP-CGANS hands-on · *Unlock Your Sensitive Data* workshop

Materials for the DP-CGANS part of the TDCC-SSH workshop **Unlock Your Sensitive Data: Generate Your Own Synthetic Dataset**
(Raoul Schram, Utrecht University, and Chang Sun, Maastricht University), 14 October 2026.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sunchang0124/syndata-workshop/blob/main/dpcgans_hands_on.ipynb)

## What is in here

| path | what |
|---|---|
| `dpcgans_hands_on.ipynb` | the 15-minute hands-on notebook (install, inspect, train, evaluate, train with differential privacy) |
| `data/census_workshop.csv` | the workshop table: 46 012 rows, 7 columns, built from the public UCI census income data (OpenML `adult` v2) |
| `data/census_train_5000.csv` | the 5 000 rows the pre-trained generators were trained on |
| `data/synthetic_*.csv` | 5 000 synthetic rows sampled from each pre-trained generator |
| `models/dpcgan_plain_300ep.pkl` | DP-CGAN trained 300 epochs without differential privacy |
| `models/dpcgan_private_100ep.pkl` | DP-CGAN trained 100 epochs with differential privacy |
| `scripts/prepare_data.py` | rebuilds `data/census_workshop.csv` |
| `scripts/pretrain.py` | retrains the fallback generators |
| `scripts/build_notebook.py` | generates the notebook; edit this, not the `.ipynb` |

## Notes for the presenter

* Training 40 epochs on 2 000 rows takes about 20 s on a laptop and about 1 to 2 minutes on the free Colab CPU. No GPU needed.
* Training time in DP-CGANS grows quickly with the number of categories, because the conditional loss works on pairs of categorical columns. The census columns were collapsed to 2 to 4 categories each for that reason. A column with 40 categories makes an epoch take minutes.
* dp-cgans caches `fitted_transformer.pkl` in the working directory and reuses it silently. The notebook deletes it before every training run. Keep that if you adapt the code.
* `private=True` uses a fixed noise multiplier; the reported epsilon grows with the number of epochs.

## Rebuild

```bash
pip install dp-cgans nbformat scikit-learn
python scripts/prepare_data.py
python scripts/pretrain.py plain 300      # ~6 min on a laptop CPU
python scripts/pretrain.py private 100
python scripts/build_notebook.py
```

## References

Sun C, van Soest J, Dumontier M. Generating synthetic personal health data using conditional generative adversarial networks combining with differential privacy. *Journal of Biomedical Informatics* 143 (2023) 104404. https://doi.org/10.1016/j.jbi.2023.104404

Package: https://github.com/sunchang0124/dp_cgans · metasyn: https://metasyn.readthedocs.io
