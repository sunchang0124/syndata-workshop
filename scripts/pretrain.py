"""Pre-train the fallback generators shipped with the workshop (run on the Mac, CPU).
Usage: python pretrain.py plain 300 | python pretrain.py private 100
Writes models/dpcgan_<mode>_<epochs>ep.pkl and data/synthetic_<mode>_<epochs>ep.csv
"""
import sys, time, pandas as pd, warnings; warnings.filterwarnings("ignore")
from dp_cgans import DP_CGAN
mode, epochs = sys.argv[1], int(sys.argv[2])
root = __file__.rsplit("/scripts/", 1)[0]
real = pd.read_csv(f"{root}/data/census_workshop.csv")
train = real.sample(5000, random_state=42).reset_index(drop=True)
train.to_csv(f"{root}/data/census_train_5000.csv", index=False)
model = DP_CGAN(epochs=epochs, batch_size=500, verbose=True, cuda=False, private=(mode == "private"), saved_transformer=None)
t = time.time(); model.fit(train); print(f"{mode} {epochs} epochs: {time.time()-t:.0f}s", flush=True)
model.save(f"{root}/models/dpcgan_{mode}_{epochs}ep.pkl")
syn = model.sample(5000); syn.to_csv(f"{root}/data/synthetic_{mode}_{epochs}ep.csv", index=False)
print(syn.describe(include="all").T, flush=True)
