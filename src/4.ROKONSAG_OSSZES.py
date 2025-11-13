import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import itertools
import copy
from datetime import datetime

# Saját modul
from utils import rokonsagkeres, formatter

# ==============================
# 1️⃣ Útvonalak
# ==============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# ==============================
# 2️⃣ Fájlok betöltése
# ==============================
print("📂 Adatok betöltése...")
rokonsag = pd.read_excel(os.path.join(DATA_DIR, "kinship_together.xlsx"))
haremo = pd.ExcelFile(os.path.join(DATA_DIR, "two_harem_approaches.xlsx"))
kin_id = pd.read_csv(os.path.join(RESULTS_DIR, "Kin_id_for_matrix.csv"))
matrix = pd.read_csv(os.path.join(RESULTS_DIR, "Kin_matrix.csv"))

print("📄 Sheetek:", haremo.sheet_names)
vez = haremo.parse("haremsondates_4py_stallion")
tag = haremo.parse("haremsondates_4py_members")

# ==============================
# 3️⃣ Háremváltások keresése
# ==============================
print("🔎 Háremváltások keresése...")

valt1 = []
for i in range(vez.shape[0]):
    for j in range(2, vez.shape[1] - 1):
        if (
            vez.iloc[i, j] != "[]"
            and vez.iloc[i, j + 1] != "[]"
            and vez.iloc[i, j] != vez.iloc[i, j + 1]
        ):
            valt1.append(
                [vez.iloc[i, j], vez.iloc[i, j + 1], vez.iloc[i, 0], vez.columns[j + 1]]
            )

valt = np.array(valt1, dtype=object)
valt[:, 3] = pd.to_datetime(valt[:, 3], errors="coerce")

print(f"📈 Összesen {len(valt)} háremváltás azonosítva.")

# ==============================
# 4️⃣ Rokonsági értékek
# ==============================
print("✅ Rokonsági értékek lekérve.")

kin = []
date = []

for i in range(2, vez.shape[1]):
    pill = [vez.iloc[j, i] for j in range(vez.shape[0]) if vez.iloc[j, i] != "[]"]
    if len(pill) > 1:
        date.append(pd.to_datetime(vez.columns[i], errors="coerce"))
        kin.append([
            rokonsagkeres(p[0], p[1], rokonsag)
            for p in itertools.combinations(pill, 2)
        ])

csopkin = copy.deepcopy(kin)
flat_kin = copy.deepcopy([x for sub in kin for x in sub])
flat_kin = np.asarray(flat_kin, dtype=float)
ddate = [d for i, sub in enumerate(kin) for d in [date[i]] * len(sub)]

# ==============================
# 5️⃣ Átlag, maszkok, regresszió
# ==============================
atlag = np.array([np.mean([x for x in sub if x != 0]) for sub in csopkin])
dbszam = np.array([int(np.sqrt(len(csopkin[i]) * 2)) + 1 for i in range(len(csopkin))])
mask2 = np.array([d >= pd.to_datetime("2009-01-01") for d in date])

date_num = np.array([mdates.date2num(x) for x in date])
rdate_num = date_num[mask2]
ratlag = atlag[mask2]

a, b = np.polyfit(rdate_num, ratlag, 1)
print("📊 Lineáris illesztés meredeksége:", a)

# ==============================
# 6️⃣ Sűrűség (heatmap) + ábra
# ==============================
plt.figure(figsize=(10, 6))

ddate_num = mdates.date2num(ddate)
xy = np.vstack([ddate_num, flat_kin])
z = gaussian_kde(xy)(xy)
z = np.asarray(z)
idx = np.argsort(z)
x, y, z = ddate_num[idx], flat_kin[idx], z[idx]

hb = plt.hexbin(ddate_num, flat_kin, gridsize=50, cmap="Blues")
counts = hb.get_array()
norm = mcolors.Normalize(vmin=counts.min(), vmax=counts.max() / 2)
hb.set_norm(norm)

plt.colorbar(hb, label="Sűrűség")

plt.plot(np.array(date)[mask2], atlag[mask2], "-", ms=4, color="r", lw=2,
         label="Vezető hímek átlagos rokonsága egymással")

# Trendvonal (2009 utáni illesztés)
plt.plot(np.array(date)[mask2], a * np.array(rdate_num) + b,
         "--", color="orange", lw=2, label="Lineáris trend (2009 után)")

plt.xlabel("Dátum")
plt.ylabel("Rokonsági index")
plt.xlim(pd.to_datetime("2009-01-01"), max(date))
plt.legend()
plt.gca().yaxis.set_major_formatter(formatter)
plt.title("Csődörök rokonsági indexe és trendje 2009 után")
plt.grid(True)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "rokonsag_vez.png")
plt.savefig(out_path, dpi=300)
plt.close()

