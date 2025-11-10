# src/main.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, gaussian_kde
import itertools

# Hozzáadjuk a src mappát az importútvonalhoz (hogy a többi fájl elérhető legyen)
sys.path.append(os.path.dirname(__file__))

from utils import rokonsagkeres, rokonsagkeres_uj, formatter
from pedigree_builder import build_pedigree_and_kinship

# --------------------------------------------------------
# 1. ADATOK BETÖLTÉSE
# --------------------------------------------------------
print("🔹 Adatok betöltése...")
data_path = os.path.join(os.path.dirname(__file__), "../data/geneo3_nodes.xlsx")
egyeni_adatok = pd.read_excel(data_path)
# --------------------------------------------------------
# 2. PEDIGRÉ ÉS ROKONSÁGI MÁTRIX KÉSZÍTÉSE
# --------------------------------------------------------
print("🔹 Pedigré és rokonsági mátrix készítése PyAGH segítségével...")

A, coef_kinship, kin_matrix, kin_id = build_pedigree_and_kinship(egyeni_adatok)

# Eredmények mentése
results_dir = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(results_dir, exist_ok=True)
kin_matrix.to_csv(os.path.join(results_dir, "Kin_matrix.csv"), index=False)
kin_id.to_csv(os.path.join(results_dir, "Kin_id_for_matrix.csv"), index=False)

# --------------------------------------------------------
# 3. RÉGI ÉS ÚJ ROKONSÁGOK ÖSSZEHASONLÍTÁSA
# --------------------------------------------------------
print("🔹 Rokonsági értékek összehasonlítása...")

# Rokonsági mátrixok betöltése
rokonsag_path = os.path.join(os.path.dirname(__file__), "../data/kinship_together.xlsx")
rokonsag = pd.read_excel(rokonsag_path)

matrix = kin_matrix
name_df = kin_id

# Nevek és kombinációk előkészítése
nevek = list(rokonsag.columns[2:])
kul = list(itertools.combinations(nevek, 2))

# Régi és új rokonsági értékek kiszámítása
regi_ell = [rokonsagkeres(a, b, rokonsag) for a, b in kul]
uj_ell = [rokonsagkeres_uj(a, b, name_df, matrix) for a, b in kul]

# Tisztítás és szűrés
uj_ell = np.array(uj_ell)
regi_ell = np.array(regi_ell)
uj_ell_sz = uj_ell[uj_ell != 0]
regi_ell_sz = regi_ell[uj_ell != 0] * 2

# --------------------------------------------------------
# 4. REGRESSZIÓ ÉS KORRELÁCIÓ SZÁMÍTÁS
# --------------------------------------------------------
print("🔹 Korreláció és regresszió számítása...")

egyutt = np.vstack([uj_ell_sz, regi_ell_sz])
z = gaussian_kde(egyutt)(egyutt)
idx = z.argsort()
uj_ell_r, regi_ell_r, z = uj_ell_sz[idx], regi_ell_sz[idx], z[idx]

of_regiuj = np.polyfit(regi_ell_r, uj_ell_r, 1)
r, p_value = pearsonr(regi_ell_r, uj_ell_r)
print(f"Pearson-korreláció: r = {r:.3f}, p = {p_value:.4f}")
print(f"Lineáris regresszió: y = {of_regiuj[0]:.3f}x + {of_regiuj[1]:.3f}")

# --------------------------------------------------------
# 5. ÁBRA KÉSZÍTÉS ÉS MENTÉS
# --------------------------------------------------------
print("🔹 Ábra generálása...")

base_cmap = plt.get_cmap("plasma")
gamma = 0.3
new_cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_cmap", base_cmap(np.linspace(0, 1, 256) ** gamma)
)

plt.figure(figsize=(6, 6))
plt.grid(zorder=0)
plt.scatter(regi_ell_r, uj_ell_r, c=z, s=5, cmap=new_cmap, zorder=2)
plt.plot([0, 1.5], [0, 1.5], color='red', linestyle='--')
x = np.linspace(0, 1.5, 100)
plt.plot(x, of_regiuj[0]*x + of_regiuj[1], color='green', label=f'{of_regiuj[0]:.3f}x + {of_regiuj[1]:.3f}')
plt.xlabel('Régi rokonság értékek')
plt.ylabel('Új rokonság értékek')
plt.colorbar(label='Sűrűség')
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.legend()
plt.tight_layout()
plt.savefig("../results/regression_plot.png", dpi=300)
plt.show()

print("Lefutott a fő szkript.")

