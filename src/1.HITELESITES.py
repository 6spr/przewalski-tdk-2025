"""
main.py — Rokonsági mátrixok összehasonlítása
TDK 2025 — Przewalski-projekt
"""

import os
import sys
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, gaussian_kde

# --- Saját modulok importja ---
sys.path.append(os.path.dirname(__file__))
from utils import rokonsagkeres, rokonsagkeres_uj, formatter, build_pedigree


# --------------------------------------------------------
# 1. ADATOK BETÖLTÉSE
# --------------------------------------------------------
print("🔹 Adatok betöltése...")

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

geneo_path = os.path.join(DATA_DIR, "geneo3_nodes.xlsx")
kinship_path = os.path.join(DATA_DIR, "kinship_together.xlsx")

egyeni_adatok = pd.read_excel(geneo_path)
rokonsag = pd.read_excel(kinship_path)


# --------------------------------------------------------
# 2. PEDIGRÉ ÉS ROKONSÁGI MÁTRIX KÉSZÍTÉSE
# --------------------------------------------------------
print("🔹 Pedigré és rokonsági mátrix készítése...")

kin_matrix, kin_id, kin_list = build_pedigree(egyeni_adatok)

print(f"📊 kin_matrix: {kin_matrix.shape[0]}x{kin_matrix.shape[1]}")


# --------------------------------------------------------
# 3. RÉGI ÉS ÚJ ROKONSÁGOK ÖSSZEHASONLÍTÁSA
# --------------------------------------------------------
print("🔹 Rokonsági értékek számítása...")

# --- Oszlopnevek tisztítása ---
rokonsag.columns = [str(c).strip().upper() for c in rokonsag.columns]
kin_id.columns = [str(c).strip().lower() for c in kin_id.columns]

if "id" not in kin_id.columns:
    kin_id.columns = ["id"]
kin_id["id"] = kin_id["id"].astype(str).str.strip().str.upper()

# --- Névlista és párok ---
nevek = list(rokonsag.columns[2:])
kulcsok = list(itertools.combinations(nevek, 2))
print(f"📈 Összehasonlítandó párok száma: {len(kulcsok):,}")

# --- Rokonsági értékek kinyerése ---
regi_ell = [rokonsagkeres(a, b, rokonsag) for a, b in kulcsok]
uj_ell = [rokonsagkeres_uj(a, b, kin_id, kin_matrix) for a, b in kulcsok]

# --- Nullák kiszűrése ---
regi_ell = np.array(regi_ell)
uj_ell = np.array(uj_ell)

maszk = uj_ell != 0
uj_ell_sz = uj_ell[maszk]
regi_ell_sz = regi_ell[maszk] * 2

print(f"✅ Aktív (nem nulla) párok: {len(uj_ell_sz):,}")


# --------------------------------------------------------
# 4. KORRELÁCIÓ ÉS REGRESSZIÓ
# --------------------------------------------------------
if len(uj_ell_sz) < 2:
    print("⚠️ Túl kevés adatpont a korrelációhoz – ábra kihagyva.")
    of_regiuj = [0, 0]
    uj_ell_r, regi_ell_r, z = np.array([]), np.array([]), np.array([])
else:
    egyutt = np.vstack([uj_ell_sz, regi_ell_sz])
    z = gaussian_kde(egyutt)(egyutt)
    idx = z.argsort()
    uj_ell_r, regi_ell_r, z = uj_ell_sz[idx], regi_ell_sz[idx], z[idx]

    of_regiuj = np.polyfit(regi_ell_r, uj_ell_r, 1)
    r, p_value = pearsonr(regi_ell_r, uj_ell_r)

    print(f"📈 Pearson-korreláció: r = {r:.3f} (p = {p_value:.4f})")
    print(f"📉 Lineáris regresszió: y = {of_regiuj[0]:.3f}x + {of_regiuj[1]:.3f}")


# --------------------------------------------------------
# 5. ÁBRA KÉSZÍTÉS ÉS MENTÉS
# --------------------------------------------------------
if len(uj_ell_sz) >= 2:
    print("🔹 Ábra mentése...")

    plt.figure(figsize=(6, 6))
    plt.grid(zorder=0)

    base_cmap = plt.get_cmap("plasma")
    gamma = 0.3
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", base_cmap(np.linspace(0, 1, 256) ** gamma)
    )

    plt.scatter(regi_ell_r, uj_ell_r, c=z, s=5, cmap=new_cmap, zorder=2)
    plt.plot([0, 1.5], [0, 1.5], 'r--', label="y = x")

    x = np.linspace(0, 1.5, 100)
    plt.plot(x, of_regiuj[0]*x + of_regiuj[1], 'g',
             label=f'{of_regiuj[0]:.3f}x + {of_regiuj[1]:.3f}')
    plt.colorbar(label='Sűrűség')

    plt.xlabel('Régi rokonság')
    plt.ylabel('Új rokonság')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "regression_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

print("✅ Lefutott a fő szkript.")

