"""
Monte Carlo szimuláció hímcsere-rokonság vizsgálathoz
(c) TDK 2025
"""

import os
import random
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
from utils import formatter, rokonsagkeres, rolling_mean, alsoadatok  # ezek a saját segédfüggvényeid

# ------------------------------------------------------------
# --- Adatok betöltése ---
# ------------------------------------------------------------

print("🔹 Adatok betöltése...")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# A fájlneveket itt állítsd be helyileg:
kin_id = pd.read_csv(os.path.join(RESULTS_DIR, "Kin_id_for_matrix.csv"))
matrix = pd.read_csv(os.path.join(RESULTS_DIR, "Kin_matrix.csv"))

rokonsag = pd.read_excel(os.path.join(DATA_DIR, "kinship_together.xlsx"))
haremo = pd.ExcelFile(os.path.join(DATA_DIR, "two_harem_approaches.xlsx"))
print("📘 A hárém fájl lapjai:", haremo.sheet_names)

vez = haremo.parse('haremsondates_4py_stallion')
tag = haremo.parse('haremsondates_4py_members')
lovak = haremo.parse('horsesondates_group')
lovak = lovak[pd.notna(lovak.iloc[:, 1])]  # NaN törlés
lovak_pure = copy.deepcopy(lovak)

egyeni_adatok = pd.read_excel(os.path.join(DATA_DIR, "geneo3_nodes.xlsx"))

# ------------------------------------------------------------
# --- Változások kigyűjtése ---
# ------------------------------------------------------------

# ------------------------------------------------------------
# --- Változások kigyűjtése ---
# ------------------------------------------------------------
valt1 = []

for i in range(vez.shape[0]):  # a változások kigyűjtése
    for j in range(2, vez.shape[1] - 1):
        if (
            vez.iloc[i, j] != "[]"
            and vez.iloc[i, j + 1] != "[]"
            and vez.iloc[i, j] != vez.iloc[i, j + 1]
        ):
            valt1.append([
                vez.iloc[i, j],
                vez.iloc[i, j + 1],
                vez.iloc[i, 0],
                vez.columns[j + 1]
            ])

valt = copy.deepcopy(np.array(valt1))
valt1 = np.array(valt1)

# Dátum és háremazonosító konverzió
valt[:, 3] = [mdates.date2num(x) for x in valt[:, 3]]
valt[:, 2] = [int(x.split("_")[1]) for x in valt[:, 2]]
valt1[:, 2] = [int(x.split("_")[1]) for x in valt1[:, 2]]

# ------------------------------------------------------------
# --- Rokonságok kiszámítása ---
# ------------------------------------------------------------
kap = np.array([float(rokonsagkeres(v[0], v[1], rokonsag)) for v in valt])
colors = ["red" if value == 0 else "blue" for value in kap]

# Csak nem-null rokonságok a görbéhez
mask = kap != 0
kap2 = kap[mask].astype(float)
datum2 = np.array(valt[:, 3])[mask].astype(float)

# Idő szerint sorba rendezés
sorban = np.argsort(datum2)
datum2 = datum2[sorban]
kap2 = kap2[sorban]

# ------------------------------------------------------------
# --- Eredeti adatok futóátlaga + pontdiagram ---
# ------------------------------------------------------------
also, indices = alsoadatok(kap2, 8, 100)
also_date = datum2[indices]
xxx, yyy = rolling_mean(also_date, also, 8)

fig, ax = plt.subplots(figsize=(10, 6))

# 🔹 Szórt pontok a váltásokhoz
ax.scatter(valt[:, 3], kap, c=colors, marker='o', edgecolor='black', s=50, label='Váltási pontok')

# 🔹 Futóátlag
ax.plot(xxx, yyy, color="blue", linewidth=2, label="Eredeti futóátlag")

# 🔹 Dátumformátum és tengelyek
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(formatter)

ax.set_xlabel("Dátum")
ax.set_ylabel("Rokonsági index")
ax.legend()
ax.grid(True)
plt.tight_layout()

# 🔹 Mentés
out_path = os.path.join(RESULTS_DIR, "valtas_og.png")
plt.savefig(out_path, dpi=300)
plt.close()

# ------------------------------------------------------------
# --- Monte Carlo randomizáció ---
# ------------------------------------------------------------

him = lovak[lovak["Gender"] == 1]  # csak a hímek

def random_valtas(valtas, himek):
    """Randomizált váltás generálása."""
    valtas_rand = copy.deepcopy(valtas)
    foglaltak = []

    for i in range(len(valtas)):
        names = list(
            himek[
                (himek["Birth"] + pd.DateOffset(years=5) <= valtas[i][3])
                & (himek["Death"] >= valtas[i][3])
            ]["Name"].values
        )

        sikerult = False
        while True:
            if not names:
                sikerult = True
                break

            uj_nev = random.choice(names)
            uj_datum1 = valtas[i][3]

            if i < len(valtas) - 1 and valtas_rand[i][2] == valtas_rand[i + 1][2]:
                uj_datum2 = valtas[i + 1][3] + pd.Timedelta(days=30)
            else:
                uj_datum2 = valtas[i][3] + pd.Timedelta(days=300)

            foglalt = False
            for nev, start, end in foglaltak:
                if nev == uj_nev and not (uj_datum2 < start or uj_datum1 > end):
                    foglalt = True
                    break

            if not foglalt:
                valtas_rand[i][1] = uj_nev
                break
            else:
                names.remove(uj_nev)

        if not sikerult:
            continue

        if i < len(valtas) - 1 and valtas_rand[i][2] == valtas_rand[i + 1][2]:
            valtas_rand[i + 1][0] = uj_nev
            foglaltak.append([uj_nev, valtas_rand[i][3],
                              valtas_rand[i + 1][3] + pd.Timedelta(days=30)])
        else:
            foglaltak.append([uj_nev, valtas_rand[i][3],
                              valtas_rand[i][3] + pd.Timedelta(days=300)])

    return valtas_rand


# ------------------------------------------------------------
# --- Monte Carlo szimuláció ---
# ------------------------------------------------------------

n_random = 1000
szazalek = 100
x_rand, y_rand = [], []

for i in range(n_random):
    kap_rand = []
    rand_valtas = random_valtas(valt1, him)

    for j in range(len(rand_valtas)):
        kap_rand.append(rokonsagkeres(rand_valtas[j][0], rand_valtas[j][1], rokonsag))

    kap_rand = [float(x) for x in kap_rand]

    datum = valt1[:, 3]
    also, indices = alsoadatok(kap_rand, 8, szazalek)
    also_date = datum[indices]
    also_vals = np.array(kap_rand)[indices]

    x, y = rolling_mean(also_date, also_vals, 8)
    plt.plot(x, y, color="red", alpha=0.15)
    x_rand.append(x)
    y_rand.append(y)

plt.plot(xxx, yyy, color="blue", linewidth=2, label="Eredeti futóátlag")
plt.grid()
plt.legend()
out_path = os.path.join(RESULTS_DIR, "MC_all.png")
plt.savefig(out_path, dpi=300)
plt.close()

# ------------------------------------------------------------
# --- Átlag + szórás kirajzolása ---
# ------------------------------------------------------------

start_date = min(valt1[:, 3])
end_date = max(valt1[:, 3])
common_x = pd.date_range(start=start_date, end=end_date, periods=200)
common_x_numeric = mdates.date2num(common_x)

all_y_interp = []

for x, y in zip(x_rand, y_rand):
    x_numeric = mdates.date2num(x)
    min_len = min(len(x_numeric), len(y))
    x_numeric = x_numeric[:min_len]
    y = y[:min_len]

    f = interp1d(x_numeric, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    all_y_interp.append(f(common_x_numeric))

all_y_interp = np.array(all_y_interp)
mean_y = np.mean(all_y_interp, axis=0)
std_y = np.std(all_y_interp, axis=0)

plt.plot(common_x_numeric, mean_y, color="red", linewidth=2, label=f"Randomizált váltások átlaga ({n_random} futás)")
plt.fill_between(common_x_numeric, mean_y - std_y, mean_y + std_y, color="red", alpha=0.2)
plt.plot(xxx, yyy, color="blue", linewidth=2, label="Eredeti futóátlag")

plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.ylabel("Rokonság")
plt.xlabel("Dátum")
plt.legend()
plt.grid()
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "MC_sd.png")
plt.savefig(out_path, dpi=300)
plt.close()
