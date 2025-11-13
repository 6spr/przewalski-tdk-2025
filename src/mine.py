import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import copy

# Saját modulok
from utils import rokonsagkeres, formatter   # <-- ezekre hagyom a hivatkozást, ahogy nálad is van

# ==============================
# 1️⃣ Útvonalak
# ==============================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# ==============================
# 2️⃣ Adatok betöltése
# ==============================
print("📂 Adatok betöltése...")

rokonsag = pd.read_excel(os.path.join(DATA_DIR, "kinship_together.xlsx"))
haremo = pd.ExcelFile(os.path.join(DATA_DIR, "two_harem_approaches.xlsx"))
egyeni_adatok = pd.read_excel(os.path.join(DATA_DIR, "geneo3_nodes.xlsx"))

kin_id = pd.read_csv(os.path.join(RESULTS_DIR, "Kin_id_for_matrix.csv"))
matrix = pd.read_csv(os.path.join(RESULTS_DIR, "Kin_matrix.csv"))

print("📄 Sheetek:", haremo.sheet_names)
vez = haremo.parse("haremsondates_4py_stallion")
tag = haremo.parse("haremsondates_4py_members")
lovak = haremo.parse("horsesondates_group")

# Tisztítás
lovak = lovak[pd.notna(lovak.iloc[:, 1])]
lovak_pure = copy.deepcopy(lovak)

# ==============================
# 3️⃣ odate előállítása
# ==============================
odate = tag.columns.tolist()[2:]  # dátumok oszlopnevei
print(f"📅 Összes dátum ({len(odate)}):", odate[:5], "...")


# ==============================
# 4️⃣ Rokonság és háremadatok számítása
# ==============================
tkin = []   # hímhez való rokonság átlaga
toszam = [] # háremtag-számok

print("⚙️ Rokonság- és háremadatok számítása...")

for i in range(tag.shape[0]):
    ki = []
    kisz = []
    for j in range(2, tag.shape[1]):
        o = []
        sz = 0
        if tag.iloc[i, j] != "[]":
            idk = tag.iloc[i, j].split(',')
            for t in range(len(idk)):
                horse_data = lovak_pure[lovak_pure['Name'] == idk[t]]
                if not horse_data.empty:
                    gender = horse_data['Gender'].values[0]
                    birth = pd.to_datetime(horse_data['Birth'].values[0], errors='coerce')

                    father_name = egyeni_adatok.loc[
                        egyeni_adatok['Name'] == horse_data['Name'].values[0], 'Father_name'
                    ]
                    mother_name = egyeni_adatok.loc[
                        egyeni_adatok['Name'] == horse_data['Name'].values[0], 'Mother_name'
                    ]
                    father_name = father_name.values[0] if not father_name.empty else None
                    mother_name = mother_name.values[0] if not mother_name.empty else None

                    szulok = [father_name, mother_name]
                    if (gender == 2) and (
                        ((odate[j - 2] - birth) / pd.Timedelta(days=365) > 3)
                        or (szulok[0] not in idk and szulok[1] not in idk)
                    ):
                        o.append(rokonsagkeres(idk[t], vez.iloc[i][j], rokonsag))
                        sz += 1
            ki.append(np.mean(o))
            kisz.append(sz)
        else:
            ki.append(None)
            kisz.append(None)
    tkin.append(ki)
    toszam.append(kisz)


# ==============================
# 5️⃣ MINE táblázat előállítása
# ==============================
mine = [['hárem', 'Név', 'kor', 'gyerekek száma',
         'vezetés hossza', 'átlagos háremnagyság vezetés alatt',
         'átlagos rokonság a nőstényekkel vezetés alatt']]

birth_dict = {row['Name']: row['Birth'] for _, row in lovak_pure.iterrows()}
offspring_dict = {row['Name']: row[21] for _, row in egyeni_adatok.iterrows()}

print("📈 MINE adatstruktúra építése...")

for i in range(len(vez)):
    v = ''
    sz = []
    k = []
    d = None
    last_valid_date = None

    for j in range(2, len(vez.iloc[0])):
        akt = vez.iloc[i, j]
        if akt != '[]':
            last_valid_date = odate[j - 2]
            if akt != v:
                if v != '' and sz and k and d is not None:
                    mine[-1].append((last_valid_date - d).days)
                    mine[-1].append(np.mean(sz))
                    mine[-1].append(np.mean(k))

                v = akt
                sz = []
                k = []
                d = odate[j - 2]
                if v in birth_dict:
                    kor = (odate[j - 2] - pd.to_datetime(birth_dict[v], errors='coerce')).days
                    gyerek = offspring_dict.get(v, np.nan)
                    mine.append([i + 1, v, kor, gyerek])
            if v in birth_dict:
                sz.append(toszam[i][j - 2])
                k.append(tkin[i][j - 2])

    if v != '' and sz and k and d is not None and last_valid_date is not None:
        mine[-1].append((last_valid_date - d).days)
        mine[-1].append(np.mean(sz))
        mine[-1].append(np.mean(k))


# ==============================
# 6️⃣ Mentés CSV-be
# ==============================
mine = np.array(mine)
mine_neveknelkul = np.delete(mine, [0, 1], axis=1)
valtnev = np.array(['kor', 'gyerek', 'vez_hossz', 'átl. háremn', 'átl. kin'])

mine_csv = pd.DataFrame(mine_neveknelkul[1:], columns=valtnev)
csv_path = os.path.join(RESULTS_DIR, "Mine_adatok.csv")
mine_csv.to_csv(csv_path, index=False)

print(f"💾 Mentve ide: {csv_path}")


# ==============================
# 7️⃣ MINE Java futtatás
# ==============================
jar_file_path = os.path.join(BASE_DIR, "../src/MINEv2.jar")
command = [
    'java', '-jar', jar_file_path,
    csv_path, '-allPairs', 'cv=0.1', '-equitability', 'id=fewBoxes'
]

print("🚀 MINE futtatása Java-val...")
try:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print("✅ MINE output:", result.stdout)
except subprocess.CalledProcessError as e:
    print(f"❌ Hiba történt: {e.stderr}")

# ================================================
# 📈 Ábrák készítése és mentése (hónap nélkül)
# ================================================
print("📈 Ábrák készítése...")

mine_df = pd.DataFrame(mine[1:], columns=mine[0])

# --- Numerikus oszlopok konvertálása ---
num_cols = [
    'kor',
    'gyerekek száma',
    'vezetés hossza',
    'átlagos háremnagyság vezetés alatt',
    'átlagos rokonság a nőstényekkel vezetés alatt'
]

for col in num_cols:
    mine_df[col] = pd.to_numeric(mine_df[col], errors='coerce')

# --- Tengelyfeliratok a táblázatjelölések szerint ---
axis_labels = {
    'kor': r'$kor$ (nap)',
    'gyerekek száma': r'$N_{\text{utód}}$',
    'vezetés hossza': r'$t_{\text{vezetés}}$ (nap)',
    'átlagos háremnagyság vezetés alatt': r'$\bar{N}_{\text{hárem}}$',
    'átlagos rokonság a nőstényekkel vezetés alatt': r'$\bar{r}_{\text{F--M}}$'
}

# --- Párok generálása (ismétlés nélkül) ---
import itertools
pairs = list(itertools.combinations(num_cols, 2))

# --- Ábrák generálása ---
plot_counter = 0
for (x_col, y_col) in pairs:
    if x_col not in mine_df.columns or y_col not in mine_df.columns:
        continue

    df = mine_df[[x_col, y_col]].dropna()
    if df.empty:
        continue

    x = df[x_col].values
    y = df[y_col].values
    colors = ['cornflowerblue'] * len(x)  # egységes szín minden pontnak

    # --- Illesztések ---
    coeffs_lin = np.polyfit(x, y, deg=1)
    poly_lin = np.poly1d(coeffs_lin)
    y_pred_lin = poly_lin(x)
    r2_lin = 1 - np.sum((y - y_pred_lin)**2) / np.sum((y - np.mean(y))**2)

    coeffs_quad = np.polyfit(x, y, deg=2)
    poly_quad = np.poly1d(coeffs_quad)
    y_pred_quad = poly_quad(x)
    r2_quad = 1 - np.sum((y - y_pred_quad)**2) / np.sum((y - np.mean(y))**2)

    # --- Simított görbék ---
    x_smooth = np.linspace(np.min(x), np.max(x), 400)
    y_smooth_lin = poly_lin(x_smooth)
    y_smooth_quad = poly_quad(x_smooth)

    # --- Ábra ---
    r2_lin_str = f"{r2_lin:.2f}".replace('.', ',')
    r2_quad_str = f"{r2_quad:.2f}".replace('.', ',')

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, c=colors, s=40, edgecolors='k', linewidths=0.3, alpha=0.8)
    plt.plot(x_smooth, y_smooth_lin, color='darkorange', lw=2,
             label=rf'1. fokú illesztés ($R^2={r2_lin_str}$)')
    plt.plot(x_smooth, y_smooth_quad, color='darkred', lw=2.5, linestyle='--',
             label=rf'2. fokú illesztés ($R^2={r2_quad_str}$)')

    plt.xlabel(axis_labels.get(x_col, x_col), fontsize=12)
    plt.ylabel(axis_labels.get(y_col, y_col), fontsize=12)
    plt.title(f"{axis_labels.get(x_col, x_col)} és {axis_labels.get(y_col, y_col)} kapcsolata", fontsize=13)
    plt.legend()
    plt.grid()
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()

    # --- Csak a 6. és 7. ábra mentése ---
    plot_counter += 1
    if plot_counter in [6, 7]:
        save_path = os.path.join(RESULTS_DIR, f"mine_plot_{plot_counter}.png")
        plt.savefig(save_path, dpi=300)
        print(f"💾 Mentve: {save_path}")

    plt.close()

print("✅ Ábrák generálása kész.")
