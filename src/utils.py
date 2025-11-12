import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import PyAGH

def build_pedigree(egyeni_adatok, output_dir="results/"):
    """
    Családfa és rokonsági mátrix előállítása PyAGH segítségével.
    """
    csaladfa = [['ID', 'Father', 'Mother']]

    # Adatok beolvasása és családfa-építés
    for i in range(len(egyeni_adatok['Name'])):
        if (
            not pd.isna(egyeni_adatok['Name'][i]) and
            not pd.isna(egyeni_adatok['Father_name'][i]) and
            not pd.isna(egyeni_adatok['Mother_name'][i]) and
            egyeni_adatok['Father_name'][i] != '-'
        ):
            csaladfa.append([
                egyeni_adatok['Name'][i],
                egyeni_adatok['Father_name'][i],
                egyeni_adatok['Mother_name'][i]
            ])

    csalad_df = pd.DataFrame(csaladfa[1:], columns=csaladfa[0])
    csalad_df.to_csv(f"{output_dir}/Loszulok.csv", index=False)

    # Pedigré rendezése
    ped_sorted = PyAGH.sortPed(csalad_df)

    # Rokonsági mátrix számítása
    A = PyAGH.makeA(ped_sorted)
    coef_kinship = PyAGH.coefKinship(A)

    # Mentések
    kin_matrix = pd.DataFrame(A[0])
    kin_matrix.to_csv(f"{output_dir}/Kin_matrix.csv", index=False)

    kin_id = pd.DataFrame(A[1])
    kin_id.to_csv(f"{output_dir}/Kin_id_for_matrix.csv", index=False)

    kin_list = pd.DataFrame(coef_kinship)
    kin_list.to_csv(f"{output_dir}/Kin_lista.csv", index=False)

    return kin_matrix, kin_id, kin_list



def rokonsagkeres(him1, him2, rokonsag):
    """Kikeresi két egyed rokonsági értékét a régi mátrixból."""
    him1 = him1.split('_')[1] if '_' in him1 else him1
    him2 = him2.split('_')[1] if '_' in him2 else him2
    him1 = him1.upper()
    him2 = him2.upper()
    if him1 in rokonsag.iloc[:, 0].values and him2 in rokonsag.columns:
        return rokonsag.loc[rokonsag[rokonsag.iloc[:, 0] == him1].index[0], him2]
    else:
        return 0

def rokonsagkeres_uj(name1, name2, name_df, matrix):
    """Kikeresi két egyed rokonsági értékét az új (PyAGH-ból származó) mátrixból."""
    name1 = name1.split('_')[1] if '_' in name1 else name1
    name2 = name2.split('_')[1] if '_' in name2 else name2
    name1 = name1.upper()
    name2 = name2.upper()

    # Apró hibajavítás (konzisztencia miatt)
    if name1 == 'ORCHIDEA':
        name1 = 'ORHIDEA'
    if name2 == 'ORCHIDEA':
        name2 = 'ORHIDEA'

    if name1 not in name_df['id'].values or name2 not in name_df['id'].values:
        print(f"One or both names not found in the DataFrame: {name1}, {name2}")
        return 0
    
    row = name_df[name_df['id'] == name1].index[0]
    col = name_df[name_df['id'] == name2].index[0]

    return matrix.loc[row][col]

def decimal_comma(x, pos):
    """Tizedespont helyett vessző formázás a tengelyen."""
    return f"{x:.2f}".replace('.', ',').rstrip('0').rstrip(',')

formatter = FuncFormatter(decimal_comma)

def custom_colormap(base="plasma", gamma=0.3):
    """Egy testreszabott színtér a vizualizációhoz."""
    base_cmap = plt.get_cmap(base)
    return mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap",
        base_cmap(np.linspace(0, 1, 256) ** gamma)
    )

def alsoadatok(kap2, bin, szazalek):
    """Kiszedi az adatok legalsó szegmensét binenként."""
    kap2 = np.array(kap2)
    bin = int(bin)
    also = np.array([])
    egys = int(len(kap2) / bin)

    for i in range(bin):
        start = egys * i
        end = egys * (i + 1) if i < bin - 1 else len(kap2)
        negys = kap2[start:end]

        if len(negys) < 2:
            continue

        min_val, max_val = np.min(negys), np.max(negys)
        threshold = min_val + szazalek / 100 * (max_val - min_val)
        low_vals = negys[negys < threshold]
        also = np.append(also, low_vals)

    indices = np.in1d(kap2, also)
    return also, indices


def rolling_mean(x, y, s):
    """Futóátlag számítása időalapú ablakokkal."""
    x = np.array(x)
    y = np.array(y)
    x_num = mdates.date2num(x)
    window = (x_num.max() - x_num.min()) / s
    roll_mean = []

    for i in range(len(x_num)):
        mask = np.abs(x_num - x_num[i]) <= window / 2
        roll_mean.append(np.mean(y[mask]))

    return x, np.array(roll_mean)
