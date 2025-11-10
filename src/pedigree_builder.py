import pandas as pd
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
