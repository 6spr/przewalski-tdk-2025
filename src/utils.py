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
