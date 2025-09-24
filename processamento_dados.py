import pandas as pd
import re, unicodedata
from datetime import datetime
from collections import Counter
from constantes import COLUMN_RENAME, DOUTOR_TERMS, COLUMNS, ESTADOS_BRASILEIROS


def remove_accents(input_text):
    input_text = str(input_text)
    input_text = (
        unicodedata.normalize("NFKD", input_text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    input_text = input_text.lower().strip()
    input_text = re.sub(r"[^a-z0-9]+", " ", input_text)
    return re.sub(r"\s+", " ", input_text).strip()


def flux_to_pivot(
    counter_dict,
    ordem_rows,
    ordem_cols,
    row_label,
    col_label,
    row_outros_label="br_outros",
    col_outros_label="ex_outros",
):
    if not counter_dict:
        return pd.DataFrame()

    main_rows = set(ordem_rows) - {row_outros_label}
    main_cols = set(ordem_cols) - {col_outros_label}

    agg_counter = Counter()
    for (origem, destino), frequencia in counter_dict.items():
        nova_origem = origem if origem in main_rows else row_outros_label
        novo_destino = destino if destino in main_cols else col_outros_label

        agg_counter[(nova_origem, novo_destino)] += frequencia

    df_flux_agg = pd.DataFrame(
        [
            (origem, destino, frequencia)
            for (origem, destino), frequencia in agg_counter.items()
        ],
        columns=[row_label, col_label, "Peso"],
    )

    pv = df_flux_agg.pivot_table(
        index=row_label,
        columns=col_label,
        values="Peso",
        aggfunc="sum",
        fill_value=0,
    )

    pv = pv.reindex(index=ordem_rows, columns=ordem_cols, fill_value=0)

    return pv

def map_universities(df_final):
    df_origem = df_final[["nome_ies_origem_norm", "uf_ies_origem"]].drop_duplicates()
    frequencia_origem = df_final[
        ["nome_ies_origem_norm", "uf_ies_origem"]
    ].value_counts()
    df_origem = frequencia_origem.reset_index(name="frequencia")
    df_origem_ordenado = df_origem.sort_values(by="frequencia", ascending=False)
    df_origem_ordenado.to_csv("./dados/universidade_origem.csv")

    df_destino = df_final[
        ["nome_ies_destino_norm", "pais_ies_destino"]
    ].drop_duplicates()
    frequencia_destino = df_final[
        ["nome_ies_destino_norm", "pais_ies_destino"]
    ].value_counts()
    df_destino = frequencia_destino.reset_index(name="frequencia")
    df_destino_ordenado = df_destino.sort_values(by="frequencia", ascending=False)
    df_destino_ordenado.to_csv("./dados/universidade_destino.csv")

    inverted_map = {}
    br_counter = 1
    ex_counter = 1

    for _, r in df_origem_ordenado.iterrows():
        university = r["nome_ies_origem_norm"]
        local = r["uf_ies_origem"]
        if university not in inverted_map:
            if local in ESTADOS_BRASILEIROS:
                code = f"br_{br_counter}"
                inverted_map[university] = code
                br_counter += 1
            else:
                code = f"ex_{ex_counter}"
                inverted_map[university] = code
                ex_counter += 1

    for _, r in df_destino_ordenado.iterrows():
        university = r["nome_ies_destino_norm"]
        local = r["pais_ies_destino"]
        if university not in inverted_map:
            if local == "BRASIL":
                code = f"br_{br_counter}"
                inverted_map[university] = code
                br_counter += 1
            else:
                code = f"ex_{ex_counter}"
                inverted_map[university] = code
                ex_counter += 1

    df_mapa_completo = pd.DataFrame(
        list(inverted_map.items()), columns=["nome_universidade", "codigo"]
    )
    df_mapa_completo[["prefixo", "numero"]] = df_mapa_completo["codigo"].str.split("_", expand=True)
    df_mapa_completo["numero"] = df_mapa_completo["numero"].astype(int)
    df_mapa_completo = df_mapa_completo.sort_values(by=["prefixo", "numero"])
    df_mapa_completo = df_mapa_completo.drop(columns=["prefixo", "numero"])
    df_mapa_completo.to_csv("./dados/mapa_universidades.csv")

    return df_mapa_completo


def data_processing():
    PATHS = [
        "./dados/br-capes-bolsas-programas-mobilidade-internacional-2010a2012-2021-03-01.csv",
        "./dados/br-capes-bolsas-programas-mobilidade-internacional-2013a2016-2021-03-01.csv",
        "./dados/br-capes-bolsas-programas-mobilidade-internacional-2017a2019-2021-03-01.csv",
    ]

    # Carrega os dados
    dfs = []
    for path in PATHS:
        df = pd.read_csv(
            path,
            encoding="latin-1",
            sep=";",
        )
        # Remove os processos duplicados
        dfs.append(df)

    # Cria o regex
    escaped_terms = [re.escape(term) for term in DOUTOR_TERMS]
    regex_pattern = "|".join(escaped_terms)

    # Filtra os dados
    dfs_filtered = []
    for df in dfs:
        df_filtered = df.loc[
            df["NM_NIVEL"].str.contains(regex_pattern, na=False, case=False),
            [*COLUMNS, "ID_PROCESSO"],
        ].copy()
        dfs_filtered.append(df_filtered)

    df_final = pd.concat(dfs_filtered)
    df_final = df_final.drop_duplicates(subset=["ID_PROCESSO"])
    df_final = df_final.drop(columns=["ID_PROCESSO"])

    rename_columns = dict(zip(COLUMNS, COLUMN_RENAME))
    df_final = df_final.rename(columns=rename_columns)

    # Remove os dados sem o nome da universidade de origem ou de destino e sem
    # data de inicio ou de fim
    df_final = df_final.dropna(
        subset=[
            "nome_ies_origem",
            "nome_ies_destino",
            "ano_inicio_bolsa",
            "ano_fim_bolsa",
            "pais_ies_destino",
            "uf_ies_origem",
        ]
    )

    df_final = (
        df_final.sort_values(["nome_beneficiario", "ano_inicio_bolsa", "ano_fim_bolsa"])
        .reset_index(drop=True)
        .copy()
    )

    # Cria um coluna com o nome da universidades normalizado
    df_final["nome_ies_origem_norm"] = (
        df_final["nome_ies_origem"].str.lower().apply(remove_accents)
    )
    df_final["nome_ies_destino_norm"] = (
        df_final["nome_ies_destino"].str.lower().apply(remove_accents)
    )

    # Cria o arquivo csv com o df final
    university_map = map_universities(df_final)

    df_mapeado = pd.merge(
        df_final,
        university_map,
        left_on="nome_ies_origem_norm",
        right_on="nome_universidade",
        how="left",
    )
    df_mapeado = df_mapeado.rename(columns={"codigo": "codigo_origem"})
    df_mapeado = df_mapeado.drop(columns=["nome_universidade"])

    df_mapeado = pd.merge(
        df_mapeado,
        university_map,
        left_on="nome_ies_destino_norm",
        right_on="nome_universidade",
        how="left",
    )
    df_mapeado = df_mapeado.rename(columns={"codigo": "codigo_destino"})
    df_mapeado = df_mapeado.drop(columns=["nome_universidade"])

    df_mapeado.to_csv("dados/br-capes-filtrado.csv")

    result = []
    for name, grp in df_mapeado.groupby("nome_beneficiario", sort=False):
        start = grp["ano_inicio_bolsa"].min()
        end = grp["ano_fim_bolsa"].max()

        univ = []
        for _, r in grp.iterrows():
            univ.append(r["codigo_origem"])
            univ.append(r["codigo_destino"])

        result.append(
            {
                "nome_beneficiario": name,
                "AnoInicio": start,
                "AnoFim": end,
                "universidade_lista": univ,
            }
        )

    df_tabela_sequencias = pd.DataFrame(result)
    df_tabela_sequencias.to_csv("dados/df_tabela_sequencias.csv")

    flux_br_to_ex = Counter()

    for _, row in df_tabela_sequencias.iterrows():
        traj = row["universidade_lista"]
        for i in range(1, len(traj)):
            ori, dst = traj[i - 1], traj[i]
            if ori == "0" or dst == "0" or ori == dst:
                continue
            if isinstance(ori, str) and isinstance(dst, str):
                if ori.startswith("br_") and dst.startswith("ex_"):
                    flux_br_to_ex[(ori, dst)] += 1

    # Ordem fixa para BRs e EXs
    ordem_br = [f"br_{i}" for i in range(1, 100)] + ["br_outros"]
    ordem_ex = [f"ex_{i}" for i in range(1, 100)] + ["ex_outros"]

    # Aplica com ordem fixa
    mat_br_to_ex = flux_to_pivot(
        flux_br_to_ex,
        ordem_rows=ordem_br,
        ordem_cols=ordem_ex,
        row_label="BR_Origem",
        col_label="EX_Destino",
        row_outros_label="br_outros",
        col_outros_label="ex_outros",
    )

    mat_br_to_ex.to_csv("./dados/mat_br_to_ex.csv")


if __name__ == "__main__":
    data_processing()
