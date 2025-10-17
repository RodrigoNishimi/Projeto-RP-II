import pandas as pd
from collections import Counter

def flux_to_pivot(
    counter_dict,
    row_prefix,
    col_prefix,
    row_label,
    col_label,
    ordem_rows=None,
    ordem_cols=None,
):
    if not counter_dict:
        return pd.DataFrame()
    df_flux = pd.DataFrame(
        [(o, d, w) for (o, d), w in counter_dict.items()],
        columns=[row_label, col_label, "Peso"],
    )

    if ordem_rows is None:
        ordem_rows = sorted(
            {o for (o, _) in counter_dict.keys() if o.startswith(row_prefix)}
        )
    if ordem_cols is None:
        ordem_cols = sorted(
            {d for (_, d) in counter_dict.keys() if d.startswith(col_prefix)}
        )

    pv = df_flux.pivot_table(
        index=row_label,
        columns=col_label,
        values="Peso",
        aggfunc="sum",
        fill_value=0,
    )
    pv = pv.reindex(index=ordem_rows, columns=ordem_cols, fill_value=0)
    return pv

def cria_mat_br_to_ex(df_tabela_sequencias):
    flux_br_to_ex = Counter()

    for _, row in df_tabela_sequencias.iterrows():
        traj = row["universidade_lista"]
        for i in range(1, len(traj)):
            ori, dst = traj[i - 1], traj[i]
            if ori == "0" or dst == "0" or ori == dst:
                continue
            if isinstance(ori, str) and isinstance(dst, str):
                if ori.startswith("br") and dst.startswith("ex_"):
                    flux_br_to_ex[(ori, dst)] += 1

    ordem_br = [f"br{i}" for i in range(1, 63)] + ["br_outros"]

    mat_br_to_ex = flux_to_pivot(
        flux_br_to_ex,
        "br",
        "ex_",
        "BR_Origem",
        "EX_Destino",
        ordem_rows=ordem_br,
    )

    mat_br_to_ex.to_csv("./dados/mat_br_to_ex.csv")

    return mat_br_to_ex


def cria_fluxos():
    df_cleaned = pd.read_csv("./dados/dados_limpos.csv")

    resultados = []

    for id_lattes, grp in df_cleaned.groupby("IDLattes", sort=False):
        if grp.empty:
            continue
        min_ano = int(grp["AnoInicio"].min())
        max_ano = int(grp["AnoFim"].max())
        n_anos = max_ano - min_ano + 1

        univ = ["0"] * n_anos

        for _, r in grp.iterrows():
            ai, af = int(r["AnoInicio"]), int(r["AnoFim"])
            grupo = r["GrupoInstituicao"]
            cod = r[
                "CodigoUniversidade"
            ]

            for ano in range(ai, af + 1):
                idx = ano - min_ano
                if not (0 <= idx < n_anos):
                    continue

                if grupo == "Universidade":
                    atual = univ[idx]
                    if isinstance(cod, str) and cod.startswith("br"):
                        if atual in ("0",) or not atual.startswith("br"):
                            univ[idx] = cod
                    elif isinstance(cod, str) and cod.startswith("ex_"):
                        if atual == "0":
                            univ[idx] = cod

        resultados.append(
            {
                "IDLattes": id_lattes,
                "AnoInicio": min_ano,
                "AnoFim": max_ano,
                "universidade_lista": univ,
            }
        )

    df_tabela_sequencias = pd.DataFrame(resultados)
    df_tabela_sequencias.to_csv("./dados/df_tabela_sequencias.csv")

    cria_mat_br_to_ex(df_tabela_sequencias)

if __name__ == "__main__":
    cria_fluxos()
