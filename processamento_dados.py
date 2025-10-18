import pandas as pd
import re, unicodedata
from datetime import datetime
from constantes import (
    DOUTOR_TERMS,
    CAMINHO,
    MAPA_UNIVERSIDADES_BR,
    MAPA_UNIVERSIDADES_EX,
    UNIVERSIDADE_KEYWORDS,
)


def normalize_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


MAPA_UNIVERSIDADES_BR_NORM = {
    code: [normalize_text(v) for v in vals]
    for code, vals in MAPA_UNIVERSIDADES_BR.items()
}
MAPA_UNIVERSIDADES_EX_NORM = {
    code: [normalize_text(v) for v in vals]
    for code, vals in MAPA_UNIVERSIDADES_EX.items()
}

_MEMO_MAP_BR = {}
_MEMO_MAP_EX = {}


def eh_hospital_universitario(nome_up: str) -> bool:
    return ("HOSPITAL" in nome_up and "UNIVERS" in nome_up) or (
        "HOSPITAL UNIVERSIT" in nome_up
    )


def eh_laboratorio_universitario(nome_up: str) -> bool:
    return "LABORAT" in nome_up and (
        "UNIVERS" in nome_up or "FACULDADE" in nome_up or "ESCOLA" in nome_up
    )


def classificar_instituicao_universidade(nome_up: str) -> str:
    universidade_patterns = re.compile("|".join(UNIVERSIDADE_KEYWORDS), re.IGNORECASE)

    if eh_hospital_universitario(nome_up) or eh_laboratorio_universitario(nome_up):
        return "Universidade"
    if re.search(universidade_patterns, nome_up):
        return "Universidade"
    return "Outro"


def atribuir_codigo_universidade_explicito_norm(row):
    if row["GrupoInstituicao"] != "Universidade":
        return None
    n_norm = row["Instituicao_norm"]
    cbr = codigo_br_ou_none_from_norm(n_norm)
    if cbr:
        return cbr
    cex = codigo_ex_ou_none_from_norm(n_norm)
    if cex:
        return cex
    return None


def codigo_br_ou_none_from_norm(n_norm: str):
    if n_norm in _MEMO_MAP_BR:
        return _MEMO_MAP_BR[n_norm]
    for code, norms in MAPA_UNIVERSIDADES_BR_NORM.items():
        if any(v and v in n_norm for v in norms):
            _MEMO_MAP_BR[n_norm] = code
            return code
    _MEMO_MAP_BR[n_norm] = None
    return None


def codigo_ex_ou_none_from_norm(n_norm: str):
    if n_norm in _MEMO_MAP_EX:
        return _MEMO_MAP_EX[n_norm]
    for code, norms in MAPA_UNIVERSIDADES_EX_NORM.items():
        if any(v and v in n_norm for v in norms):
            _MEMO_MAP_EX[n_norm] = code
            return code
    _MEMO_MAP_EX[n_norm] = None
    return None


def data_processing():
    df = pd.read_csv(CAMINHO)

    parts = []
    for t in DOUTOR_TERMS:
        t_esc = re.escape(t)
        if " " in t or "-" in t:
            parts.append(t_esc)
        else:
            parts.append(rf"\b{t_esc}\b")
    pat_any = re.compile("|".join(parts))

    df = df.copy()
    df["OutroVinculo_norm"] = df["OutroVinculo"].astype(str).map(normalize_text)

    mask_doutor = df["OutroVinculo_norm"].str.contains(pat_any, regex=True, na=False)
    df = pd.DataFrame(df[mask_doutor].copy())

    df["Instituicao"] = df["Instituicao"].astype(str)
    df["Instituicao_up"] = df["Instituicao"].str.upper()
    df["Instituicao_norm"] = df["Instituicao"].map(
        normalize_text
    )

    df["GrupoInstituicao"] = df["Instituicao_up"].map(
        classificar_instituicao_universidade
    )
    df = df[df["GrupoInstituicao"] == "Universidade"].copy()

    df_cleaned = pd.DataFrame(df.copy())

    df_cleaned = df_cleaned.dropna(subset=["AnoInicio"]).copy()
    ano_atual = datetime.now().year
    df_cleaned["AnoFim"] = df_cleaned["AnoFim"].fillna(ano_atual)

    df_cleaned["AnoInicio"] = pd.to_numeric(
        df_cleaned["AnoInicio"], errors="coerce"
    ).astype("Int64")
    df_cleaned["AnoFim"] = pd.to_numeric(df_cleaned["AnoFim"], errors="coerce").astype(
        "Int64"
    )
    df_cleaned = df_cleaned.dropna(subset=["AnoInicio", "AnoFim"])
    df_cleaned = pd.DataFrame(
        df_cleaned[df_cleaned["AnoFim"] >= df_cleaned["AnoInicio"]].copy()
    )
    df_cleaned = df_cleaned.sort_values(["IDLattes", "AnoInicio", "AnoFim"]).copy()

    df_cleaned["CodigoUniversidade"] = None
    mask_uni = df_cleaned["GrupoInstituicao"] == "Universidade"
    df_cleaned.loc[mask_uni, "CodigoUniversidade"] = df_cleaned.loc[mask_uni].apply(
        atribuir_codigo_universidade_explicito_norm, axis=1
    )
    df_cleaned.to_csv("./dados/dados_limpos.csv")


if __name__ == "__main__":
    data_processing()
