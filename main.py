import re, unicodedata
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)
from rich import print

CAMINHO = "./atuacoesDoutores.csv"

# Parâmetros principais
janela = 3  # tamanho da janela (anos anteriores)
FILTER_ZERO_TARGET = True  # ignora target '0'
PRED_MODO = "br2ex"  # 'all' | 'br2ex' | 'ex2br'
MIN_NONZERO_IN_HISTORY = 1  # exige >= N itens != '0' na janela
COLLAPSE_RARE = False  # agrupa classes raras em "outros_*"
MIN_CLASS_FREQ = 10  # frequência mínima para manter classe sem colapsar

# Novos parâmetros úteis
DROP_EX_OUTROS = True  # remove amostras cujo alvo seja 'ex_outros'
FILTER_HISTORY_BY_MODE = (
    True  # exige ao menos 1 BR no histórico quando 'br2ex' (ou 1 EX quando 'ex2br')
)


MAPA_UNIVERSIDADES_BR = {
    "br1": ["universidade de são paulo", "usp"],
    "br2": ["universidade estadual de campinas", "unicamp"],
    "br3": ["universidade estadual paulista", "unesp"],
    "br4": ["universidade federal do rio de janeiro", "ufrj"],
    "br5": ["universidade federal de minas gerais", "ufmg"],
    "br6": ["universidade federal do rio grande do sul", "ufrgs"],
    "br7": ["universidade federal de santa catarina", "ufsc"],
    "br8": ["universidade federal do paraná", "ufpr"],
    "br9": ["universidade federal da bahia", "ufba"],
    "br10": ["universidade federal de pernambuco", "ufpe"],
    "br11": ["universidade federal do ceará", "ufc"],
    "br12": ["universidade federal de são carlos", "ufscar"],
    "br13": [
        "puc sao paulo",
        "pucsp",
        "puc-sp",
        "puc sp",
        "pontificia universidade catolica de sao paulo",
    ],
    "br14": [
        "puc rio",
        "pucrio",
        "puc-rio",
        "pontificia universidade catolica do rio de janeiro",
    ],
    "br15": ["fundacao getulio vargas", "fgv"],
    "br16": ["instituto tecnologico de aeronautica", "ita"],
    "br17": ["universidade federal do abc", "ufabc"],
    "br18": ["universidade de brasilia", "unb"],
    "br19": ["universidade federal fluminense", "uff"],
    "br20": ["universidade federal de santa maria", "ufsm"],
    "br21": ["universidade do estado do rio de janeiro", "uerj"],
    "br22": [
        "universidade federal de vicosa",
        "universidade federal de viçosa",
        "ufv",
    ],
    "br23": [
        "universidade federal da paraiba",
        "universidade federal da paraíba",
        "ufpb",
    ],
    "br24": [
        "universidade federal do para",
        "universidade federal do pará",
        "ufpa",
    ],
    "br25": ["universidade federal do rio grande do norte", "ufrn"],
    "br26": [
        "universidade federal de goias",
        "universidade federal de goiás",
        "ufg",
    ],
    "br27": [
        "universidade federal de sao paulo",
        "universidade federal de são paulo",
        "unifesp",
    ],
    "br28": ["universidade federal de pelotas", "ufpel"],
    "br29": [
        "universidade federal do espirito santo",
        "universidade federal do espírito santo",
        "ufes",
    ],
    "br30": ["universidade federal de lavras", "ufla"],
    "br31": ["universidade estadual de londrina", "uel"],
    "br32": [
        "universidade federal de uberlandia",
        "universidade federal de uberlândia",
        "ufu",
    ],
    "br33": [
        "universidade estadual de maringa",
        "universidade estadual de maringá",
        "uem",
    ],
    "br34": ["universidade federal de juiz de fora", "ufjf"],
    "br35": [
        "pontificia universidade catolica de minas gerais",
        "puc minas",
        "puc-minas",
        "pucmg",
        "puc mg",
    ],
    "br36": [
        "pontificia universidade catolica do rio grande do sul",
        "puc rs",
        "pucrs",
        "puc-rs",
    ],
    "br37": ["universidade federal rural de pernambuco", "ufrpe"],
    "br38": ["universidade do estado de santa catarina", "udesc"],
    "br39": ["universidade tecnologica federal do parana", "utfpr"],
    "br40": ["universidade nove de julho", "uninove"],
    "br41": ["universidade do vale do rio dos sinos", "unisinos"],
    "br42": ["universidade presbiteriana mackenzie", "mackenzie"],
    "br43": ["universidade do estado da bahia", "uneb"],
    "br44": ["universidade de pernambuco", "upe"],
    "br45": ["universidade paulista", "unip"],
    "br46": ["universidade do estado de minas gerais", "uemg"],
    "br47": ["universidade de taubate", "unitau"],
    "br48": ["universidade do estado do amazonas", "uea"],
    "br49": ["universidade estacio de sa", "estacio"],
    "br50": ["universidade do estado de mato grosso", "unemat"],
    "br51": ["universidade metodista de piracicaba", "unimep"],
    "br52": ["universidade de caxias do sul", "ucs"],
    "br53": ["universidade catolica de brasilia", "ucb"],
    "br54": ["universidade luterana do brasil", "ulbra"],
    "br55": ["universidade do grande rio", "unigranrio"],
    "br56": ["universidade de fortaleza", "unifor"],
    "br57": ["universidade anhembi morumbi", "anhembi morumbi"],
    "br58": ["universidade de franca", "unifran"],
    "br59": ["universidade do estado do para", "uepa"],
    "br60": [
        "universidade regional integrada do alto uruguai e das missoes",
        "uri",
    ],
    "br61": ["universidade aberta do brasil", "uab"],
    "br62": ["universidade sao francisco", "usf"],
    "br63": ["escola superior de agricultura luiz de queiroz", "esalq"],
}

MAPA_UNIVERSIDADES_EX = {
    "ex_us_mit": ["massachusetts institute of technology", "mit"],
    "ex_us_harvard": ["harvard university", "harvard"],
    "ex_us_stanford": ["stanford university", "stanford"],
    "ex_us_berkeley": [
        "university of california berkeley",
        "uc berkeley",
        "berkeley",
    ],
    "ex_us_caltech": ["california institute of technology", "caltech"],
    "ex_uk_oxford": ["university of oxford", "oxford"],
    "ex_uk_cambridge": ["university of cambridge", "cambridge"],
    "ex_uk_imperial": ["imperial college london", "imperial college", "imperial"],
    "ex_uk_ucl": ["university college london", "ucl"],
    "ex_ch_eth": [
        "eth zurich",
        "eth zurich swiss federal institute of technology",
        "eth",
    ],
    "ex_ch_epfl": ["epfl", "ecole polytechnique federale de lausanne"],
    "ex_de_tum": ["technical university of munich", "tum"],
    "ex_ca_toronto": ["university of toronto", "uoft"],
    "ex_ca_mcgill": ["mcgill university", "mcgill"],
    "ex_jp_tokyo": ["university of tokyo", "utokyo", "u tokyo"],
    "ex_cn_tsinghua": ["tsinghua university", "tsinghua"],
    "ex_cn_peking": ["peking university", "beida", "peking"],
    "ex_fr_sorbonne": [
        "sorbonne university",
        "sorbonne universite",
        "universite pierre et marie curie",
    ],
    "ex_pt_coimbra": ["universidade de coimbra"],
    "ex_pt_lisboa": ["universidade de lisboa"],
    "ex_pt_porto": ["universidade do porto"],
    "ex_pt_minho": ["universidade do minho"],
    "ex_pt_aveiro": ["universidade de aveiro"],
    "ex_pt_nova": ["universidade nova de lisboa"],
    "ex_ar_uba": ["universidad de buenos aires", "uba"],
    "ex_es_ucm": ["universidad complutense de madrid", "ucm"],
    "ex_us_florida": ["university of florida", "ufl"],
    "ex_mx_unam": ["universidad nacional autonoma de mexico", "unam"],
    "ex_es_uam": ["universidad autonoma de madrid", "uam"],
    "ex_uy_udelar": ["universidad de la republica uruguay", "udelar"],
    "ex_es_salamanca": ["universidad de salamanca"],
    "ex_us_michigan": ["university of michigan", "umich"],
    "ex_ca_ubc": ["university of british columbia", "ubc"],
    "ex_us_columbia": ["columbia university", "columbia"],
    "ex_us_ucdavis": ["university of california davis", "uc davis", "davis"],
    "ex_co_unal": ["universidad nacional de colombia bogota", "unal"],
    "ex_us_cornell": ["cornell university", "cornell"],
    "ex_ar_unlp": ["universidad nacional de la plata", "unlp"],
    "ex_nl_wageningen": ["wageningen university", "wur"],
    "ex_us_texas": ["university of texas at austin", "ut austin"],
    "ex_us_wisconsin": ["university of wisconsin madison", "uw madison"],
    "ex_es_sevilla": ["universidad de sevilla"],
    "ex_co_antioquia": ["universidad de antioquia", "udea"],
    "ex_us_nyu": ["new york university", "nyu"],
    "ex_es_usc": ["universidad de santiago de compostela", "usc"],
    "ex_us_yale": ["yale university", "yale"],
    "ex_ca_alberta": ["university of alberta", "ualberta"],
    "ex_us_ucsd": ["university of california san diego", "ucsd"],
    "ex_es_granada": ["universidad de granada", "granada"],
    "ex_us_ohiost": ["ohio state university", "osu"],
    "ex_us_jhu": ["johns hopkins university", "jhu"],
    "ex_ca_ottawa": ["university of ottawa", "uottawa"],
    "ex_dk_copenhagen": ["university of copenhagen", "ku"],
    "ex_ar_unc": ["universidad nacional de cordoba argentina", "unc"],
    "ex_us_penn": ["university of pennsylvania", "upenn"],
    "ex_au_sydney": ["the university of sydney", "usyd"],
    "ex_pe_sanmarcos": ["universidad nacional mayor de san marcos", "san marcos"],
    "ex_dk_aarhus": ["aarhus university", "aarhus"],
    "ex_uk_manchester": ["university of manchester", "manchester"],
    "ex_us_msu": ["michigan state university", "msu"],
    "ex_us_ucsystem": ["university of california system", "uc system"],
    "ex_us_ucla": ["university of california los angeles", "ucla"],
    "ex_be_ghent": ["ghent university", "ugent"],
    "ex_us_duke": ["duke university", "duke"],
    "ex_au_uq": ["the university of queensland", "uq"],
    "ex_nl_leiden": ["leiden university", "leiden"],
    "ex_us_brown": ["brown university", "brown"],
    "ex_us_uiuc": ["university of illinois at urbana champaign", "uiuc"],
    "ex_nl_delft": ["delft university of technology", "tu delft"],
    "ex_uk_nottingham": ["university of nottingham", "nottingham"],
    "ex_uk_leeds": ["university of leeds", "leeds"],
    "ex_uk_bristol": ["university of bristol", "bristol"],
    "ex_co_valle": ["universidad del valle", "univalle"],
    "ex_uk_london": ["university of london"],
    "ex_es_zaragoza": ["universidad de zaragoza", "zaragoza"],
    "ex_us_chicago": ["university of chicago", "uchicago"],
    "ex_us_purdue": ["purdue university", "purdue"],
    "ex_us_arizona": ["university of arizona", "uarizona"],
    "ex_us_ncsu": ["north carolina state university", "ncsu"],
    "ex_uk_birmingham": ["university of birmingham", "birmingham"],
    "ex_se_lund": ["lund university", "lund"],
    "ex_ar_unr": ["universidad nacional de rosario", "unr"],
    "ex_us_princeton": ["princeton university", "princeton"],
    "ex_cl_uchile": ["universidad de chile", "uchile"],
    "ex_ca_waterloo": ["university of waterloo", "uwaterloo"],
    "ex_se_uppsala": ["uppsala university", "uppsala"],
    "ex_us_georgia": ["university of georgia", "uga"],
    "ex_es_upm": ["universidad politecnica de madrid", "upm"],
    "ex_es_cadiz": ["universidad de cadiz", "cadiz"],
    "ex_us_pitt": ["university of pittsburgh", "pitt"],
    "ex_uk_edinburgh": ["university of edinburgh", "edinburgh"],
    "ex_co_unal_all": ["universidad nacional de colombia"],
}


def eh_hospital_universitario(nome_up: str) -> bool:
    return ("HOSPITAL" in nome_up and "UNIVERS" in nome_up) or (
        "HOSPITAL UNIVERSIT" in nome_up
    )


def eh_laboratorio_universitario(nome_up: str) -> bool:
    return "LABORAT" in nome_up and (
        "UNIVERS" in nome_up or "FACULDADE" in nome_up or "ESCOLA" in nome_up
    )


def normalize_text(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classificar_instituicao_universidade(nome_up: str) -> str:
    UNIVERSIDADE_KEYWORDS = [
        r"\bUNIVERSIDADE\b",
        r"\bUNIVERSITY\b",
        r"\bUNIVERSIT[ÀAÁÃÄ]T?\b",
        r"\bUNIVERSIDAD\b",
        r"\bINSTITUTO FEDERAL\b",
        r"\bCENTRO UNIVERSIT[ÁA]RIO\b",
        r"\bCEFET\b",
        r"\bESCOLA SUPERIOR\b",
        r"\bESCOLA POLIT[ÉE]CNICA\b",
        r"\bFACULDADE\b",
        r"\bFACULTAD\b",
        r"\bFACULTY\b",
        r"\bSCHOOL OF\b",
        # Siglas brasileiras comuns
        r"\bUFRJ\b",
        r"\bUFF\b",
        r"\bUFRGS\b",
        r"\bUFSC\b",
        r"\bUFMG\b",
        r"\bUFBA\b",
        r"\bUNESP\b",
        r"\bUNICAMP\b",
        r"\bUSP\b",
        r"\bPUC\b",
        r"\bPUC-\w+\b",
        r"\bUF\w{2,}\b",
        r"\bIF\w{2,}\b",
    ]
    universidade_patterns = re.compile("|".join(UNIVERSIDADE_KEYWORDS), re.IGNORECASE)

    # exceções que puxam para Universidade
    if eh_hospital_universitario(nome_up) or eh_laboratorio_universitario(nome_up):
        return "Universidade"
    # somente universidade neste passo
    if re.search(universidade_patterns, nome_up):
        return "Universidade"
    # (Governo/Indústria ficam para depois, para acelerar a iteração)
    return "Outro"


def codigo_br_ou_none_from_norm(n_norm: str):
    mapa_universidades_br_norm = {
        code: [normalize_text(v) for v in vals]
        for code, vals in MAPA_UNIVERSIDADES_BR.items()
    }

    _memo_map_br = {}
    if n_norm in _memo_map_br:
        return _memo_map_br[n_norm]
    for code, norms in mapa_universidades_br_norm.items():
        if any(v and v in n_norm for v in norms):
            _memo_map_br[n_norm] = code
            return code
    _memo_map_br[n_norm] = None
    return None


def codigo_ex_ou_none_from_norm(n_norm: str):
    mapa_universidades_ex_norm = {
        code: [normalize_text(v) for v in vals]
        for code, vals in MAPA_UNIVERSIDADES_EX.items()
    }

    _memo_map_ex = {}
    if n_norm in _memo_map_ex:
        return _memo_map_ex[n_norm]
    for code, norms in mapa_universidades_ex_norm.items():
        if any(v and v in n_norm for v in norms):
            _memo_map_ex[n_norm] = code
            return code
    _memo_map_ex[n_norm] = None
    return None


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
    return None  # ainda não mapeado (resolver no fallback)


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

    # Se listas de ordem forem passadas, usa elas; senão cai no sorted (comportamento antigo)
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


def aceita_target_por_modo(y_code: str) -> bool:
    if not isinstance(y_code, str):
        return False
    if PRED_MODO == "all":
        return True
    if PRED_MODO == "br2ex":
        return y_code.startswith("ex_")
    if PRED_MODO == "ex2br":
        return y_code.startswith("br")
    return True


def historico_valido_por_modo(hist):
    if not FILTER_HISTORY_BY_MODE:
        return True
    if PRED_MODO == "br2ex":
        return any(isinstance(h, str) and h.startswith("br") for h in hist)
    if PRED_MODO == "ex2br":
        return any(isinstance(h, str) and h.startswith("ex_") for h in hist)
    return True


def main():
    # === Importa a base ===

    df = pd.read_csv(CAMINHO)

    DOUTOR_TERMS = [
        "doutorado",
        "doutorando",
        "doutoranda",
        "doutoramento",
        "doc",
        "phd",
        "doutor",
        "doutora",
    ]

    # === AGRUPA INSTITUIÇÕES ===

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
    df = df[mask_doutor].copy()

    # Normaliza colunas de instituição (uma vez só)
    df["Instituicao"] = df["Instituicao"].astype(str)
    df["Instituicao_up"] = df["Instituicao"].str.upper()  # para regex; preserva acentos
    df["Instituicao_norm"] = df["Instituicao"].map(
        normalize_text
    )  # para matching de dicionários

    df["GrupoInstituicao"] = df["Instituicao_up"].map(
        classificar_instituicao_universidade
    )
    df = df[df["GrupoInstituicao"] == "Universidade"].copy()

    df_cleaned = df.copy()

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
    df_cleaned = df_cleaned[df_cleaned["AnoFim"] >= df_cleaned["AnoInicio"]].copy()
    df_cleaned = df_cleaned.sort_values(["IDLattes", "AnoInicio", "AnoFim"]).copy()

    df_cleaned["CodigoUniversidade"] = None
    mask_uni = df_cleaned["GrupoInstituicao"] == "Universidade"
    df_cleaned.loc[mask_uni, "CodigoUniversidade"] = df_cleaned.loc[mask_uni].apply(
        atribuir_codigo_universidade_explicito_norm, axis=1
    )

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
            ]  # 'brX', 'ex_*' (inclui 'ex_outros'/'br_outros') ou None

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
                "universidade_lista": univ,  # 'brX', 'ex_*' ou '0'
            }
        )

    df_tabela_sequencias = pd.DataFrame(resultados)
    print(f"\nTrilhas geradas para {len(df_tabela_sequencias)} pesquisadores.")
    print(df_tabela_sequencias.head(10))

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

    # Ordem fixa para BRs e EXs
    ordem_br = [f"br{i}" for i in range(1, 63)] + ["br_outros"]

    # Aplica com ordem fixa
    mat_br_to_ex = flux_to_pivot(
        flux_br_to_ex,
        "br",
        "ex_",
        "BR_Origem",
        "EX_Destino",
        ordem_rows=ordem_br,
    )

    # print("\nMatriz BR → EX (contagem de transições):")
    print(mat_br_to_ex)

    mat_br_to_ex.to_csv("mat_br_to_ex.csv")

    # === MODELA BASE ===

    X_raw, y_raw = [], []

    # Gera amostras a partir das trilhas (df_tabela_sequencias vem da etapa anterior)
    for _, linha in df_tabela_sequencias.iterrows():
        traj = linha.get("universidade_lista", None)  # lista com 'brX', 'ex_*' ou '0'
        if not isinstance(traj, list) or len(traj) < janela + 1:
            continue

        # janela deslizante
        for t in range(janela, len(traj)):
            historico = traj[t - janela : t]
            proximo = traj[t]

            # 1) histórico com informação suficiente?
            n_nonzero = sum(1 for h in historico if h != "0")
            if n_nonzero < MIN_NONZERO_IN_HISTORY:
                continue

            # 2) (opcional) restringe tipo do histórico por modo
            if not historico_valido_por_modo(historico):
                continue

            # 3) filtra target '0' (opcional)
            if FILTER_ZERO_TARGET and proximo == "0":
                continue

            # 4) filtra por modo (ex.: br2ex mantém só EX como alvo)
            if not aceita_target_por_modo(proximo):
                continue

            X_raw.append(historico)
            y_raw.append(proximo)

    # DataFrames iniciais
    X = pd.DataFrame(X_raw, columns=[f"ano_{i+1}" for i in range(janela)])
    y = pd.Series(y_raw, name="proximo_codigo")

    # --- filtro crítico: remover 'ex_outros' do alvo (quando desejado) ---
    if DROP_EX_OUTROS and len(y) > 0:
        mask_keep = y != "ex_outros"
        X = X.loc[mask_keep].reset_index(drop=True)
        y = y.loc[mask_keep].reset_index(drop=True)

    # (Opcional) Colapsa classes raras do alvo
    if COLLAPSE_RARE and len(y) > 0:
        freq = Counter(y)

        def colapsa(label: str) -> str:
            if freq[label] >= MIN_CLASS_FREQ:
                return label
            if isinstance(label, str) and label.startswith("br"):
                return "outros_br"
            if isinstance(label, str) and label.startswith("ex_"):
                return "outros_ex"
            return "outros"

        y = y.map(colapsa)

    # print(f"{len(X)} amostras geradas.")
    # print(f"Classes únicas no target: {y.nunique()}")
    # print("\nDistribuição do target (top 15):")
    print(y.value_counts().head(15))

    # === Codificar valores para o modelo ===

    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna("0").astype(str),  # Garante que todos os dados sejam strings
        y.astype(str),  # Garante que o alvo também seja string
        test_size=0.2,
        random_state=42,
        stratify=(
            y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        ),  # Estratifica se houver mais de uma classe
    )

    # 2) One-Hot Encoder apenas em X (treina no conjunto de treino, transforma em ambos)
    # Compatibilidade: scikit-learn >=1.2 usa 'sparse_output'; versões anteriores usam 'sparse'
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    X_train_enc = ohe.fit_transform(X_train)
    X_test_enc = ohe.transform(X_test)

    # 3) LabelEncoder em y (treina com todos os valores de y para evitar erros de "classe desconhecida")
    le_y = LabelEncoder()
    le_y.fit(y.astype(str))

    y_train_enc = le_y.transform(y_train)
    y_test_enc = le_y.transform(y_test)

    # (Opcional) Variáveis úteis para análise posterior
    feature_names = ohe.get_feature_names_out(
        X.columns
    )  # Nomes das novas colunas criadas
    classes_y = le_y.classes_  # Rótulos originais do alvo

    # Imprime informações sobre os dados processados
    # print(f"Shape de X_train_enc: {X_train_enc.shape}, Shape de X_test_enc: {X_test_enc.shape}")
    # print(f"Número de classes no alvo (y): {len(classes_y)}")

    # Exemplo de como reverter as previsões para os rótulos originais:
    # y_pred_labels = le_y.inverse_transform(y_pred)

    # === Treinar o Random Forest ===

    # Treinar
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,  # pode limitar para evitar overfitting
        random_state=42,
        n_jobs=-1,
        class_weight=None,  # mude para 'balanced' se houver muito desbalanceamento
    )
    clf.fit(X_train_enc, y_train_enc)

    # Prever
    y_pred_enc = clf.predict(X_test_enc)

    # Relatório de classificação
    labels_todos = np.arange(len(le_y.classes_))
    nomes_classes = le_y.inverse_transform(labels_todos)

    print("Relatório de Classificação:")
    print(
        classification_report(
            y_test_enc,
            y_pred_enc,
            labels=labels_todos,
            target_names=nomes_classes,
            zero_division=0,
        )
    )

    # Matriz de confusão
    cm = confusion_matrix(y_test_enc, y_pred_enc, labels=labels_todos)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in nomes_classes],
        columns=[f"pred_{c}" for c in nomes_classes],
    )
    print(cm_df)

    # Top-k accuracy
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test_enc)
        top3 = top_k_accuracy_score(y_test_enc, y_proba, k=3, labels=labels_todos)
        print(f"Top-3 accuracy: {top3:.3f}")

    # Features mais importantes
    try:
        importances = clf.feature_importances_
        if "feature_names" in locals():
            feat_imp = pd.Series(importances, index=feature_names).sort_values(
                ascending=False
            )[:15]
            print("\nTop 15 features mais importantes:")
            print(feat_imp)

            # Gráfico
            feat_imp.plot(kind="barh", figsize=(8, 6))
            plt.title("Top 15 Features mais importantes")
            plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
