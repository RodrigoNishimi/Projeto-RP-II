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
import statsmodels.api as sm
from collections import Counter, defaultdict
import plotly.graph_objects as go

CAMINHO = "./atuacoesDoutores.csv"
#CAMINHO = r"C:\Users\jvfon\Downloads\atuacoesDoutores.csv"


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

    mat_br_to_ex.to_csv("./dados/mat_br_to_ex.csv")

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
    X = pd.DataFrame(X_raw, columns=[f"ano_{i + 1}" for i in range(janela)])
    X = pd.DataFrame(X_raw, columns=[f"ano_{i + 1}" for i in range(janela)])
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

        

    # ======================
    # 1) Dicionários: nomes e coordenadas
    # ======================

    # rótulos de apresentação (código -> nome curto)
    label_br = {
        'br1':  'USP',
        'br2':  'UNICAMP',
        'br3':  'UNESP',
        'br4':  'UFRJ',
        'br5':  'UFMG',
        'br6':  'UFRGS',
        'br7':  'UFSC',
        'br8':  'UFPR',
        'br9':  'UFBA',
        'br10': 'UFPE',
        'br11': 'UFC',
        'br12': 'UFSCar',
        'br13': 'PUC-SP',
        'br14': 'PUC-Rio',
        'br15': 'FGV',
        'br16': 'ITA',
        'br17': 'UFABC',
        'br18': 'UnB',
        'br19': 'UFF',
        'br20': 'UFSM',
        'br21': 'UERJ',
        'br22': 'UFV',
        'br23': 'UFPB',
        'br24': 'UFPA',
        'br25': 'UFRN',
        'br26': 'UFG',
        'br27': 'UNIFESP',
        'br28': 'UFPel',
        'br29': 'UFES',
        'br30': 'UFLA',
        'br31': 'UEL',
        'br32': 'UFU',
        'br33': 'UEM',
        'br34': 'UFJF',
        'br35': 'PUC-Minas',
        'br36': 'PUC-RS',
        'br37': 'UFRPE',
        'br38': 'UDESC',
        'br39': 'UTFPR',
        'br40': 'UNINOVE',
        'br41': 'Unisinos',
        'br42': 'Mackenzie',
        'br43': 'UNEB',
        'br44': 'UPE',
        'br45': 'UNIP',
        'br46': 'UEMG',
        'br47': 'UNITAU',
        'br48': 'UEA',
        'br49': 'Estácio',
        'br50': 'UNEMAT',
        'br51': 'UNIMEP',
        'br52': 'UCS',
        'br53': 'UCB',
        'br54': 'ULBRA',
        'br55': 'UNIGRANRIO',
        'br56': 'UNIFOR',
        'br57': 'Anhembi Morumbi',
        'br58': 'UNIFRAN',
        'br59': 'UEPA',
        'br60': 'URI',
        'br61': 'UAB',
        'br62': 'USF',
        'br63': 'ESALQ',

    }

    label_ex = {
        'ex_us_mit':       'MIT',
        'ex_us_harvard':   'Harvard',
        'ex_us_stanford':  'Stanford',
        'ex_us_berkeley':  'UC Berkeley',
        'ex_us_caltech':   'Caltech',
        'ex_uk_oxford':    'Oxford',
        'ex_uk_cambridge': 'Cambridge',
        'ex_uk_imperial':  'Imperial',
        'ex_uk_ucl':       'UCL',
        'ex_ch_eth':       'ETH Zürich',
        'ex_ch_epfl':      'EPFL',
        'ex_de_tum':       'TUM',
        'ex_ca_toronto':   'U. Toronto',
        'ex_ca_mcgill':    'McGill',
        'ex_jp_tokyo':     'U. Tokyo',
        'ex_cn_tsinghua':  'Tsinghua',
        'ex_cn_peking':    'Peking',
        'ex_fr_sorbonne':  'Sorbonne',
        'ex_pt_coimbra':  'Coimbra',
        'ex_pt_lisboa':   'Lisboa',
        'ex_pt_porto':    'Porto',
        'ex_pt_minho':    'Minho',
        'ex_pt_aveiro':   'Aveiro',
        'ex_pt_nova':     'NOVA Lisboa',
        'ex_ar_uba':      'UBA',
        'ex_es_ucm':      'UCM',
        'ex_us_florida':  'U. Florida',
        'ex_mx_unam':      'UNAM',
        'ex_es_uam':       'UAM',
        'ex_uy_udelar':    'Udelar',
        'ex_es_salamanca': 'Salamanca',
        'ex_us_michigan':  'Michigan',
        'ex_ca_ubc':       'UBC',
        'ex_us_columbia':  'Columbia',
        'ex_us_ucdavis':   'UC Davis',
        'ex_co_unal':      'UNAL',
        'ex_us_cornell':   'Cornell',
        'ex_ar_unlp':      'UNLP',
        'ex_nl_wageningen':'Wageningen',
        'ex_us_texas':     'UT Austin',
        'ex_us_wisconsin': 'UW Madison',
        'ex_es_sevilla':   'Sevilla',
        'ex_co_antioquia' : 'Antioquia',
        'ex_us_nyu'       : 'NYU',
        'ex_es_usc'       : 'USC',
        'ex_us_yale'      : 'Yale',
        'ex_ca_alberta'   : 'U. Alberta',
        'ex_us_ucsd'      : 'UCSD',
        'ex_es_granada'   : 'Granada',
        'ex_us_ohiost'    : 'Ohio State',
        'ex_us_jhu'       : 'Johns Hopkins',
        'ex_ca_ottawa'    : 'Ottawa',
        'ex_dk_copenhagen': 'Copenhagen',
        'ex_ar_unc'       : 'UNC (Córdoba)',
        'ex_us_penn'      : 'UPenn',
        'ex_au_sydney'    : 'Sydney',
        'ex_pe_sanmarcos' : 'San Marcos',
        'ex_dk_aarhus'    : 'Aarhus',
        'ex_uk_manchester': 'Manchester',
        'ex_us_msu'       : 'MSU',
        'ex_us_ucsystem'  : 'UC System',
        'ex_us_ucla'      : 'UCLA',
        'ex_be_ghent'     : 'Ghent',
        'ex_us_duke'      : 'Duke',
        'ex_au_uq'        : 'UQ',
        'ex_nl_leiden'    : 'Leiden',
        'ex_us_brown'     : 'Brown',
        'ex_us_uiuc'      : 'UIUC',
        'ex_nl_delft'     : 'TU Delft',
        'ex_uk_nottingham': 'Nottingham',
        'ex_uk_leeds'     : 'Leeds',
        'ex_uk_bristol'   : 'Bristol',
        'ex_co_valle'     : 'Univalle',
        'ex_uk_london'    : 'U. London',
        'ex_es_zaragoza'  : 'Zaragoza',
        'ex_us_chicago'   : 'UChicago',
        'ex_us_purdue'    : 'Purdue',
        'ex_us_arizona'   : 'Arizona',
        'ex_us_ncsu'      : 'NCSU',
        'ex_uk_birmingham': 'Birmingham',
        'ex_se_lund'      : 'Lund',
        'ex_ar_unr'       : 'UNR',
        'ex_us_princeton' : 'Princeton',
        'ex_cl_uchile'    : 'U. Chile',
        'ex_ca_waterloo'  : 'Waterloo',
        'ex_se_uppsala'   : 'Uppsala',
        'ex_us_georgia'   : 'UGA',
        'ex_es_upm'       : 'UPM',
        'ex_es_cadiz'     : 'Cádiz',
        'ex_us_pitt'      : 'Pitt',
        'ex_uk_edinburgh' : 'Edinburgh',
        'ex_co_unal_all'  : 'UNAL',
    }

    # Coordenadas aproximadas (lat, lon) — ajuste à vontade
    # BR (campus principal típico)
    coords_br = {
        'br1':  (-23.5617, -46.7308),  # USP, São Paulo (Cidade Universitária)
        'br2':  (-22.8170, -47.0698),  # UNICAMP, Campinas
        'br3':  (-22.3246, -49.0286),  # UNESP, Botucatu (aprox.)
        'br4':  (-22.8610, -43.2330),  # UFRJ, Ilha do Fundão, Rio
        'br5':  (-19.8700, -43.9670),  # UFMG, Belo Horizonte
        'br6':  (-30.0331, -51.2196),  # UFRGS, Porto Alegre (aprox. central)
        'br7':  (-27.6008, -48.5190),  # UFSC, Florianópolis
        'br8':  (-25.4284, -49.2733),  # UFPR, Curitiba (aprox. central)
        'br9':  (-12.9980, -38.5070),  # UFBA, Salvador (Ondina aprox.)
        'br10': ( -8.0557, -34.9510),  # UFPE, Recife
        'br11': ( -3.7450, -38.5740),  # UFC, Fortaleza (Pici aprox.)
        'br12': (-21.9810, -47.8800),  # UFSCar, São Carlos
        'br13': (-23.5390, -46.6720),  # PUC-SP, Perdizes (aprox.)
        'br14': (-22.9790, -43.2320),  # PUC-Rio, Gávea
        'br15': (-22.9100, -43.1730),  # FGV-Rio (aprox. Praia de Botafogo)
        'br16': (-23.2156, -45.8612),  # ITA, São José dos Campos
        'br17': (-23.6460, -46.5320),  # UFABC, Santo André
        'br18': (-15.7646, -47.8705),  # UnB, Brasília
        'br19': (-22.9180, -43.1290),  # UFF, Niterói
        'br20': (-29.7174, -53.7169),  # UFSM, Santa Maria
        'br21': (-22.9090, -43.2290),  # UERJ, Rio
        'br22': (-20.7610, -42.8690),  # UFV, Viçosa
        'br23': ( -7.1357, -34.8450),  # UFPB, João Pessoa
        'br24': ( -1.4748, -48.4520),  # UFPA, Belém
        'br25': ( -5.8390, -35.1990),  # UFRN, Natal
        'br26': (-16.6040, -49.2660),  # UFG, Goiânia
        'br27': (-23.6515, -46.5733),  # UNIFESP, SP (campus São Paulo)
        'br28': (-31.7710, -52.3420),  # UFPel, Pelotas
        'br29': (-20.2770, -40.3030),  # UFES, Vitória
        'br30': (-21.2260, -44.9840),  # UFLA, Lavras
        'br31': (-23.3100, -51.1620),  # UEL, Londrina
        'br32': (-18.9180, -48.2570),  # UFU, Uberlândia
        'br33': (-23.4050, -51.9410),  # UEM, Maringá
        'br34': (-21.7700, -43.3670),  # UFJF, Juiz de Fora
        'br35': (-19.9220, -43.9900),  # PUC Minas, BH
        'br36': (-30.0590, -51.1720),  # PUCRS, Porto Alegre
        'br37': ( -8.0170, -34.9470),  # UFRPE, Recife
        'br38': (-27.6019, -48.5197),  # UDESC, Florianópolis
        'br39': (-25.4515, -49.2310),  # UTFPR, Curitiba
        'br40': (-23.5382, -46.7356),  # UNINOVE, São Paulo
        'br41': (-29.7915, -51.1506),  # Unisinos, São Leopoldo
        'br42': (-23.5498, -46.6523),  # Mackenzie, Higienópolis, SP
        'br43': (-12.9426, -38.4804),  # UNEB, Salvador
        'br44': (-8.0476, -34.8989),   # UPE, Recife
        'br45': (-23.5366, -46.5617),  # UNIP, São Paulo
        'br46': (-19.9201, -43.9346),  # UEMG, BH (sede)
        'br47': (-23.0246, -45.5593),  # UNITAU, Taubaté
        'br48': (-3.0944, -60.0170),   # UEA, Manaus
        'br49': (-22.9129, -43.2302),  # Estácio, Rio
        'br50': (-15.6324, -56.0921),  # UNEMAT, Cáceres
        'br51': (-22.7340, -47.6483),  # UNIMEP, Piracicaba
        'br52': (-29.1669, -51.1790),  # UCS, Caxias do Sul
        'br53': (-15.8631, -47.9137),  # UCB, Brasília
        'br54': (-29.6905, -51.1322),  # ULBRA, Canoas
        'br55': (-22.8053, -43.2065),  # UNIGRANRIO, Duque de Caxias
        'br56': (-3.7710, -38.4811),   # UNIFOR, Fortaleza
        'br57': (-23.5975, -46.6359),  # Anhembi Morumbi, SP
        'br58': (-20.5382, -47.4009),  # UNIFRAN, Franca
        'br59': (-1.4550, -48.5022),   # UEPA, Belém
        'br60': (-27.6417, -54.2639),  # URI, Santo Ângelo
        'br61': (-15.7753, -47.9440),  # UAB, Brasília (coordenação central)
        'br62': (-22.8540, -46.3190),  # USF, Bragança Paulista
        'br63': (-22.7058, -47.6357),  # ESALQ/USP, Piracicaba
    }

    # EX (campus principal)
    coords_ex = {
        'ex_us_mit':       ( 42.3601, -71.0942),   # Cambridge, MA
        'ex_us_harvard':   ( 42.3770, -71.1167),   # Cambridge, MA
        'ex_us_stanford':  ( 37.4275, -122.1697),  # Stanford, CA
        'ex_us_berkeley':  ( 37.8719, -122.2585),  # Berkeley, CA
        'ex_us_caltech':   ( 34.1377, -118.1253),  # Pasadena, CA
        'ex_uk_oxford':    ( 51.7548,  -1.2544),   # Oxford
        'ex_uk_cambridge': ( 52.2043,   0.1149),   # Cambridge
        'ex_uk_imperial':  ( 51.4988,  -0.1749),   # London (South Kensington)
        'ex_uk_ucl':       ( 51.5246,  -0.1340),   # London (Bloomsbury)
        'ex_ch_eth':       ( 47.3763,   8.5476),   # Zürich
        'ex_ch_epfl':      ( 46.5191,   6.5668),   # Lausanne
        'ex_de_tum':       ( 48.2620,  11.6670),   # TUM Garching (aprox.)
        'ex_ca_toronto':   ( 43.6629, -79.3957),   # Toronto
        'ex_ca_mcgill':    ( 45.5048, -73.5772),   # Montreal
        'ex_jp_tokyo':     ( 35.7126, 139.7610),   # Tóquio (Hongo)
        'ex_cn_tsinghua':  ( 40.0030, 116.3260),   # Pequim
        'ex_cn_peking':    ( 39.9869, 116.3055),   # Pequim
        'ex_fr_sorbonne':  ( 48.8470,   2.3440),   # Paris
        'ex_jp_kyoto':     (35.0260, 135.7808),
        'ex_jp_osaka':     (34.8225, 135.5231),
        'ex_jp_tohoku':    (38.2554, 140.8520),
        'ex_cn_shanghai':  (31.1976, 121.4326),
        'ex_cn_fudan':     (31.2990, 121.5037),
        'ex_cn_cas':       (39.9042, 116.4074),
        'ex_pt_coimbra':  (40.2070,  -8.4250),   # Universidade de Coimbra
        'ex_pt_lisboa':   (38.7527,  -9.1567),   # Universidade de Lisboa (Cidade Universitária)
        'ex_pt_porto':    (41.1780,  -8.5970),   # Universidade do Porto
        'ex_pt_minho':    (41.5610,  -8.3960),   # Universidade do Minho, Braga
        'ex_pt_aveiro':   (40.6300,  -8.6570),   # Universidade de Aveiro
        'ex_pt_nova':     (38.7370,  -9.1540),   # Universidade NOVA de Lisboa (Campolide)
        'ex_ar_uba':      (-34.5990, -58.3730),  # UBA, Buenos Aires
        'ex_es_ucm':      (40.4520,  -3.7280),   # UCM, Madrid
        'ex_us_florida':  (29.6516, -82.3248),   # University of Florida, Gainesville
        'ex_mx_unam':      (19.3322, -99.1860),   # Cidade do México
        'ex_es_uam':       (40.5439,  -3.6974),   # Madrid
        'ex_uy_udelar':    (-34.9011, -56.1645),  # Montevidéu
        'ex_es_salamanca': (40.9629,  -5.6689),   # Salamanca
        'ex_us_michigan':  (42.2780, -83.7382),   # Ann Arbor, MI
        'ex_ca_ubc':       (49.2606, -123.2460),  # Vancouver
        'ex_us_columbia':  (40.8075, -73.9626),   # NYC
        'ex_us_ucdavis':   (38.5382, -121.7617),  # Davis, CA
        'ex_co_unal':      ( 4.6386, -74.0841),   # Bogotá
        'ex_us_cornell':   (42.4534, -76.4735),   # Ithaca, NY
        'ex_ar_unlp':      (-34.9092, -57.9463),  # La Plata
        'ex_nl_wageningen':(51.9851,   5.6630),   # Wageningen
        'ex_us_texas':     (30.2849, -97.7341),   # Austin, TX
        'ex_us_wisconsin': (43.0766, -89.4125),   # Madison, WI
        'ex_es_sevilla':   (37.3772,  -5.9869),   # Sevilla
        'ex_co_antioquia' : (6.2675, -75.5680),     # Medellín
        'ex_us_nyu'       : (40.7295, -73.9965),    # NYC
        'ex_es_usc'       : (42.8770, -8.5540),     # Santiago de Compostela
        'ex_us_yale'      : (41.3163, -72.9223),    # New Haven, CT
        'ex_ca_alberta'   : (53.5232, -113.5263),   # Edmonton
        'ex_us_ucsd'      : (32.8801, -117.2340),   # San Diego, CA
        'ex_es_granada'   : (37.1847, -3.6056),     # Granada
        'ex_us_ohiost'    : (40.0076, -83.0300),    # Columbus, OH
        'ex_us_jhu'       : (39.3299, -76.6205),    # Baltimore, MD
        'ex_ca_ottawa'    : (45.4231, -75.6831),    # Ottawa
        'ex_dk_copenhagen': (55.6805, 12.5714),     # Copenhagen
        'ex_ar_unc'       : (-31.4420, -64.1930),   # Córdoba
        'ex_us_penn'      : (39.9522, -75.1932),    # Philadelphia, PA
        'ex_au_sydney'    : (-33.8880, 151.1875),   # Sydney
        'ex_pe_sanmarcos' : (-12.0563, -77.0844),   # Lima
        'ex_dk_aarhus'    : (56.1681, 10.2027),     # Aarhus
        'ex_uk_manchester': (53.4670, -2.2330),     # Manchester
        'ex_us_msu'       : (42.7018, -84.4822),    # East Lansing, MI
        'ex_us_ucsystem'  : (37.8719, -122.2585),   # Referência Berkeley
        'ex_us_ucla'      : (34.0689, -118.4452),   # Los Angeles, CA
        'ex_be_ghent'     : (51.0470, 3.7270),      # Ghent
        'ex_us_duke'      : (36.0014, -78.9382),    # Durham, NC
        'ex_au_uq'        : (-27.4975, 153.0137),   # Brisbane
        'ex_nl_leiden'    : (52.1601, 4.4970),      # Leiden
        'ex_us_brown'     : (41.8268, -71.4025),    # Providence, RI
        'ex_us_uiuc'      : (40.1020, -88.2272),    # Urbana-Champaign, IL
        'ex_nl_delft'     : (52.0020, 4.3700),      # Delft
        'ex_uk_nottingham': (52.9380, -1.1940),     # Nottingham
        'ex_uk_leeds'     : (53.8067, -1.5550),     # Leeds
        'ex_uk_bristol'   : (51.4584, -2.6030),     # Bristol
        'ex_co_valle'     : (3.3752, -76.5340),     # Cali
        'ex_uk_london'    : (51.5072, -0.1276),     # Londres
        'ex_es_zaragoza'  : (41.6488, -0.8891),     # Zaragoza
        'ex_us_chicago'   : (41.7897, -87.5997),    # Chicago, IL
        'ex_us_purdue'    : (40.4237, -86.9212),    # West Lafayette, IN
        'ex_us_arizona'   : (32.2319, -110.9501),   # Tucson, AZ
        'ex_us_ncsu'      : (35.7847, -78.6821),    # Raleigh, NC
        'ex_uk_birmingham': (52.4508, -1.9305),     # Birmingham
        'ex_se_lund'      : (55.7058, 13.1932),     # Lund
        'ex_ar_unr'       : (-32.9525, -60.6394),   # Rosario
        'ex_us_princeton' : (40.3431, -74.6551),    # Princeton, NJ
        'ex_cl_uchile'    : (-33.4489, -70.6633),   # Santiago
        'ex_ca_waterloo'  : (43.4723, -80.5449),    # Waterloo
        'ex_se_uppsala'   : (59.8586, 17.6389),     # Uppsala
        'ex_us_georgia'   : (33.9480, -83.3773),    # Athens, GA
        'ex_es_upm'       : (40.4418, -3.7294),     # Madrid
        'ex_es_cadiz'     : (36.5297, -6.2927),     # Cádiz
        'ex_us_pitt'      : (40.4444, -79.9608),    # Pittsburgh, PA
        'ex_uk_edinburgh' : (55.9444, -3.1888),     # Edinburgh
        'ex_co_unal_all'  : (4.6386, -74.0841),     # Bogotá (sede principal)
    }

    # Fallback por centróide de país (para EX sem coordenada específica)
    country_centroids = {
        'BR': (-14.2350, -51.9253),
        'US': (39.8283,  -98.5795),
        'UK': (55.3781,   -3.4360),
        'CH': (46.8182,     8.2275),
        'DE': (51.1657,    10.4515),
        'CA': (56.1304,  -106.3468),
        'JP': (36.2048,   138.2529),
        'CN': (35.8617,   104.1954),
        'FR': (46.2276,     2.2137),
        'IT': (41.8719,    12.5674),
        'ES': (40.4637,    -3.7492),
        'PT': (39.3999,    -8.2245),
        'NL': (52.1326,     5.2913),
        'SE': (60.1282,    18.6435),
        'AU': (-25.2744,  133.7751),
        'AR': (-38.4161,  -63.6167),
        'OT': (30.0,         0.0),
    }

    def ex_to_country(ex_code: str) -> str:
        m = re.match(r'ex_([a-z]{2})_', str(ex_code))
        return m.group(1).upper() if m else None

    # ======================
    # 2) Contar fluxos BR -> EX (específicos)
    # ======================
    flux_br_to_ex = Counter()
    for _, row in df_tabela_sequencias.iterrows():
        traj = row['universidade_lista']
        if not isinstance(traj, list) or len(traj) < 2:
            continue
        for i in range(1, len(traj)):
            ori, dst = traj[i-1], traj[i]
            if isinstance(ori, str) and isinstance(dst, str):
                # filtra apenas destinos EX específicos (descarta ex_outros)
                if ori.startswith('br') and dst.startswith('ex_') and dst != 'ex_outros':
                    flux_br_to_ex[(ori, dst)] += 1

    print(f"Total de pares BR→EX diferentes (sem ex_outros): {len(flux_br_to_ex)}")
    # ======================
    # 3) Filtros para reduzir ruído
    # ======================
    MIN_W = 1         # ↓ teste mínimo para não perder fluxos fracos (ajuste depois)
    TOP_K_POR_BR = None  # None = sem corte Top-K (ajuste depois)

    # Aplica peso mínimo
    flux_filtrado = {k: w for k, w in flux_br_to_ex.items() if w >= MIN_W}

    # Top-K por BR (opcional)
    if TOP_K_POR_BR is not None:
        by_br = defaultdict(list)
        for (br, ex), w in flux_filtrado.items():
            by_br[br].append((ex, w))
        flux_compacto = {}
        for br, lst in by_br.items():
            lst.sort(key=lambda x: x[1], reverse=True)
            for ex, w in lst[:TOP_K_POR_BR]:
                flux_compacto[(br, ex)] = w
    else:
        flux_compacto = flux_filtrado

    print(f"Arestas após filtros: {len(flux_compacto)}")

    # Log: EX que faltam coordenadas específicas
    ex_sem_coord = sorted({ex for (_, ex) in flux_compacto.keys() if ex not in coords_ex})
    print("EX sem coordenada específica (usando fallback por país):", ex_sem_coord[:50])

    # ======================
    # 4) Plot em mapa (Plotly)
    # ======================
    def scale(val, vmin, vmax, out_min, out_max):
        if vmax - vmin < 1e-9:
            return (out_min + out_max) / 2.0
        return out_min + (val - vmin) * (out_max - out_min) / (vmax - vmin)

    fig = go.Figure()

    # Normalização de espessura/opacidade
    if flux_compacto:
        pesos = np.array(list(flux_compacto.values()), dtype=float)
        pmin, pmax = pesos.min(), pesos.max()
    else:
        pesos = np.array([1.0]); pmin = pmax = 1.0

    # 4.1) Linhas BR -> EX (com fallback por país)
    arestas_plotadas = 0
    for (br, ex), w in sorted(flux_compacto.items(), key=lambda x: -x[1]):
        lat1, lon1 = coords_br.get(br, (None, None))
        lat2, lon2 = coords_ex.get(ex, (None, None))

        if lat1 is None:
            continue

        if lat2 is None:
            pais = ex_to_country(ex) or 'OT'
            lat2, lon2 = country_centroids.get(pais, country_centroids['OT'])

        width = scale(w, pmin, pmax, 1.0, 6.0)
        alpha = scale(w, pmin, pmax, 0.3, 0.9)
        fig.add_trace(go.Scattergeo(
            lon=[lon1, lon2],
            lat=[lat1, lat2],
            mode='lines',
            line=dict(width=width, color=f'rgba(66, 135, 245, {alpha})'),
            hoverinfo='text',
            text=f"{label_br.get(br, br)} → {label_ex.get(ex, ex)}: {w}"
        ))
        arestas_plotadas += 1

    # ======================
    # 4.2 e 4.3) Marcadores com hover detalhado
    # ======================

    def lbl_br(c): return label_br.get(c, c)
    def lbl_ex(c): return label_ex.get(c, c)

    # Detalhamento: ex -> {br: w}, br -> {ex: w}
    detalhe_ex = defaultdict(Counter)
    detalhe_br = defaultdict(Counter)
    for (br, ex), w in flux_compacto.items():
        detalhe_ex[ex][br] += w
        detalhe_br[br][ex] += w

    # Totais por nó
    tot_por_br = Counter()
    tot_por_ex = Counter()
    for (br, ex), w in flux_compacto.items():
        tot_por_br[br] += w
        tot_por_ex[ex] += w

    # 4.2) BR markers (saídas) com hover detalhado
    if tot_por_br:
        vals = list(tot_por_br.values())
        bmin, bmax = min(vals), max(vals)
        br_lons, br_lats, br_sizes, br_texts, br_labels = [], [], [], [], []
        for br, total in sorted(tot_por_br.items(), key=lambda x: -x[1]):
            lat, lon = coords_br.get(br, (None, None))
            if lat is None:
                continue
            # monte hover
            dests = detalhe_br.get(br, {})
            linhas = [f"{lbl_br(br)} — Total saídas: {total}"]
            for ex, w in sorted(dests.items(), key=lambda x: -x[1]):
                linhas.append(f"  ~ {lbl_ex(ex)}: {w}")
            hover_txt = "<br>".join(linhas)

            br_lons.append(lon); br_lats.append(lat)
            br_sizes.append(scale(total, bmin, bmax, 8, 22))
            br_texts.append(hover_txt)
            br_labels.append(lbl_br(br))

        fig.add_trace(go.Scattergeo(
            name='BR',
            lon=br_lons, lat=br_lats,
            mode='markers',
            text=br_labels,
            textposition='bottom center',
            hovertext=br_texts, hoverinfo='text',
            marker=dict(size=br_sizes, color='royalblue', line=dict(width=0.8, color='white'))
        ))

    # 4.3) EX markers (entradas) com hover detalhado
    if tot_por_ex:
        vals = list(tot_por_ex.values())
        emin, emax = min(vals), max(vals)
        ex_lons, ex_lats, ex_sizes, ex_texts, ex_labels = [], [], [], [], []
        for ex, total in sorted(tot_por_ex.items(), key=lambda x: -x[1]):
            lat, lon = coords_ex.get(ex, (None, None))
            if lat is None:
                pais = ex_to_country(ex) or 'OT'
                lat, lon = country_centroids.get(pais, country_centroids['OT'])
            fontes = detalhe_ex.get(ex, {})
            linhas = [f"{lbl_ex(ex)} — Total entradas: {total}"]
            for br, w in sorted(fontes.items(), key=lambda x: -x[1]):
                linhas.append(f"  ~ {lbl_br(br)}: {w}")
            hover_txt = "<br>".join(linhas)

            ex_lons.append(lon); ex_lats.append(lat)
            ex_sizes.append(scale(total, emin, emax, 8, 26))
            ex_texts.append(hover_txt)
            ex_labels.append(lbl_ex(ex))

        fig.add_trace(go.Scattergeo(
            name='EX',
            lon=ex_lons, lat=ex_lats,
            mode='markers',
            text=ex_labels,
            textposition='top center',
            hovertext=ex_texts, hoverinfo='text',
            marker=dict(size=ex_sizes, color='orange', line=dict(width=0.8, color='white'))
        ))

    # 4.4) Estilo do mapa
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#3ba96a",
        coastlinecolor="#2e8553",
        showcoastlines=True,
        landcolor="#2e8553",
        oceancolor="#2f4499",
        showocean=True
    )

    # Ajuste de cores (linhas e marcadores) com tratamento para name=None
    for trace in fig.data:
        mode = (getattr(trace, 'mode', '') or '').lower()
        name = (getattr(trace, 'name', '') or '')

        # Linhas (fluxos)
        if 'lines' in mode and hasattr(trace, 'line') and trace.line is not None:
            trace.line.color = '#df3b43'  # vermelho escuro suave

        # Marcadores (universidades BR/EX)
        if 'markers' in mode and hasattr(trace, 'marker') and trace.marker is not None:
            is_br = 'BR' in name.upper()
            is_ex = 'EX' in name.upper()

            # Fallback por cor original (se necessário)
            if not (is_br or is_ex):
                current_color = getattr(trace.marker, 'color', None)
                if isinstance(current_color, str):
                    c = current_color.lower()
                    if c in ('royalblue', '#4169e1', 'rgb(65, 105, 225)'):
                        is_br = True
                    elif c in ('orange', '#ffa500', '#f89c74'):
                        is_ex = True

            # Cores ajustadas
            if is_br:
                trace.marker.color = "#007bff"   # BR
            elif is_ex:
                trace.marker.color = "#ffc107"   # EX
            else:
                trace.marker.color = "#28a745"   # neutro

            # Texto das labels (branco, leve "stroke" via shadow se disponível)
            trace.textfont = dict(
                color='white',
                size=12,
                family='Arial',
                weight='bold',
                shadow="black 1px 1px 2px"  # pode ser ignorado em versões antigas do Plotly
            )


    fig.update_layout(
        title=f"Fluxos BR → Universidades Estrangeiras (arestas ≥ {MIN_W}; Top {TOP_K_POR_BR} por BR)",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    fig.show()
    print(f"Arestas desenhadas: {arestas_plotadas}")

    ## MAPA USP -> EX ##
    # ======================
    # 4) Plot em mapa (Plotly) - Focado na USP (br1)
    # ======================
    def scale(val, vmin, vmax, out_min, out_max):
        if vmax - vmin < 1e-9:
            return (out_min + out_max) / 2.0
        return out_min + (val - vmin) * (out_max - out_min) / (vmax - vmin)

    fig = go.Figure()

    # Normalização de espessura/opacidade (considerando todos os fluxos para manter a escala consistente)
    if flux_compacto:
        pesos = np.array(list(flux_compacto.values()), dtype=float)
        pmin, pmax = pesos.min(), pesos.max()
    else:
        pesos = np.array([1.0]); pmin = pmax = 1.0

    # 4.1) Linhas BR -> EX (filtradas para USP)
    arestas_plotadas = 0
    for (br, ex), w in sorted(flux_compacto.items(), key=lambda x: -x[1]):

        # <<< ALTERAÇÃO 1: Mantém o filtro para desenhar apenas as linhas da USP >>>
        if br != 'br1':
            continue

        lat1, lon1 = coords_br.get(br, (None, None))
        lat2, lon2 = coords_ex.get(ex, (None, None))

        if lat1 is None:
            continue

        if lat2 is None:
            pais = ex_to_country(ex) or 'OT'
            lat2, lon2 = country_centroids.get(pais, country_centroids['OT'])

        width = scale(w, pmin, pmax, 1.0, 6.0)
        alpha = scale(w, pmin, pmax, 0.3, 0.9)
        fig.add_trace(go.Scattergeo(
            lon=[lon1, lon2],
            lat=[lat1, lat2],
            mode='lines',
            line=dict(width=width, color=f'rgba(66, 135, 245, {alpha})'),
            hoverinfo='text',
            text=f"{label_br.get(br, br)} → {label_ex.get(ex, ex)}: {w}"
        ))
        arestas_plotadas += 1

    # ======================
    # 4.2 e 4.3) Marcadores com hover detalhado (filtrados para USP e seus destinos)
    # ======================

    def lbl_br(c): return label_br.get(c, c)
    def lbl_ex(c): return label_ex.get(c, c)

    detalhe_ex = defaultdict(Counter)
    detalhe_br = defaultdict(Counter)
    for (br, ex), w in flux_compacto.items():
        detalhe_ex[ex][br] += w
        detalhe_br[br][ex] += w

    tot_por_br = Counter({br: sum(dests.values()) for br, dests in detalhe_br.items()})
    tot_por_ex = Counter({ex: sum(origs.values()) for ex, origs in detalhe_ex.items()})

    # 4.2) Marcador da USP (origem)
    if 'br1' in tot_por_br:
        br = 'br1'
        total = tot_por_br[br]
        lat, lon = coords_br.get(br, (None, None))

        if lat is not None:
            dests = detalhe_br.get(br, {})
            linhas = [f"{lbl_br(br)} — Total saídas: {total}"]
            for ex, w in sorted(dests.items(), key=lambda x: -x[1]):
                linhas.append(f"  ~ {lbl_ex(ex)}: {w}")
            hover_txt = "<br>".join(linhas)

            fig.add_trace(go.Scattergeo(
                name='BR',
                lon=[lon], lat=[lat],
                mode='markers+text',
                text=[lbl_br(br)],
                textposition='bottom center',
                hovertext=[hover_txt], hoverinfo='text',
                marker=dict(size=22, color='royalblue', line=dict(width=0.8, color='white'))
            ))

    # 4.3) Marcadores dos destinos da USP
    # <<< ALTERAÇÃO 2: Primeiro, pegamos a lista de destinos APENAS da USP >>>
    destinos_da_usp = set(detalhe_br.get('br1', {}).keys())

    if tot_por_ex and destinos_da_usp:
        vals = [tot_por_ex[ex] for ex in destinos_da_usp if ex in tot_por_ex]
        emin, emax = min(vals) if vals else 1, max(vals) if vals else 1

        ex_lons, ex_lats, ex_sizes, ex_texts, ex_labels = [], [], [], [], []

        for ex, total in sorted(tot_por_ex.items(), key=lambda x: -x[1]):
            # <<< ALTERAÇÃO 3: Desenha o marcador somente se for um destino da USP >>>
            if ex not in destinos_da_usp:
                continue

            lat, lon = coords_ex.get(ex, (None, None))
            if lat is None:
                pais = ex_to_country(ex) or 'OT'
                lat, lon = country_centroids.get(pais, country_centroids['OT'])

            # O hovertext aqui mostrará o total de entradas para a universidade (de todas as origens BR),
            # mas o marcador só aparece se a USP for uma das origens.
            fontes = detalhe_ex.get(ex, {})
            linhas = [f"{lbl_ex(ex)} — Total entradas: {total}"]
            for br_origem, w in sorted(fontes.items(), key=lambda x: -x[1]):
                linhas.append(f"  ~ {lbl_br(br_origem)}: {w}")
            hover_txt = "<br>".join(linhas)

            ex_lons.append(lon); ex_lats.append(lat)
            ex_sizes.append(scale(total, emin, emax, 8, 26))
            ex_texts.append(hover_txt)
            ex_labels.append(lbl_ex(ex))

        fig.add_trace(go.Scattergeo(
            name='EX',
            lon=ex_lons, lat=ex_lats,
            mode='markers',
            text=ex_labels,
            textposition='top center',
            hovertext=ex_texts, hoverinfo='text',
            marker=dict(size=ex_sizes, color='orange', line=dict(width=0.8, color='white'))
        ))
    # 4.4) Estilo do mapa
    fig.update_geos(
        projection_type="natural earth",
        showcountries=True,
        countrycolor="#3ba96a",
        coastlinecolor="#2e8553",
        showcoastlines=True,
        landcolor="#2e8553",
        oceancolor="#2f4499",
        showocean=True
    )

    # Ajuste de cores (linhas e marcadores) com tratamento para name=None
    for trace in fig.data:
        mode = (getattr(trace, 'mode', '') or '').lower()
        name = (getattr(trace, 'name', '') or '')

        # Linhas (fluxos)
        if 'lines' in mode and hasattr(trace, 'line') and trace.line is not None:
            trace.line.color = '#df3b43'  # vermelho escuro suave

        # Marcadores (universidades BR/EX)
        if 'markers' in mode and hasattr(trace, 'marker') and trace.marker is not None:
            is_br = 'BR' in name.upper()
            is_ex = 'EX' in name.upper()

            # Fallback por cor original (se necessário)
            if not (is_br or is_ex):
                current_color = getattr(trace.marker, 'color', None)
                if isinstance(current_color, str):
                    c = current_color.lower()
                    if c in ('royalblue', '#4169e1', 'rgb(65, 105, 225)'):
                        is_br = True
                    elif c in ('orange', '#ffa500', '#f89c74'):
                        is_ex = True

            # Cores ajustadas
            if is_br:
                trace.marker.color = "#007bff"   # BR
            elif is_ex:
                trace.marker.color = "#ffc107"   # EX
            else:
                trace.marker.color = "#28a745"   # neutro

            # Texto das labels (branco, leve "stroke" via shadow se disponível)
            trace.textfont = dict(
                color='white',
                size=12,
                family='Arial',
                weight='bold',
                shadow="black 1px 1px 2px"  # pode ser ignorado em versões antigas do Plotly
            )


    fig.update_layout(
        title=f"Fluxos Universidade De São Paulo (USP) → Universidades Estrangeiras (arestas ≥ {MIN_W}; Top {TOP_K_POR_BR} por BR)",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )

    fig.show()
    print(f"Arestas desenhadas: {arestas_plotadas}")

if __name__ == "__main__":
    main()
