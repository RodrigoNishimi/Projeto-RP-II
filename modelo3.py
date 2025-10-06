# modelo_gravitacional_ppml.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from constantes import (
    LINGUA_UNIVERSIDADES,
    MASSA_UNIVERSIDADES_EX,
    MASSA_UNIVERSIDADES_BR,
)

warnings.filterwarnings("ignore")


def preparar_dados(mat_fluxo_path, mat_distancias_path, dataset_capes_path):
    # Leitura
    mat_fluxo = pd.read_csv(mat_fluxo_path, index_col=0)
    mat_dist = pd.read_csv(mat_distancias_path, index_col=0)
    dataset_capes = pd.read_csv(dataset_capes_path, index_col=0)

    # 1. transformar em formato longo
    fluxo_long = mat_fluxo.stack().reset_index()
    fluxo_long.columns = ["BR_Origem", "EX_Destino", "Fluxo"]

    dist_long = mat_dist.stack().reset_index()
    dist_long.columns = ["BR_Origem", "EX_Destino", "Distancia"]

    df = pd.merge(fluxo_long, dist_long, on=["BR_Origem", "EX_Destino"], how="left")

    # 2. massas (origem/destino derivadas do fluxo total)
    massa_origem = mat_fluxo.sum(axis=1)
    massa_destino = mat_fluxo.sum(axis=0)

    df["Massa_Origem"] = df["BR_Origem"].map(massa_origem)
    df["Massa_Destino"] = df["EX_Destino"].map(massa_destino)

    # 3. massas externas (ranking/prestígio) se estiverem disponíveis nas constantes
    df["Massa_BR_rank"] = df["BR_Origem"].map(MASSA_UNIVERSIDADES_BR)
    df["Massa_EX_rank"] = df["EX_Destino"].map(MASSA_UNIVERSIDADES_EX)

    # 4. língua: mesma língua entre origem e destino (mais interpretável que muitas dummies)
    lingua_origem = df["BR_Origem"].map(LINGUA_UNIVERSIDADES)
    lingua_destino = df["EX_Destino"].map(LINGUA_UNIVERSIDADES)
    df["Same_Language"] = (lingua_origem == lingua_destino).astype(int)
    # Caso queira manter dummies separadas, descomente:
    # dummies_l = pd.get_dummies(lingua_destino.fillna("Desconhecido"), prefix="ling_dest")
    # df = pd.concat([df, dummies_l], axis=1)

    # 5. similaridade de áreas (Jaccard) - processar dataset_capes
    # assumir dataset_capes com colunas codigo_origem, codigo_destino, area_avaliacao (conforme original)
    areas_por_univ = {}
    for _, r in dataset_capes.iterrows():
        o = r.get("codigo_origem")
        d = r.get("codigo_destino")
        a = r.get("area_avaliacao")
        if pd.notna(a):
            if pd.notna(o):
                areas_por_univ.setdefault(o, set()).add(a)
            if pd.notna(d):
                areas_por_univ.setdefault(d, set()).add(a)

    # função vectorizada de jaccard
    def jaccard_pair(br, ex):
        Ao = areas_por_univ.get(br, set())
        Ad = areas_por_univ.get(ex, set())
        if not Ao or not Ad:
            return 0.0
        inter = len(Ao & Ad)
        uni = len(Ao | Ad)
        return inter / uni if uni > 0 else 0.0

    df["Similaridade_Areas"] = df.apply(
        lambda r: jaccard_pair(r["BR_Origem"], r["EX_Destino"]), axis=1
    )

    # 6. tratar missings e transformar variáveis contínuas
    # manter linhas com Fluxo == 0 (PPML aceita zeros)
    # porém devemos excluir pares sem distância/massa_rank se forem essenciais
    df = df[df["Distancia"].notna() & df["Distancia"] > 0].copy()

    # preencher massas derivadas do ranking com small value se NA, para não remover observações
    df["Massa_BR_rank"] = df["Massa_BR_rank"].fillna(df["Massa_BR_rank"].median())
    df["Massa_EX_rank"] = df["Massa_EX_rank"].fillna(df["Massa_EX_rank"].median())

    # logs (features)
    df["log_Distancia"] = np.log(df["Distancia"].astype(float))
    # massas do fluxo (pode haver zeros => adicionar 1 antes do log para estabilidade)
    df["log_Massa_Origem"] = np.log(df["Massa_Origem"].fillna(0) + 1)
    df["log_Massa_Destino"] = np.log(df["Massa_Destino"].fillna(0) + 1)
    # massas de ranking
    df["log_Massa_BR_rank"] = np.log(df["Massa_BR_rank"].astype(float))
    df["log_Massa_EX_rank"] = np.log(df["Massa_EX_rank"].astype(float))

    # reset index
    df = df.reset_index(drop=True)
    return df


def diagnosticos_overdispersion(df):
    mean_y = df["Fluxo"].mean()
    var_y = df["Fluxo"].var()
    print(f"Fluxo: média = {mean_y:.3f}, variância = {var_y:.3f}")
    if var_y > mean_y:
        print(
            "Sinais de overdispersion (var > mean) -> considerar Negative Binomial ou Zero-Inflated."
        )
    else:
        print("Sem forte overdispersion aparente.")


def _calc_vif_df(df, features):
    X = df[features].fillna(0).astype(float)
    Xc = add_constant(X, has_constant="add")
    vif = pd.DataFrame(
        {
            "feature": Xc.columns,
            "VIF": [
                variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])
            ],
        }
    )
    return vif


def ajustar_ppml(
    df,
    include_fixed_effects=False,
    try_cluster=True,
    cluster_col="BR_Origem",
    force_drop_high_vif=False,
    vif_threshold=50,
):
    base_vars = [
        "log_Massa_Origem",
        "log_Massa_Destino",
        "log_Massa_BR_rank",
        "log_Massa_EX_rank",
        "Similaridade_Areas",
        "log_Distancia",
        "Same_Language",
    ]

    # opcional: dropar variáveis com VIF altíssimo
    if force_drop_high_vif:
        vif = _calc_vif_df(df, base_vars)
        print("VIF inicial:\n", vif.sort_values("VIF", ascending=False))
        high = vif[(vif["VIF"] > vif_threshold) & (vif["feature"] != "const")][
            "feature"
        ].tolist()
        # remover 'const' se aparecer na lista
        high = [h for h in high if h in base_vars]
        if high:
            print("Dropping high-VIF features:", high)
            base_vars = [b for b in base_vars if b not in high]

    if include_fixed_effects:
        formula = "Fluxo ~ " + " + ".join(base_vars) + " + C(BR_Origem) + C(EX_Destino)"
    else:
        formula = "Fluxo ~ " + " + ".join(base_vars)

    print("Fórmula:", formula)

    # split
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 1) Fit sem solicitar cov_type -> evita cálculo imediato que deu erro
    model = smf.glm(formula=formula, data=train, family=sm.families.Poisson())
    try:
        res_plain = model.fit()
    except Exception as e:
        print("Erro ao ajustar o GLM (fit()):", e)
        raise

    # 2) Tentar obter covariância clusterizada / robusta sem quebrar o fit
    cov_result = None
    if try_cluster:
        try:
            cov_result = res_plain.get_robustcov_results(
                cov_type="cluster", cov_kwds={"groups": train[cluster_col]}
            )
            print("Usando cluster robust cov (por {}).".format(cluster_col))
        except np.linalg.LinAlgError as e:
            print("Cluster robust cov falhou por LinAlgError:", e)
            # tentativa fallback HC3
            try:
                cov_result = res_plain.get_robustcov_results(cov_type="HC3")
                print("Fallback: usando HC3 robust cov.")
            except Exception as e2:
                print("Fallback HC3 também falhou:", e2)
                cov_result = res_plain  # usar resultado sem robust cov
        except Exception as e:
            print("Erro ao tentar cluster cov:", e)
            try:
                cov_result = res_plain.get_robustcov_results(cov_type="HC3")
                print("Fallback: usando HC3 robust cov (exceção genérica).")
            except:
                cov_result = res_plain
    else:
        cov_result = res_plain

    # 3) Predição e métricas
    test["Predito"] = cov_result.predict(test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    rmse = np.sqrt(mean_squared_error(test["Fluxo"], test["Predito"]))
    mae = mean_absolute_error(test["Fluxo"], test["Predito"])
    print(f"RMSE (test): {rmse:.3f}, MAE: {mae:.3f}")

    return cov_result, train, test


def plot_real_vs_pred(test_df):
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df["Fluxo"], test_df["Predito"], alpha=0.6)
    maxv = max(test_df["Fluxo"].max(), test_df["Predito"].max())
    plt.plot([0, maxv], [0, maxv], "r--")
    plt.xlabel("Fluxo Real")
    plt.ylabel("Fluxo Predito (PPML)")
    plt.title("Real vs Predito (test set)")
    plt.grid(True)
    plt.show()


def main():
    df = preparar_dados(
        "./dados/mat_br_to_ex.csv",
        "./dados/matriz_distancias.csv",
        "./dados/br-capes-filtrado.csv",
    )
    print("Observações após preparação:", len(df))
    print(df[["BR_Origem", "EX_Destino", "Fluxo", "Distancia", "log_Distancia"]].head())

    diagnosticos_overdispersion(df)

    # quick VIF check (apenas sobre variáveis contínuas selecionadas, sem FE)
    X_vif = df[
        [
            "log_Massa_Origem",
            "log_Massa_Destino",
            "log_Massa_BR_rank",
            "log_Massa_EX_rank",
            "Similaridade_Areas",
            "log_Distancia",
            "Same_Language",
        ]
    ].fillna(0)
    print("\nVIF (valores altos >10 indicam multicolinearidade):")
    print(_calc_vif_df(X_vif, X_vif.columns.tolist()))

    # ajustar PPML sem efeitos fixos (mais rápido)
    res, train, test = ajustar_ppml(
        df, include_fixed_effects=False, try_cluster=True, cluster_col="BR_Origem"
    )

    print("\nResumo do modelo (PPML):")
    print(res.summary())

    # avaliar e plotar
    plot_real_vs_pred(test)

    # Se quiser: ajustar com efeitos fixos (pode ser custoso se houver muitas universidades)
    # res_fe, train_fe, test_fe = ajustar_ppml(df, include_fixed_effects=True, cluster_by_origin=True)
    # print(res_fe.summary())


if __name__ == "__main__":
    main()
