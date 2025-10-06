import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from constantes import LINGUA_UNIVERSIDADES
from constantes import MASSA_UNIVERSIDADES_BR
from constantes import MASSA_UNIVERSIDADES_EX


def criar_dummies_lingua(df):
    """Cria variáveis dummy para cada língua"""
    linguas_destino = df["EX_Destino"].map(LINGUA_UNIVERSIDADES).fillna("Desconhecido")
    dummies_lingua = pd.get_dummies(linguas_destino, prefix="lingua").astype(float)
    return pd.concat([df, dummies_lingua], axis=1)


def processar_areas(dataset_capes):
    """Cria um dicionário com todas as áreas de cada universidade"""
    areas_por_univ = {}
    for _, row in dataset_capes.iterrows():
        origem = row["codigo_origem"]
        destino = row["codigo_destino"]
        area = row["area_avaliacao"]

        if pd.notna(area):
            if origem not in areas_por_univ:
                areas_por_univ[origem] = set()
            if destino not in areas_por_univ:
                areas_por_univ[destino] = set()

            areas_por_univ[origem].add(area)
            areas_por_univ[destino].add(area)

    return areas_por_univ


def calcular_similaridade_areas(row, areas_por_univ):
    """Calcula índice de Jaccard para similaridade entre áreas"""
    origem = row["BR_Origem"]
    destino = row["EX_Destino"]

    areas_origem = areas_por_univ.get(origem, set())
    areas_destino = areas_por_univ.get(destino, set())

    if not areas_origem or not areas_destino:
        return 0.0

    # Índice de Jaccard: intersecção / união
    intersecao = len(areas_origem.intersection(areas_destino))
    uniao = len(areas_origem.union(areas_destino))

    return float(intersecao) / float(uniao) if uniao > 0 else 0.0


def main():
    mat_fluxo = pd.read_csv("./dados/mat_br_to_ex.csv", index_col=0)
    mat_distancias = pd.read_csv("./dados/matriz_distancias.csv", index_col=0)
    dataset_capes = pd.read_csv("./dados/br-capes-filtrado.csv", index_col=0)

    """
    Args:
        mat_fluxo (pd.DataFrame): Matriz com fluxos de pesquisadores (origem BR, destino EX).
        mat_distancias (pd.DataFrame): Matriz com as distâncias entre as universidades.
        lingua_universidades (pd.Series): Série com a língua principal de cada universidade.
        dataset_capes (pd.DataFrame): Dataset da CAPES com informações de mobilidade.
    """

    # Processar áreas e calcular similaridades
    areas_por_univ = processar_areas(dataset_capes)

    # 1. Preparar os dados para a regressão
    # Transformar as matrizes em um formato "longo" (uma linha por par origem-destino)
    fluxo_long = mat_fluxo.stack().reset_index()
    fluxo_long.columns = ["BR_Origem", "EX_Destino", "Fluxo"]

    dist_long = mat_distancias.stack().reset_index()
    dist_long.columns = ["BR_Origem", "EX_Destino", "Distancia"]

    # Unir os dataframes de fluxo e distância
    df_gravitacional = pd.merge(fluxo_long, dist_long, on=["BR_Origem", "EX_Destino"])

    # 2. Calcular as "massas" de origem e destino
    # Massa de Origem (Oi): Total de pesquisadores saindo de cada universidade BR
    massa_origem = mat_fluxo.sum(axis=1)

    # Massa de Destino (Dj): Total de pesquisadores chegando em cada universidade EX
    massa_destino = mat_fluxo.sum(axis=0)

    # Adicionar as massas ao dataframe principal
    df_gravitacional["Massa_Origem"] = df_gravitacional["BR_Origem"].map(massa_origem)
    df_gravitacional["Massa_Destino"] = df_gravitacional["EX_Destino"].map(
        massa_destino
    )

    df_gravitacional["Massa_BR"] = df_gravitacional["BR_Origem"].map(
        MASSA_UNIVERSIDADES_BR
    )

    df_gravitacional["Massa_EX"] = df_gravitacional["EX_Destino"].map(
        MASSA_UNIVERSIDADES_EX
    )

    # 3. Filtrar dados e aplicar logaritmo
    # O modelo log-linear não funciona com valores nulos ou zero.
    # Filtramos pares sem fluxo e onde a massa ou distância seja zero.
    df_gravitacional = df_gravitacional[
        (df_gravitacional["Fluxo"] > 0)
        & (df_gravitacional["Massa_Origem"] > 0)
        & (df_gravitacional["Massa_Destino"] > 0)
        & (df_gravitacional["Distancia"] > 0)
        & (df_gravitacional["Massa_BR"] > 0)
        & (df_gravitacional["Massa_EX"] > 0)
    ].copy()

    # Aplicar logaritmos
    colunas_log = [
        ("Fluxo", "log_Fluxo"),
        ("Massa_Origem", "log_Massa_Origem"),
        ("Massa_Destino", "log_Massa_Destino"),
        ("Distancia", "log_Distancia"),
        ("Massa_BR", "log_Massa_BR"),
        ("Massa_EX", "log_Massa_EX"),
    ]

    for col_orig, col_log in colunas_log:
        df_gravitacional[col_log] = np.log(df_gravitacional[col_orig].astype(float))

    print(
        f"\nNúmero de observações (pares universidade) para o modelo: {len(df_gravitacional)}"
    )
    print("Amostra dos dados preparados para regressão:")
    print(df_gravitacional.head())

    # Criar dummies para línguas
    df_gravitacional = criar_dummies_lingua(df_gravitacional)

    # Calcular similaridade de áreas
    df_gravitacional["Similaridade_Areas"] = df_gravitacional.apply(
        lambda row: calcular_similaridade_areas(row, areas_por_univ), axis=1
    )

    # 4. Preparar variáveis para o modelo
    colunas_lingua = [
        col for col in df_gravitacional.columns if col.startswith("lingua_")
    ]
    colunas_modelo = [
        "log_Massa_Origem",
        "log_Massa_Destino",
        "log_Massa_BR",
        "log_Massa_EX",
        "Similaridade_Areas",
        "log_Distancia",
    ] + colunas_lingua

    # Verificar e converter todas as variáveis para float
    for col in colunas_modelo:
        df_gravitacional[col] = pd.to_numeric(df_gravitacional[col], errors="coerce")

    # Remover linhas com valores nulos
    df_gravitacional = df_gravitacional.dropna(subset=colunas_modelo)

    if len(df_gravitacional) < 4:
        print("\nNão há dados suficientes para executar o modelo de regressão.")
        return

    # Preparar variáveis para o modelo
    X = df_gravitacional[colunas_modelo]
    y = df_gravitacional["log_Fluxo"]

    # Adicionar uma constante (intercepto) ao modelo
    X = sm.add_constant(X)

    # Criar e treinar o modelo
    modelo = sm.OLS(y, X).fit()

    # 5. Exibir os resultados
    print("\n\n--- RESULTADOS DO MODELO GRAVITACIONAL ---")
    print(modelo.summary())
    print("""
    --- INTERPRETAÇÃO DOS RESULTADOS ---
    - R-squared: Indica a proporção da variância do fluxo explicada pelo modelo.
    - Coeficientes (coef):
        - log_Massa_*: Elasticidade do fluxo em relação às diferentes massas.
        - lingua_*: Efeito específico de cada língua no fluxo de pesquisadores.
        - Similaridade_Areas: Impacto da sobreposição de áreas de pesquisa (0 a 1).
            Um valor positivo indica que maior similaridade aumenta o fluxo.
            O valor representa a mudança percentual no fluxo para cada aumento
            de 1 ponto no índice de similaridade.
        - log_Distancia: Elasticidade do fluxo em relação à distância.
    - P>|t|: Significância estatística dos coeficientes (p < 0.05 é significativo).
    """)

    # Adicionar previsões ao dataframe
    df_gravitacional["Predito_log_Fluxo"] = modelo.predict(X)
    df_gravitacional["Predito_Fluxo"] = np.exp(df_gravitacional["Predito_log_Fluxo"])
    print("\nAmostra dos dados com previsões do modelo:")
    print(df_gravitacional.head())

    # Plotar valores reais vs. previstos
    # Plotar valores reais vs. previstos
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df_gravitacional["Fluxo"],
        df_gravitacional["Predito_Fluxo"],
        c=df_gravitacional["Similaridade_Areas"],
        cmap="viridis",
        s=100,
    )
    plt.colorbar(scatter, label="Similaridade de Áreas")
    plt.plot(
        [0, df_gravitacional["Fluxo"].max()],
        [0, df_gravitacional["Fluxo"].max()],
        "r--",
        label="Linha de Perfeição (y=x)",
    )
    plt.title("Fluxo Real vs. Fluxo Predito pelo Modelo Gravitacional")
    plt.xlabel("Fluxo Real de Pesquisadores")
    plt.ylabel("Fluxo Predito de Pesquisadores")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
