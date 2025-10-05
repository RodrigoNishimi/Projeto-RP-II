import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from constantes import LINGUA_UNIVERSIDADES
from constantes import MASSA_UNVERSIDADES_BR
from constantes import MASSA_UNIVERSIDADES_EX


def main():
    mat_fluxo = pd.read_csv("./dados/mat_br_to_ex.csv", index_col=0)
    mat_distancias = pd.read_csv("./dados/matriz_distancias.csv", index_col=0)
    lingua_universidades = LINGUA_UNIVERSIDADES

    """
    Args:
        mat_fluxo (pd.DataFrame): Matriz com fluxos de pesquisadores (origem BR, destino EX).
        mat_distancias (pd.DataFrame): Matriz com as distâncias entre as universidades.
        lingua_universidades (pd.Series): Série com a língua principal de cada universidade.
    """

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
        MASSA_UNVERSIDADES_BR
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

    # Aplicar logaritmo natural. Usamos np.log1p (log(1+x)) para estabilidade,
    # mas como já filtramos os zeros, np.log também funcionaria.
    df_gravitacional["log_Fluxo"] = np.log(df_gravitacional["Fluxo"].astype(float))
    df_gravitacional["log_Massa_Origem"] = np.log(
        df_gravitacional["Massa_Origem"].astype(float)
    )
    df_gravitacional["log_Massa_Destino"] = np.log(
        df_gravitacional["Massa_Destino"].astype(float)
    )
    df_gravitacional["log_Distancia"] = np.log(
        df_gravitacional["Distancia"].astype(float)
    )
    df_gravitacional["log_Massa_BR"] = np.log(
        df_gravitacional["Massa_BR"].astype(float)
    )
    df_gravitacional["log_Massa_EX"] = np.log(
        df_gravitacional["Massa_EX"].astype(float)
    )

    print(
        f"\nNúmero de observações (pares universidade) para o modelo: {len(df_gravitacional)}"
    )
    print("Amostra dos dados preparados para regressão:")
    print(df_gravitacional.head())

    # Adicionar a variável de língua
    # Para universidades brasileiras (origem) a língua é sempre "Português"
    # Para universidades estrangeiras (destino) usamos o dicionário LINGUA_UNIVERSIDADES
    df_gravitacional["Mesma_Lingua"] = (
        df_gravitacional["EX_Destino"].map(lingua_universidades) == "Português"
    ).astype(int)

    # 4. Executar o Modelo de Regressão Linear (OLS)
    if len(df_gravitacional) < 4:
        print("\nNão há dados suficientes para executar o modelo de regressão.")
        return

    # Variáveis independentes (X) e dependente (y)
    X = df_gravitacional[
        [
            "log_Massa_Origem",
            "log_Massa_Destino",
            "log_Massa_BR",
            "log_Massa_EX",
            "Mesma_Lingua",
            "log_Distancia",
        ]
    ]
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
    - R-squared: Indica a proporção da variância do fluxo de pesquisadores que é explicada pelas massas e distância.
    - Coeficientes (coef):
        - log_Massa_Origem (alpha): Elasticidade do fluxo em relação à massa de origem. Um aumento de 1% na massa de origem leva a uma mudança de ~alpha% no fluxo.
        - log_Massa_Destino (beta): Elasticidade do fluxo em relação à massa de destino.
        - log_Distancia (gamma): Elasticidade do fluxo em relação à distância. Espera-se um valor negativo, indicando que o fluxo diminui com a distância.
        - Mesma_Lingua (delta): Impacto de ter a mesma língua principal nas universidades de origem e destino. Um coeficiente positivo indica que a mesma língua aumenta o fluxo.
    - P>|t|: O p-valor. Se for baixo (ex: < 0.05), o coeficiente é estatisticamente significativo.
    """)

    # Adicionar previsões ao dataframe
    df_gravitacional["Predito_log_Fluxo"] = modelo.predict(X)
    df_gravitacional["Predito_Fluxo"] = np.exp(df_gravitacional["Predito_log_Fluxo"])
    print("\nAmostra dos dados com previsões do modelo:")
    print(df_gravitacional.head())

    # Plotar valores reais vs. previstos
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="Fluxo", y="Predito_Fluxo", data=df_gravitacional, hue="Mesma_Lingua", s=100
    )
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
