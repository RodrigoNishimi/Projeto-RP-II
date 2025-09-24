import numpy as np
import pandas as pd
import statsmodels.api as sm
from constantes import LINGUA_UNIVERSIDADES


def main():
    mat_fluxo = pd.read_csv("./dados/mat_br_to_ex.csv", index_col=0)
    mat_distancias = pd.read_csv("./dados/matriz_distancias.csv", index_col=0)
    lingua_universidades = LINGUA_UNIVERSIDADES

    """
    Prepara os dados e executa um modelo gravitacional de mobilidade.

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

    # 3. Filtrar dados e aplicar logaritmo
    # O modelo log-linear não funciona com valores nulos ou zero.
    # Filtramos pares sem fluxo e onde a massa ou distância seja zero.
    df_gravitacional = df_gravitacional[
        (df_gravitacional["Fluxo"] > 0)
        & (df_gravitacional["Massa_Origem"] > 0)
        & (df_gravitacional["Massa_Destino"] > 0)
        & (df_gravitacional["Distancia"] > 0)
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
        ["log_Massa_Origem", "log_Massa_Destino", "log_Distancia", "Mesma_Lingua"]
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


if __name__ == "__main__":
    main()
