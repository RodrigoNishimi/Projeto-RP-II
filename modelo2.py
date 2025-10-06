import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from constantes import LINGUA_UNIVERSIDADES
from constantes import MASSA_UNVERSIDADES_BR
from constantes import MASSA_UNIVERSIDADES_EX

# Suas funções auxiliares (criar_dummies_lingua, processar_areas, calcular_similaridade_areas)
# permanecem as mesmas, pois já estão bem implementadas.
# ... (cole suas funções auxiliares aqui) ...


def criar_dummies_lingua(df):
    """Cria variáveis dummy para cada língua"""
    linguas_destino = df["EX_Destino"].map(LINGUA_UNIVERSIDADES).fillna("Desconhecido")
    # Limpando nomes de colunas para serem compatíveis com a fórmula do statsmodels
    linguas_destino.name = "lingua"
    dummies_lingua = pd.get_dummies(linguas_destino, prefix="lingua").astype(float)
    # Evitar caracteres especiais nos nomes das colunas
    dummies_lingua.columns = [
        col.replace(" ", "_").replace("-", "_") for col in dummies_lingua.columns
    ]
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

    intersecao = len(areas_origem.intersection(areas_destino))
    uniao = len(areas_origem.union(areas_destino))

    return float(intersecao) / float(uniao) if uniao > 0 else 0.0


def main_aprimorado():
    # --- 1. Carregamento e Preparação dos Dados ---
    mat_fluxo = pd.read_csv("./dados/mat_br_to_ex.csv", index_col=0)
    mat_distancias = pd.read_csv("./dados/matriz_distancias.csv", index_col=0)
    dataset_capes = pd.read_csv("./dados/br-capes-filtrado.csv", index_col=0)

    fluxo_long = mat_fluxo.stack().reset_index()
    fluxo_long.columns = ["BR_Origem", "EX_Destino", "Fluxo"]

    dist_long = mat_distancias.stack().reset_index()
    dist_long.columns = ["BR_Origem", "EX_Destino", "Distancia"]

    df_gravitacional = pd.merge(fluxo_long, dist_long, on=["BR_Origem", "EX_Destino"])

    # Adicionar massas de prestígio
    df_gravitacional["Massa_BR"] = df_gravitacional["BR_Origem"].map(
        MASSA_UNVERSIDADES_BR
    )
    df_gravitacional["Massa_EX"] = df_gravitacional["EX_Destino"].map(
        MASSA_UNIVERSIDADES_EX
    )

    # --- 2. Engenharia de Variáveis ---
    # Adicionar dummies de lingua
    df_gravitacional = criar_dummies_lingua(df_gravitacional)

    # Adicionar similaridade de áreas
    areas_por_univ = processar_areas(dataset_capes)
    df_gravitacional["Similaridade_Areas"] = df_gravitacional.apply(
        lambda row: calcular_similaridade_areas(row, areas_por_univ), axis=1
    )

    # Aplicar logaritmo APENAS às variáveis independentes contínuas.
    # Adicionamos 1 para evitar log(0) em distância ou massa, caso existam.
    df_gravitacional["log_Distancia"] = np.log(df_gravitacional["Distancia"] + 1)
    df_gravitacional["log_Massa_BR"] = np.log(df_gravitacional["Massa_BR"] + 1)
    df_gravitacional["log_Massa_EX"] = np.log(df_gravitacional["Massa_EX"] + 1)

    # **MUDANÇA IMPORTANTE**: Não filtramos mais o fluxo == 0.
    # O modelo de Poisson utiliza todas as observações.
    df_gravitacional = (
        df_gravitacional.dropna()
    )  # Remove NaNs de massas ou outras colunas

    print(
        f"\nNúmero total de observações para o modelo de Poisson: {len(df_gravitacional)}"
    )
    print("Amostra dos dados preparados para regressão:")
    print(df_gravitacional.head())

    # --- 3. Construção e Treinamento do Modelo de Poisson ---

    # Construir a fórmula do modelo dinamicamente
    colunas_lingua = [
        col
        for col in df_gravitacional.columns
        if col.startswith("lingua_") and col != "lingua_Desconhecido"
    ]

    # A fórmula facilita a especificação do modelo
    # A variável dependente 'Fluxo' NÃO é logaritmizada
    variaveis_independentes = [
        "log_Massa_BR",
        "log_Massa_EX",
        "log_Distancia",
        "Similaridade_Areas",
    ] + colunas_lingua

    formula = f"Fluxo ~ {' + '.join(variaveis_independentes)}"
    print(f"\nFórmula do modelo:\n{formula}")

    # Treinar o modelo de Poisson
    # Adicionar tratamento para possíveis erros de convergência
    try:
        modelo_poisson = smf.poisson(formula, data=df_gravitacional).fit()
    except Exception as e:
        print(f"Ocorreu um erro ao treinar o modelo: {e}")
        # Tentar com um otimizador diferente se houver problemas de convergência
        try:
            modelo_poisson = smf.poisson(formula, data=df_gravitacional).fit(
                method="newton"
            )
        except Exception as e_newton:
            print(f"Falha também com o método Newton: {e_newton}")
            return

    # --- 4. Exibir e Interpretar os Resultados ---
    print("\n\n--- RESULTADOS DO MODELO GRAVITACIONAL DE POISSON ---")
    print(modelo_poisson.summary())
    print("""
    --- INTERPRETAÇÃO DOS RESULTADOS (MODELO DE POISSON) ---
    - Pseudo R-squ.: Análogo ao R-squared, indica o ajuste do modelo.
    - Coeficientes (coef):
        - Representam a mudança no LOG do fluxo esperado.
        - Para interpretar: calcule exp(coef). Um valor de exp(coef) = 1.10 para 'log_Massa_BR'
          significa que um aumento de 1% na massa aumenta o fluxo esperado em aproximadamente 10%.
          Um valor de exp(coef) = 0.95 para 'log_Distancia' significa que um aumento de 1%
          na distância diminui o fluxo em 5%.
    - P>|z|: Significância estatística dos coeficientes (p < 0.05 é significativo).
    """)

    # Exponencial dos coeficientes para facilitar a interpretação
    print("\nCoeficientes Exponenciais (exp(coef)):")
    print(np.exp(modelo_poisson.params))

    # --- 5. Visualização dos Resultados ---
    df_gravitacional["Predito_Fluxo"] = modelo_poisson.predict()

    plt.figure(figsize=(12, 8))
    # Para melhor visualização com muitos zeros, focamos nos fluxos observados
    subset_plot = df_gravitacional[df_gravitacional["Fluxo"] > 0]

    scatter = plt.scatter(
        subset_plot["Fluxo"],
        subset_plot["Predito_Fluxo"],
        c=subset_plot["Similaridade_Areas"],
        cmap="viridis",
        alpha=0.6,
        s=80,
    )
    plt.colorbar(scatter, label="Similaridade de Áreas")

    max_val = max(subset_plot["Fluxo"].max(), subset_plot["Predito_Fluxo"].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Linha de Perfeição (y=x)")

    plt.title("Fluxo Real vs. Fluxo Predito (Modelo de Poisson)")
    plt.xlabel("Fluxo Real de Pesquisadores (Observado > 0)")
    plt.ylabel("Fluxo Predito de Pesquisadores")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main_aprimorado()
