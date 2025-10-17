import pandas as pd
from geopy.distance import great_circle

from constantes import UNIS_BR, UNIS_ESTRANGEIRAS

# --- 2. Lógica para gerar a matriz ---

# Pega os nomes das universidades para usar como rótulos na matriz
nomes_br = list(UNIS_BR.keys())
nomes_estrangeiras = list(UNIS_ESTRANGEIRAS.keys())

# Cria um DataFrame vazio com os nomes corretos nas linhas e colunas
matriz_distancias = pd.DataFrame(index=nomes_br, columns=nomes_estrangeiras)

# Itera sobre cada universidade brasileira (linhas)
for nome_br, coords_br in UNIS_BR.items():
    # Itera sobre cada universidade estrangeira (colunas)
    for nome_estrangeira, coords_estrangeira in UNIS_ESTRANGEIRAS.items():
        # Calcula a distância usando a fórmula great_circle (Haversine)
        # O resultado é dado em quilômetros por padrão
        distancia = great_circle(coords_br, coords_estrangeira).kilometers

        # Preenche a célula correspondente na matriz com a distância calculada
        matriz_distancias.loc[nome_br, nome_estrangeira] = int(distancia)


# --- 3. Exibir o resultado ---

print("Matriz de Distâncias Geográficas (em quilômetros):")
print(matriz_distancias)

# Salva a matriz em um arquivo CSV
matriz_distancias.to_csv("./dados/matriz_distancias.csv")
print("\nMatriz salva na pasta dados em 'matriz_distancias.csv'")
