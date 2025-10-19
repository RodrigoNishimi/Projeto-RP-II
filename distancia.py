import pandas as pd
from geopy.distance import great_circle


unis_br = {
    "br_1": (-23.5587, -46.7319),  # universidade de sao paulo
    "br_2": (-30.0335, -51.2177),  # universidade federal do rio grande do sul
    "br_3": (-22.8571, -43.2338),  # universidade federal do rio de janeiro
    "br_4": (-19.8696, -43.9648),  # universidade federal de minas gerais
    "br_5": (-27.6007, -48.5188),  # universidade federal de santa catarina
    "br_6": (-22.8176, -47.0696),  # universidade estadual de campinas
    "br_7": (-15.7634, -47.8722),  # universidade de brasilia
    "br_8": (-22.3485, -49.0307),  # unesp Bauru
    "br_9": (-25.4284, -49.2636),  # universidade federal do parana
    "br_10": (-8.0535, -34.9507),  # universidade federal de pernambuco
    "br_11": (-13.0076, -38.5126),  # universidade federal da bahia
    "br_12": (-22.890137, -48.497053),  # unesp Botucatu
    "br_13": (-22.9035, -43.1293),  # universidade federal fluminense
    "br_14": (-3.7461, -38.5746),  # universidade federal do ceara
    "br_15": (-5.8398, -35.2023),  # universidade federal do rio grande do norte
    "br_16": (-22.0016, -47.8817),  # universidade federal de sao carlos
    "br_17": (-30.0573, -51.1731),  # pontificia universidade catolica do RS
    "br_18": (-22.9198, -43.2307),  # universidade do estado do rio de janeiro
    "br_19": (-20.7601, -42.8687),  # universidade federal de vicosa
    "br_20": (-21.2335, -44.9798),  # universidade federal de lavras
    "br_21": (-29.7188, -53.7149),  # universidade federal de santa maria
    "br_22": (-22.9772, -43.2562),  # pontificia universidade catolica do rio de janeiro
    "br_23": (-1.4746, -48.4556),  # universidade federal do para
    "br_24": (-16.6083, -49.2319),  # universidade federal de goias
    "br_25": (-31.7709, -52.3392),  # universidade federal de pelotas (Reitoria)
    "br_26": (-23.5358, -46.6738),  # pontificia universidade catolica de sao paulo
    "br_27": (-23.4093, -51.9382),  # universidade estadual de maringa
    "br_28": (-23.5982, -46.6436),  # universidade federal de sao paulo
    "br_29": (-7.1384, -34.8455),  # universidade federal da paraiba
}

# EX (campus principal)
unis_estrangeiras = {
    "ex_1": (38.7525, -9.1589),
    "ex_2": (40.2080, -8.4241),
    "ex_3": (37.8022, -122.2714),
    "ex_4": (41.1465, -8.6157),
    "ex_5": (41.5508, -8.4263),
    "ex_6": (51.5211, -0.1289),
    "ex_7": (48.9141, 2.4185),
    "ex_8": (38.7339, -9.1602),
    "ex_9": (29.6436, -82.3549),
    "ex_10": (41.3867, 2.1639),
    "ex_11": (42.3744, -71.1169),
    "ex_12": (41.5006, 2.1051),
    "ex_13": (40.6297, -8.6579),
    "ex_14": (40.4411, -3.6864),
    "ex_15": (30.2742, -97.7400),
    "ex_16": (40.1084, -88.2277),
    "ex_17": (48.8450, 2.3969),
    "ex_18": (43.6635, -79.3958),
    "ex_19": (40.4490, -3.7270),
    "ex_20": (51.9644, 5.6631),
    "ex_21": (-34.5997, -58.3731),
    "ex_22": (48.8470, 2.3440),
    "ex_23": (40.8075, -73.9619),
    "ex_24": (45.5041, -73.6143),
    "ex_25": (52.2053, 0.1131),
    "ex_26": (48.8297, 2.3808),
    "ex_27": (48.9447, 2.3633),
    "ex_28": (37.3808, -5.9912),
    "ex_29": (48.9042, 2.2140),
}

# --- 2. Lógica para gerar a matriz ---

# Pega os nomes das universidades para usar como rótulos na matriz
nomes_br = list(unis_br.keys())
nomes_estrangeiras = list(unis_estrangeiras.keys())

# Cria um DataFrame vazio com os nomes corretos nas linhas e colunas
matriz_distancias = pd.DataFrame(index=nomes_br, columns=nomes_estrangeiras)

# Itera sobre cada universidade brasileira (linhas)
for nome_br, coords_br in unis_br.items():
    # Itera sobre cada universidade estrangeira (colunas)
    for nome_estrangeira, coords_estrangeira in unis_estrangeiras.items():
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
