import pandas as pd
from geopy.distance import great_circle


unis_br = {
    "br1": (-23.5617, -46.7308),  # USP, São Paulo (Cidade Universitária)
    "br2": (-22.8170, -47.0698),  # UNICAMP, Campinas
    "br3": (-22.3246, -49.0286),  # UNESP, Botucatu (aprox.)
    "br4": (-22.8610, -43.2330),  # UFRJ, Ilha do Fundão, Rio
    "br5": (-19.8700, -43.9670),  # UFMG, Belo Horizonte
    "br6": (-30.0331, -51.2196),  # UFRGS, Porto Alegre (aprox. central)
    "br7": (-27.6008, -48.5190),  # UFSC, Florianópolis
    "br8": (-25.4284, -49.2733),  # UFPR, Curitiba (aprox. central)
    "br9": (-12.9980, -38.5070),  # UFBA, Salvador (Ondina aprox.)
    "br10": (-8.0557, -34.9510),  # UFPE, Recife
    "br11": (-3.7450, -38.5740),  # UFC, Fortaleza (Pici aprox.)
    "br12": (-21.9810, -47.8800),  # UFSCar, São Carlos
    "br13": (-23.5390, -46.6720),  # PUC-SP, Perdizes (aprox.)
    "br14": (-22.9790, -43.2320),  # PUC-Rio, Gávea
    "br15": (-22.9100, -43.1730),  # FGV-Rio (aprox. Praia de Botafogo)
    "br16": (-23.2156, -45.8612),  # ITA, São José dos Campos
    "br17": (-23.6460, -46.5320),  # UFABC, Santo André
    "br18": (-15.7646, -47.8705),  # UnB, Brasília
    "br19": (-22.9180, -43.1290),  # UFF, Niterói
    "br20": (-29.7174, -53.7169),  # UFSM, Santa Maria
    "br21": (-22.9090, -43.2290),  # UERJ, Rio
    "br22": (-20.7610, -42.8690),  # UFV, Viçosa
    "br23": (-7.1357, -34.8450),  # UFPB, João Pessoa
    "br24": (-1.4748, -48.4520),  # UFPA, Belém
    "br25": (-5.8390, -35.1990),  # UFRN, Natal
    "br26": (-16.6040, -49.2660),  # UFG, Goiânia
    "br27": (-23.6515, -46.5733),  # UNIFESP, SP (campus São Paulo)
    "br28": (-31.7710, -52.3420),  # UFPel, Pelotas
    "br29": (-20.2770, -40.3030),  # UFES, Vitória
    "br30": (-21.2260, -44.9840),  # UFLA, Lavras
    "br31": (-23.3100, -51.1620),  # UEL, Londrina
    "br32": (-18.9180, -48.2570),  # UFU, Uberlândia
    "br33": (-23.4050, -51.9410),  # UEM, Maringá
    "br34": (-21.7700, -43.3670),  # UFJF, Juiz de Fora
    "br35": (-19.9220, -43.9900),  # PUC Minas, BH
    "br36": (-30.0590, -51.1720),  # PUCRS, Porto Alegre
    "br37": (-8.0170, -34.9470),  # UFRPE, Recife
    "br38": (-27.6019, -48.5197),  # UDESC, Florianópolis
    "br39": (-25.4515, -49.2310),  # UTFPR, Curitiba
    "br40": (-23.5382, -46.7356),  # UNINOVE, São Paulo
    "br41": (-29.7915, -51.1506),  # Unisinos, São Leopoldo
    "br42": (-23.5498, -46.6523),  # Mackenzie, Higienópolis, SP
    "br43": (-12.9426, -38.4804),  # UNEB, Salvador
    "br44": (-8.0476, -34.8989),  # UPE, Recife
    "br45": (-23.5366, -46.5617),  # UNIP, São Paulo
    "br46": (-19.9201, -43.9346),  # UEMG, BH (sede)
    "br47": (-23.0246, -45.5593),  # UNITAU, Taubaté
    "br48": (-3.0944, -60.0170),  # UEA, Manaus
    "br49": (-22.9129, -43.2302),  # Estácio, Rio
    "br50": (-15.6324, -56.0921),  # UNEMAT, Cáceres
    "br51": (-22.7340, -47.6483),  # UNIMEP, Piracicaba
    "br52": (-29.1669, -51.1790),  # UCS, Caxias do Sul
    "br53": (-15.8631, -47.9137),  # UCB, Brasília
    "br54": (-29.6905, -51.1322),  # ULBRA, Canoas
    "br55": (-22.8053, -43.2065),  # UNIGRANRIO, Duque de Caxias
    "br56": (-3.7710, -38.4811),  # UNIFOR, Fortaleza
    "br57": (-23.5975, -46.6359),  # Anhembi Morumbi, SP
    "br58": (-20.5382, -47.4009),  # UNIFRAN, Franca
    "br59": (-1.4550, -48.5022),  # UEPA, Belém
    "br60": (-27.6417, -54.2639),  # URI, Santo Ângelo
    "br61": (-15.7753, -47.9440),  # UAB, Brasília (coordenação central)
    "br62": (-22.8540, -46.3190),  # USF, Bragança Paulista
    "br63": (-22.7058, -47.6357),  # ESALQ/USP, Piracicaba
}

# EX (campus principal)
unis_estrangeiras = {
    "ex_us_mit": (42.3601, -71.0942),  # Cambridge, MA
    "ex_us_harvard": (42.3770, -71.1167),  # Cambridge, MA
    "ex_us_stanford": (37.4275, -122.1697),  # Stanford, CA
    "ex_us_berkeley": (37.8719, -122.2585),  # Berkeley, CA
    "ex_us_caltech": (34.1377, -118.1253),  # Pasadena, CA
    "ex_uk_oxford": (51.7548, -1.2544),  # Oxford
    "ex_uk_cambridge": (52.2043, 0.1149),  # Cambridge
    "ex_uk_imperial": (51.4988, -0.1749),  # London (South Kensington)
    "ex_uk_ucl": (51.5246, -0.1340),  # London (Bloomsbury)
    "ex_ch_eth": (47.3763, 8.5476),  # Zürich
    "ex_ch_epfl": (46.5191, 6.5668),  # Lausanne
    "ex_de_tum": (48.2620, 11.6670),  # TUM Garching (aprox.)
    "ex_ca_toronto": (43.6629, -79.3957),  # Toronto
    "ex_ca_mcgill": (45.5048, -73.5772),  # Montreal
    "ex_jp_tokyo": (35.7126, 139.7610),  # Tóquio (Hongo)
    "ex_cn_tsinghua": (40.0030, 116.3260),  # Pequim
    "ex_cn_peking": (39.9869, 116.3055),  # Pequim
    "ex_fr_sorbonne": (48.8470, 2.3440),  # Paris
    "ex_jp_kyoto": (35.0260, 135.7808),
    "ex_jp_osaka": (34.8225, 135.5231),
    "ex_jp_tohoku": (38.2554, 140.8520),
    "ex_cn_shanghai": (31.1976, 121.4326),
    "ex_cn_fudan": (31.2990, 121.5037),
    "ex_cn_cas": (39.9042, 116.4074),
    "ex_pt_coimbra": (40.2070, -8.4250),  # Universidade de Coimbra
    "ex_pt_lisboa": (
        38.7527,
        -9.1567,
    ),  # Universidade de Lisboa (Cidade Universitária)
    "ex_pt_porto": (41.1780, -8.5970),  # Universidade do Porto
    "ex_pt_minho": (41.5610, -8.3960),  # Universidade do Minho, Braga
    "ex_pt_aveiro": (40.6300, -8.6570),  # Universidade de Aveiro
    "ex_pt_nova": (38.7370, -9.1540),  # Universidade NOVA de Lisboa (Campolide)
    "ex_ar_uba": (-34.5990, -58.3730),  # UBA, Buenos Aires
    "ex_es_ucm": (40.4520, -3.7280),  # UCM, Madrid
    "ex_us_florida": (29.6516, -82.3248),  # University of Florida, Gainesville
    "ex_mx_unam": (19.3322, -99.1860),  # Cidade do México
    "ex_es_uam": (40.5439, -3.6974),  # Madrid
    "ex_uy_udelar": (-34.9011, -56.1645),  # Montevidéu
    "ex_es_salamanca": (40.9629, -5.6689),  # Salamanca
    "ex_us_michigan": (42.2780, -83.7382),  # Ann Arbor, MI
    "ex_ca_ubc": (49.2606, -123.2460),  # Vancouver
    "ex_us_columbia": (40.8075, -73.9626),  # NYC
    "ex_us_ucdavis": (38.5382, -121.7617),  # Davis, CA
    "ex_co_unal": (4.6386, -74.0841),  # Bogotá
    "ex_us_cornell": (42.4534, -76.4735),  # Ithaca, NY
    "ex_ar_unlp": (-34.9092, -57.9463),  # La Plata
    "ex_nl_wageningen": (51.9851, 5.6630),  # Wageningen
    "ex_us_texas": (30.2849, -97.7341),  # Austin, TX
    "ex_us_wisconsin": (43.0766, -89.4125),  # Madison, WI
    "ex_es_sevilla": (37.3772, -5.9869),  # Sevilla
    "ex_co_antioquia": (6.2675, -75.5680),  # Medellín
    "ex_us_nyu": (40.7295, -73.9965),  # NYC
    "ex_es_usc": (42.8770, -8.5540),  # Santiago de Compostela
    "ex_us_yale": (41.3163, -72.9223),  # New Haven, CT
    "ex_ca_alberta": (53.5232, -113.5263),  # Edmonton
    "ex_us_ucsd": (32.8801, -117.2340),  # San Diego, CA
    "ex_es_granada": (37.1847, -3.6056),  # Granada
    "ex_us_ohiost": (40.0076, -83.0300),  # Columbus, OH
    "ex_us_jhu": (39.3299, -76.6205),  # Baltimore, MD
    "ex_ca_ottawa": (45.4231, -75.6831),  # Ottawa
    "ex_dk_copenhagen": (55.6805, 12.5714),  # Copenhagen
    "ex_ar_unc": (-31.4420, -64.1930),  # Córdoba
    "ex_us_penn": (39.9522, -75.1932),  # Philadelphia, PA
    "ex_au_sydney": (-33.8880, 151.1875),  # Sydney
    "ex_pe_sanmarcos": (-12.0563, -77.0844),  # Lima
    "ex_dk_aarhus": (56.1681, 10.2027),  # Aarhus
    "ex_uk_manchester": (53.4670, -2.2330),  # Manchester
    "ex_us_msu": (42.7018, -84.4822),  # East Lansing, MI
    "ex_us_ucsystem": (37.8719, -122.2585),  # Referência Berkeley
    "ex_us_ucla": (34.0689, -118.4452),  # Los Angeles, CA
    "ex_be_ghent": (51.0470, 3.7270),  # Ghent
    "ex_us_duke": (36.0014, -78.9382),  # Durham, NC
    "ex_au_uq": (-27.4975, 153.0137),  # Brisbane
    "ex_nl_leiden": (52.1601, 4.4970),  # Leiden
    "ex_us_brown": (41.8268, -71.4025),  # Providence, RI
    "ex_us_uiuc": (40.1020, -88.2272),  # Urbana-Champaign, IL
    "ex_nl_delft": (52.0020, 4.3700),  # Delft
    "ex_uk_nottingham": (52.9380, -1.1940),  # Nottingham
    "ex_uk_leeds": (53.8067, -1.5550),  # Leeds
    "ex_uk_bristol": (51.4584, -2.6030),  # Bristol
    "ex_co_valle": (3.3752, -76.5340),  # Cali
    "ex_uk_london": (51.5072, -0.1276),  # Londres
    "ex_es_zaragoza": (41.6488, -0.8891),  # Zaragoza
    "ex_us_chicago": (41.7897, -87.5997),  # Chicago, IL
    "ex_us_purdue": (40.4237, -86.9212),  # West Lafayette, IN
    "ex_us_arizona": (32.2319, -110.9501),  # Tucson, AZ
    "ex_us_ncsu": (35.7847, -78.6821),  # Raleigh, NC
    "ex_uk_birmingham": (52.4508, -1.9305),  # Birmingham
    "ex_se_lund": (55.7058, 13.1932),  # Lund
    "ex_ar_unr": (-32.9525, -60.6394),  # Rosario
    "ex_us_princeton": (40.3431, -74.6551),  # Princeton, NJ
    "ex_cl_uchile": (-33.4489, -70.6633),  # Santiago
    "ex_ca_waterloo": (43.4723, -80.5449),  # Waterloo
    "ex_se_uppsala": (59.8586, 17.6389),  # Uppsala
    "ex_us_georgia": (33.9480, -83.3773),  # Athens, GA
    "ex_es_upm": (40.4418, -3.7294),  # Madrid
    "ex_es_cadiz": (36.5297, -6.2927),  # Cádiz
    "ex_us_pitt": (40.4444, -79.9608),  # Pittsburgh, PA
    "ex_uk_edinburgh": (55.9444, -3.1888),  # Edinburgh
    "ex_co_unal_all": (4.6386, -74.0841),  # Bogotá (sede principal)
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
