import re
import ast
import pandas as pd
import numpy as np
from rich import print
from collections import Counter, defaultdict
import plotly.graph_objects as go

def main():
    df_tabela_sequencias = pd.read_csv("./dados/df_tabela_sequencias.csv")
    df_tabela_sequencias["universidade_lista"] = df_tabela_sequencias[
        "universidade_lista"
    ].apply(ast.literal_eval)


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
