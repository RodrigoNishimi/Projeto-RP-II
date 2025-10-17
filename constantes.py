CAMINHO = "./dados/atuacoesDoutores.csv"

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

MASSA_UNVERSIDADES_BR = {
    "br1": 100.00,  # USP
    "br2": 95.84,  # UNICAMP
    "br3": 83.13,  # UNESP
    "br4": 88.08,  # UFRJ
    "br5": 84.43,  # UFMG
    "br6": 82.50,  # UFRGS
    "br7": 76.84,  # UFSC
    "br8": 71.93,  # UFPR
    "br9": 56.40,  # UFBA
    "br10": 60.18,  # UFPE
    "br11": 44.20,  # UFC
    "br12": 55.60,  # UFSCar
    "br13": 44.97,  # PUC-SP
    "br14": 61.27,  # PUC-Rio
    "br15": 51.01,  # FGV
    "br16": 43.83,  # ITA
    "br17": 27.97,  # UFABC
    "br18": 70.83,  # UnB
    "br19": 49.37,  # UFF
    "br20": 43.14,  # UFSM
    "br21": 51.52,  # UERJ
    "br22": 47.96,  # UFV
    "br23": 32.55,  # UFPB
    "br24": 36.37,  # UFPA
    "br25": 33.64,  # UFRN
    "br26": 41.59,  # UFG
    "br27": 72.56,  # UNIFESP
    "br28": 34.34,  # UFPel
    "br29": 25.79,  # UFES
    "br30": 46.99,  # UFLA
    "br31": 40.59,  # UEL
    "br32": 39.06,  # UFU
    "br33": 37.28,  # UEM
    "br34": 33.72,  # UFJF
    "br35": 25.21,  # PUC-Minas
    "br36": 53.68,  # PUC-RS
    "br37": 23.19,  # UFRPE
    "br38": 26.06,  # UDESC
    "br39": 28.27,  # UTFPR
    "br40": 16.27,  # UNINOVE
    "br41": 37.89,  # Unisinos
    "br42": 38.35,  # Mackenzie
    "br43": 16.94,  # UNEB
    "br44": 15.65,  # UPE
    "br45": 14.95,  # UNIP
    "br46": 12.05,  # UEMG
    "br47": 10.88,  # UNITAU
    "br48": 12.56,  # UEA
    "br49": 14.15,  # Estácio
    "br50": 9.85,  # UNEMAT
    "br51": 10.23,  # UNIMEP
    "br52": 21.61,  # UCS
    "br53": 14.88,  # UCB
    "br54": 6.75,  # ULBRA
    "br55": 11.50,  # UNIGRANRIO
    "br56": 26.85,  # UNIFOR
    "br57": 13.78,  # Anhembi Morumbi
    "br58": 11.05,  # UNIFRAN
    "br59": 13.91,  # UEPA
    "br60": 10.15,  # URI
    "br61": 5.00,  # UAB
    "br62": 10.97,  # USF
    "br63": 65.50,  # ESALQ
}


MASSA_UNIVERSIDADES_EX = {
    "ex_us_mit": 100.00,  # MIT
    "ex_us_harvard": 99.85,  # Harvard
    "ex_us_stanford": 99.80,  # Stanford
    "ex_us_berkeley": 99.55,  # UC Berkeley
    "ex_us_caltech": 99.40,  # Caltech
    "ex_uk_oxford": 99.90,  # Oxford
    "ex_uk_cambridge": 99.95,  # Cambridge
    "ex_uk_imperial": 99.75,  # Imperial
    "ex_uk_ucl": 99.25,  # UCL
    "ex_ch_eth": 99.60,  # ETH Zürich
    "ex_ch_epfl": 97.80,  # EPFL
    "ex_de_tum": 97.10,  # TUM
    "ex_ca_toronto": 98.70,  # U. Toronto
    "ex_ca_mcgill": 97.75,  # McGill
    "ex_jp_tokyo": 97.90,  # U. Tokyo
    "ex_cn_tsinghua": 98.40,  # Tsinghua
    "ex_cn_peking": 98.60,  # Peking
    "ex_fr_sorbonne": 96.50,  # Sorbonne
    "ex_pt_coimbra": 73.40,  # Coimbra
    "ex_pt_lisboa": 82.20,  # Lisboa
    "ex_pt_porto": 83.15,  # Porto
    "ex_pt_minho": 68.21,  # Minho
    "ex_pt_aveiro": 68.88,  # Aveiro
    "ex_pt_nova": 76.80,  # NOVA Lisboa
    "ex_ar_uba": 93.12,  # UBA
    "ex_es_ucm": 89.60,  # UCM
    "ex_us_florida": 91.80,  # U. Florida
    "ex_mx_unam": 92.80,  # UNAM
    "ex_es_uam": 88.70,  # UAM
    "ex_uy_udelar": 31.45,  # Udelar
    "ex_es_salamanca": 64.10,  # Salamanca
    "ex_us_michigan": 97.80,  # Michigan
    "ex_ca_ubc": 97.40,  # UBC
    "ex_us_columbia": 98.30,  # Columbia
    "ex_us_ucdavis": 89.95,  # UC Davis
    "ex_co_unal": 35.10,  # UNAL
    "ex_us_cornell": 98.65,  # Cornell
    "ex_ar_unlp": 48.75,  # UNLP
    "ex_nl_wageningen": 92.40,  # Wageningen
    "ex_us_texas": 95.80,  # UT Austin
    "ex_us_wisconsin": 92.55,  # UW Madison
    "ex_es_sevilla": 66.85,  # Sevilla
    "ex_co_antioquia": 30.25,  # Antioquia
    "ex_us_nyu": 97.10,  # NYU
    "ex_es_usc": 59.90,  # USC
    "ex_us_yale": 98.90,  # Yale
    "ex_ca_alberta": 92.60,  # U. Alberta
    "ex_us_ucsd": 96.80,  # UCSD
    "ex_es_granada": 75.20,  # Granada
    "ex_us_ohiost": 88.15,  # Ohio State
    "ex_us_jhu": 98.20,  # Johns Hopkins
    "ex_ca_ottawa": 81.30,  # Ottawa
    "ex_dk_copenhagen": 95.70,  # Copenhagen
    "ex_ar_unc": 45.90,  # UNC (Córdoba)
    "ex_us_penn": 99.10,  # UPenn
    "ex_au_sydney": 98.50,  # Sydney
    "ex_pe_sanmarcos": 32.50,  # San Marcos
    "ex_dk_aarhus": 92.85,  # Aarhus
    "ex_uk_manchester": 97.50,  # Manchester
    "ex_us_msu": 86.80,  # MSU
    "ex_us_ucsystem": 99.00,  # UC System
    "ex_us_ucla": 98.10,  # UCLA
    "ex_be_ghent": 91.40,  # Ghent
    "ex_us_duke": 96.70,  # Duke
    "ex_au_uq": 96.85,  # UQ
    "ex_nl_leiden": 93.90,  # Leiden
    "ex_us_brown": 95.10,  # Brown
    "ex_us_uiuc": 94.75,  # UIUC
    "ex_nl_delft": 96.40,  # TU Delft
    "ex_uk_nottingham": 90.15,  # Nottingham
    "ex_uk_leeds": 94.90,  # Leeds
    "ex_uk_bristol": 95.55,  # Bristol
    "ex_co_valle": 25.40,  # Univalle
    "ex_uk_london": 98.80,  # U. London
    "ex_es_zaragoza": 69.15,  # Zaragoza
    "ex_us_chicago": 99.20,  # UChicago
    "ex_us_purdue": 93.30,  # Purdue
    "ex_us_arizona": 89.10,  # Arizona
    "ex_us_ncsu": 80.80,  # NCSU
    "ex_uk_birmingham": 93.50,  # Birmingham
    "ex_se_lund": 94.80,  # Lund
    "ex_ar_unr": 28.10,  # UNR
    "ex_us_princeton": 99.30,  # Princeton
    "ex_cl_uchile": 91.10,  # U. Chile
    "ex_ca_waterloo": 90.50,  # Waterloo
    "ex_se_uppsala": 92.70,  # Uppsala
    "ex_us_georgia": 78.55,  # UGA
    "ex_es_upm": 79.90,  # UPM
    "ex_es_cadiz": 34.60,  # Cádiz
    "ex_us_pitt": 87.90,  # Pitt
    "ex_uk_edinburgh": 98.25,  # Edinburgh
    "ex_co_unal_all": 35.10,  # UNAL
}

LINGUA_UNIVERSIDADES = {
    "ex_us_mit": "Inglês",
    "ex_us_harvard": "Inglês",
    "ex_us_stanford": "Inglês",
    "ex_us_berkeley": "Inglês",
    "ex_us_caltech": "Inglês",
    "ex_uk_oxford": "Inglês",
    "ex_uk_cambridge": "Inglês",
    "ex_uk_imperial": "Inglês",
    "ex_uk_ucl": "Inglês",
    "ex_ch_eth": "Inglês",
    "ex_ch_epfl": "Inglês",
    "ex_de_tum": "Inglês",
    "ex_ca_toronto": "Inglês",
    "ex_ca_mcgill": "Inglês",
    "ex_jp_tokyo": "Japonês",
    "ex_cn_tsinghua": "Mandarim",
    "ex_cn_peking": "Mandarim",
    "ex_fr_sorbonne": "Francês",
    "ex_pt_coimbra": "Português",
    "ex_pt_lisboa": "Português",
    "ex_pt_porto": "Português",
    "ex_pt_minho": "Português",
    "ex_pt_aveiro": "Português",
    "ex_pt_nova": "Português",
    "ex_ar_uba": "Espanhol",
    "ex_es_ucm": "Espanhol",
    "ex_us_florida": "Inglês",
    "ex_mx_unam": "Espanhol",
    "ex_es_uam": "Espanhol",
    "ex_uy_udelar": "Espanhol",
    "ex_es_salamanca": "Espanhol",
    "ex_us_michigan": "Inglês",
    "ex_ca_ubc": "Inglês",
    "ex_us_columbia": "Inglês",
    "ex_us_ucdavis": "Inglês",
    "ex_co_unal": "Espanhol",
    "ex_us_cornell": "Inglês",
    "ex_ar_unlp": "Espanhol",
    "ex_nl_wageningen": "Inglês",
    "ex_us_texas": "Inglês",
    "ex_us_wisconsin": "Inglês",
    "ex_es_sevilla": "Espanhol",
    "ex_co_antioquia": "Espanhol",
    "ex_us_nyu": "Inglês",
    "ex_es_usc": "Espanhol",
    "ex_us_yale": "Inglês",
    "ex_ca_alberta": "Inglês",
    "ex_us_ucsd": "Inglês",
    "ex_es_granada": "Espanhol",
    "ex_us_ohiost": "Inglês",
    "ex_us_jhu": "Inglês",
    "ex_ca_ottawa": "Inglês",
    "ex_dk_copenhagen": "Inglês",
    "ex_ar_unc": "Espanhol",
    "ex_us_penn": "Inglês",
    "ex_au_sydney": "Inglês",
    "ex_pe_sanmarcos": "Espanhol",
    "ex_dk_aarhus": "Inglês",
    "ex_uk_manchester": "Inglês",
    "ex_us_msu": "Inglês",
    "ex_us_ucsystem": "Inglês",
    "ex_us_ucla": "Inglês",
    "ex_be_ghent": "Inglês",
    "ex_us_duke": "Inglês",
    "ex_au_uq": "Inglês",
    "ex_nl_leiden": "Inglês",
    "ex_us_brown": "Inglês",
    "ex_us_uiuc": "Inglês",
    "ex_nl_delft": "Inglês",
    "ex_uk_nottingham": "Inglês",
    "ex_uk_leeds": "Inglês",
    "ex_uk_bristol": "Inglês",
    "ex_co_valle": "Espanhol",
    "ex_uk_london": "Inglês",
    "ex_es_zaragoza": "Espanhol",
    "ex_us_chicago": "Inglês",
    "ex_us_purdue": "Inglês",
    "ex_us_arizona": "Inglês",
    "ex_us_ncsu": "Inglês",
    "ex_uk_birmingham": "Inglês",
    "ex_se_lund": "Inglês",
    "ex_ar_unr": "Espanhol",
    "ex_us_princeton": "Inglês",
    "ex_cl_uchile": "Espanhol",
    "ex_ca_waterloo": "Inglês",
    "ex_se_uppsala": "Inglês",
    "ex_us_georgia": "Inglês",
    "ex_es_upm": "Espanhol",
    "ex_es_cadiz": "Espanhol",
    "ex_us_pitt": "Inglês",
    "ex_uk_edinburgh": "Inglês",
    "ex_co_unal_all": "Espanhol",
}

UNIS_BR = {
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
UNIS_ESTRANGEIRAS = {
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
