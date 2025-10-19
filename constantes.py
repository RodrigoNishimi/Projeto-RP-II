ESTADOS_BRASILEIROS = [
    "ACRE",
    "ALAGOAS",
    "AMAPÁ",
    "AMAZONAS",
    "BAHIA",
    "CEARÁ",
    "DISTRITO FEDERAL",
    "ESPÍRITO SANTO",
    "GOIÁS",
    "MARANHÃO",
    "MATO GROSSO",
    "MATO GROSSO DO SUL",
    "MINAS GERAIS",
    "PARÁ",
    "PARAÍBA",
    "PARANÁ",
    "PERNAMBUCO",
    "PIAUÍ",
    "RIO DE JANEIRO",
    "RIO GRANDE DO NORTE",
    "RIO GRANDE DO SUL",
    "RONDÔNIA",
    "RORAIMA",
    "SANTA CATARINA",
    "SÃO PAULO",
    "SERGIPE",
    "TOCANTINS",
]

COLUMNS = [
    "NM_BENEFICIARIO",
    "NM_NIVEL",
    "AN_INICIO",
    "AN_FIM",
    "NM_IES_ORIGEM_PRINCIPAL_DA",
    "NM_IES_ESTUDO_PRINCIPAL_DA",
    "NM_AREA_AVALIACAO",
    "NM_UF_IES_ORIGEM",
    "NM_PAIS_IES_ESTUDO",
]

COLUMN_RENAME = [
    "nome_beneficiario",
    "nivel_bolsa",
    "ano_inicio_bolsa",
    "ano_fim_bolsa",
    "nome_ies_origem",
    "nome_ies_destino",
    "area_avaliacao",
    "uf_ies_origem",
    "pais_ies_destino",
]

DOUTOR_TERMS = [
    "DOUTORADO PLENO",
    "DOUTORADO SANDUÍCHE",
    "DOUTOR PLENO",
    "DOUTOR SÊNIOR",
    "DOUTOR JÚNIOR",
    "JOVEM DOUTOR",
    "ESTÁGIO PÓS-DOUTORAL",
    "PÓS-DOUTORADO",
    "ESTÁGIO SÊNIOR",
    "PESQUISADOR VISITANTE ESPECIAL",
    "PROFESSOR VISITANTE",
    "PROFESSOR VISITANTE SÊNIOR",
    "PROFESSOR VISITANTE JÚNIOR",
    "PROFESSOR VISITANTE DO EXTERIOR SÊNIOR",
    "PROFESSOR VISITANTE DO EXTERIOR PLENO",
    "PROFESSOR/PESQUISADOR VISITANTE NO EXTERIOR",
    "PROFESSOR CONVIDADO",
]

LINGUA_UNIVERSIDADES = {
    "ex_1": "Português",
    "ex_2": "Português",
    "ex_3": "Inglês",
    "ex_4": "Português",
    "ex_5": "Português",
    "ex_6": "Inglês",
    "ex_7": "Francês",
    "ex_8": "Português",
    "ex_9": "Inglês",
    "ex_10": "Catalão",
    "ex_11": "Inglês",
    "ex_12": "Catalão",
    "ex_13": "Português",
    "ex_14": "Espanhol",
    "ex_15": "Inglês",
    "ex_16": "Inglês",
    "ex_17": "Francês",
    "ex_18": "Inglês",
    "ex_19": "Espanhol",
    "ex_20": "Holandês",
    "ex_21": "Espanhol",
    "ex_22": "Francês",
    "ex_23": "Inglês",
    "ex_24": "Francês",
    "ex_25": "Inglês",
    "ex_26": "Francês",
    "ex_27": "Francês",
    "ex_28": "Espanhol",
    "ex_29": "Francês",
}

# Massa (pontuação) baseada no "Overall Score" do Center for World University Rankings (CWUR) 2024.
MASSA_UNIVERSIDADES_BR = {
    "br_1": 81.2,  # universidade de sao paulo
    "br_2": 74.7,  # universidade federal do rio grande do sul
    "br_3": 75.5,  # universidade federal do rio de janeiro
    "br_4": 74.4,  # universidade federal de minas gerais
    "br_5": 72.4,  # universidade federal de santa catarina
    "br_6": 75.9,  # universidade estadual de campinas
    "br_7": 71.5,  # universidade de brasilia
    "br_8": 75.0,  # universidade estadual paulista julio de mesquita filho sede
    "br_9": 68.2,  # universidade federal do parana
    "br_10": 71.2,  # universidade federal de pernambuco
    "br_11": 70.5,  # universidade federal da bahia
    "br_12": 75.0,  # universidade estadual paulista julio de mesquita filho
    "br_13": 70.7,  # universidade federal fluminense
    "br_14": 70.8,  # universidade federal do ceara
    "br_15": 70.8,  # universidade federal do rio grande do norte
    "br_16": 70.8,  # universidade federal de sao carlos
    "br_17": 68.1,  # pontificia universidade catolica do rio grande do sul
    "br_18": 71.4,  # universidade do estado do rio de janeiro
    "br_19": 70.7,  # universidade federal de vicosa
    "br_20": 69.1,  # universidade federal de lavras
    "br_21": 70.4,  # universidade federal de santa maria
    "br_22": 67.1,  # pontificia universidade catolica do rio de janeiro
    "br_23": 69.1,  # universidade federal do para
    "br_24": 70.0,  # universidade federal de goias
    "br_25": 70.7,  # universidade federal de pelotas
    "br_26": 65.0,  # pontificia universidade catolica de sao paulo (Fora do top 2000, pontuação estimada abaixo da última classificada)
    "br_27": 68.9,  # universidade estadual de maringa
    "br_28": 73.6,  # universidade federal de sao paulo
    "br_29": 69.3,  # universidade federal da paraiba
}

MASSA_UNIVERSIDADES_EX = {
    "ex_1": 77.3,  # universidade de lisboa
    "ex_2": 74.9,  # universidade de coimbra
    "ex_3": 89.8,  # university of california system (Usando UC Berkeley, a principal)
    "ex_4": 76.5,  # universidade do porto
    "ex_5": 73.4,  # universidade do minho
    "ex_6": 87.2,  # university of london (Usando University College London - UCL, a principal)
    "ex_7": 74.0,  # universite paris nord paris xiii (Atual: Université Sorbonne Paris Nord)
    "ex_8": 74.6,  # universidade nova de lisboa
    "ex_9": 82.2,  # university of florida
    "ex_10": 80.6,  # universitat de barcelona
    "ex_11": 100.0,  # harvard university (Rank #1 Global)
    "ex_12": 78.8,  # universitat autonoma de barcelona
    "ex_13": 73.1,  # universidade de aveiro
    "ex_14": 80.2,  # consejo superior de investigaciones cientificas (CSIC)
    "ex_15": 83.8,  # university of texas system (Usando UT Austin, a principal)
    "ex_16": 84.4,  # university of illinois (Usando o campus principal, Urbana-Champaign)
    "ex_17": 67.4,  # universite sorbonne nouvelle paris iii
    "ex_18": 87.5,  # university of toronto
    "ex_19": 78.0,  # universidad complutense de madrid
    "ex_20": 80.2,  # wageningen university
    "ex_21": 76.9,  # universidad de buenos aires
    "ex_22": 75.5,  # universite pantheon sorbonne paris 1
    "ex_23": 92.0,  # columbia university
    "ex_24": 80.7,  # universite de montreal
    "ex_25": 94.1,  # university of cambridge
    "ex_26": 85.1,  # universite paris diderot paris vii (Atual: Université Paris Cité)
    "ex_27": 69.7,  # universite de vincennes a saint denis paris viii
    "ex_28": 75.2,  # universidad de sevilla
    "ex_29": 71.8,  # universite paris ouest nanterre la defense paris x (Atual: Paris Nanterre University)
}
