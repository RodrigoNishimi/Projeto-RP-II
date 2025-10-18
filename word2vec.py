import pandas as pd
import ast
from gensim.models import Word2Vec
import multiprocessing

print("Iniciando script...")

try:
    df_sequencias = pd.read_csv("./dados/df_tabela_sequencias.csv")
except FileNotFoundError:
    print("Erro: Arquivo './dados/df_tabela_sequencias.csv' não encontrado.")
    print("Por favor, execute seu script 'fluxos.py' primeiro.")
    exit()

print("Arquivo df_tabela_sequencias.csv carregado.")

df_sequencias['universidade_lista'] = df_sequencias['universidade_lista'].apply(ast.literal_eval)

corpus = []
for traj in df_sequencias['universidade_lista']:
    frase_filtrada = [codigo for codigo in traj if codigo != '0']

    if len(frase_filtrada) >= 2:
        corpus.append(frase_filtrada)

print(f"Corpus criado com {len(corpus)} trajetórias (frases).")

if not corpus:
    print("Aviso: O corpus está vazio. Encerrando.")
    exit()

print("Iniciando o treinamento do Word2Vec...")

model = Word2Vec(
    sentences=corpus,
    sg=1,
    vector_size=100,
    window=5,
    min_count=5,
    workers=multiprocessing.cpu_count()
)

print("Treinamento concluído.")

model_filename = "migracao_cientifica.word2vec"
model.save(model_filename)
print(f"Modelo salvo em: {model_filename}")


print("\n--- Explorando o Modelo (Somente Correlações Estrangeiras) ---")

codigo_alvo = 'br1'
n_top_estrangeiras = 10

try:
    todos_similares = model.wv.most_similar(codigo_alvo, topn=100)

    similares_estrangeiros = []
    for universidade, similaridade in todos_similares:
        if universidade.startswith("ex_"):
            similares_estrangeiros.append((universidade, similaridade))

    print(f"\nTop {n_top_estrangeiras} universidades ESTRANGEIRAS mais similares a '{codigo_alvo}':")

    if not similares_estrangeiros:
        print(f"  Nenhuma universidade estrangeira encontrada entre os top {100} similares.")
    else:
        for universidade, similaridade in similares_estrangeiros[:n_top_estrangeiras]:
            print(f"  {universidade}: {similaridade:.4f}")

except KeyError:
    print(f"\nNão foi possível calcular similaridade para '{codigo_alvo}' (não está no vocabulário).")


codigo_alvo_2 = 'br4'

try:
    todos_similares_2 = model.wv.most_similar(codigo_alvo_2, topn=100)

    similares_estrangeiros_2 = []
    for universidade, similaridade in todos_similares_2:
        if universidade.startswith("ex_"):
            similares_estrangeiros_2.append((universidade, similaridade))

    print(f"\nTop {n_top_estrangeiras} universidades ESTRANGEIRAS mais similares a '{codigo_alvo_2}':")

    if not similares_estrangeiros_2:
        print(f"  Nenhuma universidade estrangeira encontrada entre os top {100} similares.")
    else:
        for universidade, similaridade in similares_estrangeiros_2[:n_top_estrangeiras]:
            print(f"  {universidade}: {similaridade:.4f}")

except KeyError:
    print(f"\nNão foi possível calcular similaridade para '{codigo_alvo_2}' (não está no vocabulário).")

print("\nScript finalizado.")
