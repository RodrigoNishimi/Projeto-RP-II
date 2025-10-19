import pandas as pd
from gensim.models import Word2Vec

MODEL_FILENAME = "word2vec_br_ex_model.model"
MAP_FILENAME = "dados/mapa_universidades.csv"
TOP_N = 10

def load_resources():
    """Carrega o modelo W2V e o mapa de nomes de universidades."""
    try:
        model = Word2Vec.load(MODEL_FILENAME)
        print(f"Modelo '{MODEL_FILENAME}' carregado.")
    except FileNotFoundError:
        print(f"Erro: Modelo '{MODEL_FILENAME}' não encontrado.")
        print("Por favor, execute 'word2vec.py' primeiro para treinar e salvar o modelo.")
        return None, None

    try:
        df_mapa = pd.read_csv(MAP_FILENAME, index_col=0)
        mapa_nomes = pd.Series(df_mapa.nome_universidade.values, index=df_mapa.codigo).to_dict()
        print(f"Mapa de universidades '{MAP_FILENAME}' carregado.")
    except FileNotFoundError:
        print(f"Erro: Mapa de universidades '{MAP_FILENAME}' não encontrado.")
        print("Por favor, execute 'processamento_dados.py' primeiro.")
        return None, None

    return model, mapa_nomes

def explore_model(model, mapa_nomes):
    """Inicia um loop interativo para explorar o modelo."""
    print("\n--- Explorador de Similaridade de Universidades (Foco BR <-> EX) ---")
    print("Digite um código (ex: 'br_1', 'ex_1') para ver universidades similares.")
    print("Digite 'q' ou 'sair' para fechar o explorador.")

    while True:
        print("-" * 50)
        codigo_input = input("Digite o código da universidade: ").strip().lower()

        if codigo_input in ['q', 'sair']:
            print("Encerrando o explorador.")
            break

        if not codigo_input:
            continue

        try:
            similares = model.wv.most_similar(codigo_input, topn=40)

            nome_input = mapa_nomes.get(codigo_input, codigo_input)

            print(f"\nTop {TOP_N} destinos ESTRANGEIROS ('ex_') mais similares a '{nome_input}' ({codigo_input}):")

            count_printed = 0
            for codigo, score in similares:

                if str(codigo).startswith('ex_'):

                    nome_universidade = mapa_nomes.get(codigo, f"Código desconhecido ({codigo})")
                    print(f"  - {nome_universidade} ({codigo}) | Similaridade: {score:.4f}")

                    count_printed += 1

                    if count_printed >= TOP_N:
                        break

            if count_printed == 0:
                print("  Nenhum destino estrangeiro ('ex_') encontrado nos top 40 resultados.")

        except KeyError:
            print(f"\nErro: O código '{codigo_input}' não foi encontrado no vocabulário do modelo.")
            print("Isso pode significar que a universidade não apareceu com frequência suficiente (min_count=5).")
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    model, mapa_nomes = load_resources()
    if model and mapa_nomes:
        explore_model(model, mapa_nomes)
