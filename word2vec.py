import pandas as pd
from gensim.models import Word2Vec
import logging
import ast

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def train_br_ex_mobility_model():
    """
    Treina um modelo Word2Vec focado especificamente na mobilidade
    internacional BR -> EX, usando pares de transição como corpus.
    """
    try:
        df_sequencias = pd.read_csv("dados/df_tabela_sequencias.csv")
    except FileNotFoundError as e:
        print(f"Erro ao carregar o arquivo: {e}")
        print("Certifique-se de que 'df_tabela_sequencias.csv' foi gerado por 'processamento_dados.py'.")
        return

    df_sequencias['universidade_lista'] = df_sequencias['universidade_lista'].apply(ast.literal_eval)

    corpus_br_ex = []
    for traj in df_sequencias['universidade_lista']:
        for i in range(len(traj) - 1):
            origem = str(traj[i])
            destino = str(traj[i+1])

            if origem.startswith('br_') and destino.startswith('ex_'):
                corpus_br_ex.append([origem, destino])

    if not corpus_br_ex:
        print("Nenhuma trajetória de mobilidade internacional (BR -> EX) foi encontrada nos dados.")
        return

    print(f"Corpus específico de mobilidade BR->EX criado com {len(corpus_br_ex)} transições.")
    print("\nIniciando o treinamento do modelo Word2Vec (BR->EX)...")

    w2v_model = Word2Vec(
        sentences=corpus_br_ex,
        sg=1,
        vector_size=300,
        window=1,
        min_count=5,
        epochs=20,
        ns_exponent=1.0,
        negative=5,
        workers=4
    )

    print("\nTreinamento concluído.")

    model_filename = "word2vec_br_ex_model.model"
    w2v_model.save(model_filename)
    print(f"Modelo salvo como '{model_filename}'")
    print("Use 'explorar_modelo.py' para interagir com o modelo.")

if __name__ == '__main__':
    train_br_ex_mobility_model()
