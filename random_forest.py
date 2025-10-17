import pandas as pd
import ast
from collections import Counter
import numpy as np
from rich import print
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

def prepare_data(df, window_size):
    """
    Prepara os dados usando uma janela deslizante de tamanho fixo,
    conforme a metodologia do artigo.
    """
    X_raw, y_raw = [], []
    print(f"\n[bold blue]Preparando dados para janela de tamanho: {window_size}[/bold blue]")

    for _, linha in df.iterrows():
        traj = linha.get("universidade_lista", None)
        # A trajetória precisa ser longa o suficiente para a janela + 1 (destino)
        if not isinstance(traj, list) or len(traj) < window_size + 1:
            continue

        # Itera sobre a trajetória para criar múltiplos exemplos
        for t in range(window_size, len(traj)):
            historico = traj[t - window_size : t]
            destino = traj[t]

            # Garante que o histórico não seja apenas de '0's e o destino não seja '0'
            if any(h != "0" for h in historico) and destino != "0":
                X_raw.append(historico)
                y_raw.append(destino)

    if not X_raw:
        print("[yellow]Nenhum dado válido gerado para este tamanho de janela.[/yellow]")
        return None, None

    print(f"Gerados {len(X_raw)} exemplos.")
    return X_raw, y_raw


def run_classification_for_window(df, window_size):
    """
    Executa todo o pipeline de classificação para um determinado tamanho de janela.
    """
    X_raw, y_raw = prepare_data(df, window_size)

    if not X_raw:
        return

    # Filtra classes com poucas amostras para permitir a estratificação
    class_counts = Counter(y_raw)
    multi_sample_classes = {cls for cls, count in class_counts.items() if count >= 2}
    if not multi_sample_classes:
        print("[yellow]Não foi possível estratificar: nenhuma classe com 2+ amostras.[/yellow]")
        return

    X_filtered, y_filtered = [], []
    for x_sample, y_sample in zip(X_raw, y_raw):
        if y_sample in multi_sample_classes:
            X_filtered.append(x_sample)
            y_filtered.append(y_sample)

    if not X_filtered:
        print("[yellow]Nenhum dado restante após filtrar classes raras.[/yellow]")
        return

    X_raw, y_raw = X_filtered, y_filtered

    # Padding
    max_len = max(len(h) for h in X_raw)
    X_pad = np.array([h + ["0"] * (max_len - len(h)) for h in X_raw])
    y = np.array(y_raw)

    # Ajuste dinâmico do test_size
    n_classes = len(np.unique(y))
    desired_test_size = 0.2
    proposed_test_samples = int(len(y) * desired_test_size)

    if proposed_test_samples < n_classes:
        test_size_param = n_classes
    else:
        test_size_param = desired_test_size

    if len(y) <= test_size_param:
        print(f"[yellow]Não há dados suficientes ({len(y)}) para criar um conjunto de teste de tamanho {test_size_param}. Pulando esta janela.[/yellow]")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X_pad, y, test_size=test_size_param, random_state=42, stratify=y
    )

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train_enc = ohe.fit_transform(X_train)
    X_test_enc = ohe.transform(X_test)

    le_y = LabelEncoder()
    le_y.fit(y.astype(str))
    y_train_enc = le_y.transform(y_train)
    y_test_enc = le_y.transform(y_test)

    # Modelo Random Forest (sem GridSearchCV para um teste mais rápido)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_enc, y_train_enc)

    y_pred_test = rf.predict(X_test_enc)

    # Avaliação
    f1_macro = f1_score(y_test_enc, y_pred_test, average='macro', zero_division=0)
    print(f"[bold green]Resultado para Janela = {window_size} -> F1-Score (macro): {f1_macro:.4f}[/bold green]")

    unique_labels_test = np.unique(np.concatenate((y_test_enc, y_pred_test)))
    target_names_test = le_y.inverse_transform(unique_labels_test)

    print(f"[bold]Relatório de Classificação para Janela = {window_size}[/bold]")
    print(
        classification_report(
            y_test_enc,
            y_pred_test,
            labels=unique_labels_test,
            target_names=target_names_test,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    df_sequencias = pd.read_csv("./dados/df_tabela_sequencias.csv")
    df_sequencias["universidade_lista"] = df_sequencias["universidade_lista"].apply(ast.literal_eval)

    # --- Loop de Experimentação ---
    # Testa diferentes tamanhos de janela para ver qual tem a melhor performance.
    tamanhos_de_janela = [2, 3, 4, 5]
    for tamanho in tamanhos_de_janela:
        run_classification_for_window(df_sequencias, tamanho)
        print("-" * 80)
