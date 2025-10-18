import pandas as pd
import ast
from collections import Counter
import numpy as np
from rich import print
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV

PRED_MODO = "br2ex"
FILTER_ZERO_TARGET = True
MIN_NONZERO_IN_HISTORY = 1
COLLAPSE_RARE = True
MIN_CLASS_FREQ = 10

DROP_EX_OUTROS = True
FILTER_HISTORY_BY_MODE = True


def run_classification_full_trajectory():
    df_tabela_sequencias = pd.read_csv("./dados/df_tabela_sequencias.csv")
    df_tabela_sequencias["universidade_lista"] = df_tabela_sequencias["universidade_lista"].apply(ast.literal_eval)

    X_raw, y_raw = [], []

    for _, linha in df_tabela_sequencias.iterrows():
        traj = linha.get("universidade_lista", None)

        if not isinstance(traj, list) or len(traj) < 2:
            continue

        historico = traj[:-1]
        proximo = traj[-1]


        if sum(1 for h in historico if h != "0") < MIN_NONZERO_IN_HISTORY:
            continue

        if FILTER_HISTORY_BY_MODE and not any(
            isinstance(h, str) and h.startswith("br") for h in historico
        ):
            continue

        if FILTER_ZERO_TARGET and proximo == "0":
            continue

        if not str(proximo).startswith("ex_"):
            continue

        X_raw.append(historico)
        y_raw.append(proximo)

    if not X_raw:
        print("Nenhuma amostra válida foi gerada após a filtragem. Encerrando.")
        return

    max_len = max(len(h) for h in X_raw)
    X_padded = [["0"] * (max_len - len(h)) + h for h in X_raw]

    X = pd.DataFrame(X_padded, columns=[f"passo_{i + 1}" for i in range(max_len)])
    y = pd.Series(y_raw, name="proximo_codigo")
    print(X)

    if COLLAPSE_RARE and len(y) > 0:
        freq = Counter(y)

        def colapsa(label: str) -> str:
            if freq[label] >= MIN_CLASS_FREQ:
                return label
            if isinstance(label, str) and label.startswith("ex_"):
                return "outros_ex"
            return "outros"

        y = y.map(colapsa)

    if DROP_EX_OUTROS and len(y) > 0:
        mask_keep = y != "outros_ex"
        X = X.loc[mask_keep].reset_index(drop=True)
        y = y.loc[mask_keep].reset_index(drop=True)

    if X.empty:
        print("Todas as amostras foram removidas após o processamento. Encerrando.")
        return

    stratify_y = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna("0").astype(str),
        y.astype(str),
        test_size=0.2,
        random_state=42,
        stratify=stratify_y,
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    X_train_enc = ohe.fit_transform(X_train)
    X_test_enc = ohe.transform(X_test)

    all_y_classes = y.unique()
    le_y = LabelEncoder().fit(all_y_classes)

    y_train_enc = le_y.transform(y_train)
    y_test_enc = le_y.transform(y_test)

    knn = KNeighborsClassifier(n_jobs=-1)

    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    clf = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    clf.fit(X_train_enc, y_train_enc)

    print("Melhores Parâmetros Encontrados:")
    print(clf.best_params_)

    y_pred_enc = clf.predict(X_test_enc)

    labels_todos = np.arange(len(le_y.classes_))
    nomes_classes = le_y.classes_

    print("Relatório de Classificação (KNN):")
    print(
        classification_report(
            y_test_enc,
            y_pred_enc,
            labels=labels_todos,
            target_names=nomes_classes,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    run_classification_full_trajectory()
