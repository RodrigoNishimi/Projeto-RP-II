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

janela = 3
FILTER_ZERO_TARGET = True
PRED_MODO = "br2ex"
MIN_NONZERO_IN_HISTORY = 1
COLLAPSE_RARE = True
MIN_CLASS_FREQ = 10

DROP_EX_OUTROS = True

def run_classification_for_window():
    df_tabela_sequencias = pd.read_csv("./dados/df_tabela_sequencias.csv")
    df_tabela_sequencias["universidade_lista"] = df_tabela_sequencias["universidade_lista"].apply(ast.literal_eval)

    X_raw, y_raw = [], []

    for _, linha in df_tabela_sequencias.iterrows():
        traj = linha.get("universidade_lista", None)
        if not isinstance(traj, list) or len(traj) < janela + 1:
            continue

        for t in range(janela, len(traj)):
            historico = traj[t - janela : t]
            proximo = traj[t]

            n_nonzero = sum(1 for h in historico if h != "0")
            if n_nonzero < MIN_NONZERO_IN_HISTORY:
                continue

            if not any(isinstance(h, str) and h.startswith("br") for h in historico):
                continue

            if FILTER_ZERO_TARGET and proximo == "0":
                continue

            if not proximo.startswith("ex_"):
                continue

            X_raw.append(historico)
            y_raw.append(proximo)

    X = pd.DataFrame(X_raw, columns=[f"ano_{i + 1}" for i in range(janela)])
    y = pd.Series(y_raw, name="proximo_codigo")
    print(y)

    if DROP_EX_OUTROS and len(y) > 0:
        mask_keep = y != "ex_outros"
        X = X.loc[mask_keep].reset_index(drop=True)
        y = y.loc[mask_keep].reset_index(drop=True)

    if COLLAPSE_RARE and len(y) > 0:
        freq = Counter(y)

        def colapsa(label: str) -> str:
            if freq[label] >= MIN_CLASS_FREQ:
                return label
            if isinstance(label, str) and label.startswith("br"):
                return "outros_br"
            if isinstance(label, str) and label.startswith("ex_"):
                return "outros_ex"
            return "outros"

        y = y.map(colapsa)


    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna("0").astype(str),
        y.astype(str),
        test_size=0.2,
        random_state=42,
        stratify=(
            y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
        )
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    X_train_enc = ohe.fit_transform(X_train)
    X_test_enc = ohe.transform(X_test)

    le_y = LabelEncoder()
    le_y.fit(y.astype(str))

    y_train_enc = le_y.transform(y_train)
    y_test_enc = le_y.transform(y_test)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150, 200, 300, 400],
        'max_depth': [10, 30, None],
        'min_samples_leaf': [1, 5]
    }

    clf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    clf.fit(X_train_enc, y_train_enc)

    print(clf.best_estimator_)

    y_pred_enc = clf.predict(X_test_enc)

    labels_todos = np.arange(len(le_y.classes_))
    nomes_classes = le_y.inverse_transform(labels_todos)

    print("Relatório de Classificação:")
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
    run_classification_for_window()
