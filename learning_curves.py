from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.features import ManualTfidfVectorizer, Word2VecVectorizer
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor
from src.tokenize import tokenize


MAX_ROWS = 120_000
PERCENTILE_FILTER = 99.0
RANDOM_STATE = 42

TARGET_NAME = "calories"
TRAIN_SIZES = np.linspace(0.05, 1.0, 8)
VAL_SIZE = 0.2

EXPERIMENTS = [
    ("tfidf", "ridge"),
    ("tfidf", "elasticnet_gd"),
    ("tfidf", "random_forest"),
]


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


def build_vectorizers():
    return {
        "count": lambda: CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=2000,
        ),
        "tfidf": lambda: TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=2000,
        ),
        "manual_tfidf": lambda: ManualTfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
        ),
        "word2vec": lambda: Word2VecVectorizer(
            vector_size=64,
            window=5,
            min_count=2,
            epochs=5,
        ),
    }


def build_models():
    return {
        "ridge": {
            "factory": lambda: Ridge(alpha=1.0),
            "requires_dense": False,
        },
        "random_forest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=10,
                max_depth=15,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
            ),
            "requires_dense": True,
        },
        "elasticnet_gd": {
            "factory": lambda: ElasticNetGDRegressor(
                learning_rate=0.01,
                max_iter=1000,
                l1=0.0001,
                l2=0.0001,
            ),
            "requires_dense": False,
        },
        "mlp": {
            "factory": lambda: MLPRegressor(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
            "requires_dense": True,
        },
    }


def load_data():
    rows = load_training_rows(Path("data/recipes.csv"))[:MAX_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    target_to_index = {
        "calories": "calories",
        "fat": "fat",
        "carbohydrates": "carbohydrates",
        "protein": "protein",
    }
    if TARGET_NAME not in target_to_index:
        valid = ", ".join(target_to_index.keys())
        raise ValueError(f"Nieznany TARGET_NAME='{TARGET_NAME}'. Dostepne: {valid}")

    texts = [row["instructions"] for row in rows]
    y = np.array([float(row[target_to_index[TARGET_NAME]]) for row in rows], dtype=np.float64)

    return texts, y


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)

def build_pipeline(vectorizer_factory, model_factory, requires_dense):
    steps = [("vectorizer", vectorizer_factory())]
    if requires_dense:
        steps.append(("to_dense", ToDenseTransformer()))
    steps.append(("model", model_factory()))
    return Pipeline(steps)


def compute_holdout_curve(pipeline, X_train, y_train, X_val, y_val, train_sizes):
    train_mae = []
    val_mae = []
    train_sizes_abs = []

    n_train = len(X_train)
    for frac in train_sizes:
        size = max(2, int(n_train * float(frac)))
        size = min(size, n_train)

        X_part = X_train[:size]
        y_part = y_train[:size]

        pipeline.fit(X_part, y_part)

        y_train_pred = pipeline.predict(X_part)
        y_val_pred = pipeline.predict(X_val)

        train_mae.append(mean_absolute_error(y_part, y_train_pred))
        val_mae.append(mean_absolute_error(y_val, y_val_pred))
        train_sizes_abs.append(size)

    return np.array(train_sizes_abs), np.array(train_mae), np.array(val_mae)


def plot_curve(train_sizes_abs, train_mae, val_mae, title, output_path):

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes_abs, train_mae, marker="o", label="train")
    ax.plot(train_sizes_abs, val_mae, marker="o", label="val")

    ax.set_title(title)
    ax.set_xlabel("train size")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    vectorizers = build_vectorizers()
    models = build_models()

    img_dir = Path("img")
    img_dir.mkdir(parents=True, exist_ok=True)

    print("Wczytuje dane...")
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(
        f"Mam {len(X):,} probek. Train: {len(X_train):,}, val: {len(X_val):,}."
    )

    total_jobs = len(EXPERIMENTS)
    run_start = perf_counter()
    summary = []

    for i, (vec_name, model_name) in enumerate(EXPERIMENTS, start=1):
        print(f"\n{i}/{total_jobs}: {vec_name} + {model_name}")

        model_info = models[model_name]
        pipeline = build_pipeline(
            vectorizer_factory=vectorizers[vec_name],
            model_factory=model_info["factory"],
            requires_dense=model_info["requires_dense"],
        )

        job_start = perf_counter()
        train_sizes_abs, train_mae, val_mae = compute_holdout_curve(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_sizes=TRAIN_SIZES,
        )

        out_name = f"learning_curve_{sanitize_name(vec_name)}__{sanitize_name(model_name)}.png"
        out_path = img_dir / out_name
        plot_curve(
            train_sizes_abs=train_sizes_abs,
            train_mae=train_mae,
            val_mae=val_mae,
            title=f"{vec_name} + {model_name}",
            output_path=out_path,
        )

        elapsed = perf_counter() - job_start
        best_val_mae = float(np.min(val_mae))
        summary.append((vec_name, model_name, best_val_mae, elapsed, out_path.as_posix()))
        print(f"Zapisane: {out_path.as_posix()} (best val MAE: {best_val_mae:.4f}, {elapsed:.1f}s)")

    print("\nPodsumowanie:")
    for vec_name, model_name, mae, elapsed, out_path in sorted(summary, key=lambda x: x[2]):
        print(
            f"  {vec_name:14s} + {model_name:14s} | "
            f"best val MAE={mae:.4f} | czas={elapsed:.1f}s | {out_path}"
        )

    print(f"\nGotowe. Caly run: {perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()