import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.features import ManualTfidfVectorizer, Word2VecVectorizer
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor
from src.tokenize import tokenize


MAX_ROWS = 120_000
PERCENTILE_FILTER = 99.0
RANDOM_STATE = 42
TARGET_NAMES = ["calories", "fat", "carbohydrates", "protein"]
TFIDF_MAX_FEATURES = 2000

SELECTED_EXPERIMENTS = [
    ("tfidf", "ridge"),
    ("tfidf", "random_forest"),
]


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


def build_models():
    return {
        "ridge": {
            "factory": lambda: Ridge(alpha=1.0),
            "requires_dense": False,
            "supports_multioutput": True,
        },
        "elasticnet_gd": {
            "factory": lambda: ElasticNetGDRegressor(
                learning_rate=0.01,
                max_iter=1000,
                l1=0.0001,
                l2=0.0001,
            ),
            "requires_dense": False,
            "supports_multioutput": False,
        },
        "random_forest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=10,
                max_depth=15,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "requires_dense": True,
            "supports_multioutput": True,
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
            "supports_multioutput": True,
        },
    }


def build_vectorizers():
    return {
        "count": lambda: CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=TFIDF_MAX_FEATURES,
        ),
        "tfidf": lambda: TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=TFIDF_MAX_FEATURES,
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


def build_pipeline(vectorizer_factory, model_factory, requires_dense, supports_multioutput):
    model = model_factory()
    if not supports_multioutput:
        model = MultiOutputRegressor(model)

    steps = [
        (
            "vectorizer",
            vectorizer_factory(),
        )
    ]
    if requires_dense:
        steps.append(("to_dense", ToDenseTransformer()))
    steps.append(("model", model))
    return Pipeline(steps)


def load_data():
    rows = load_training_rows(Path("data/recipes.csv"))[:MAX_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    y = np.array(
        [[float(row[target_name]) for target_name in TARGET_NAMES] for row in rows],
        dtype=np.float64,
    )
    return texts, y

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trenuje wybrane pary (vectorizer, model) i zapisuje je do saved/app/*.joblib"
    )
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Wymus retrening nawet jesli plik modelu juz istnieje.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vectorizers = build_vectorizers()
    models = build_models()

    output_dir = Path("saved/app")
    output_dir.mkdir(parents=True, exist_ok=True)

    to_train = []
    for vectorizer_name, model_name in SELECTED_EXPERIMENTS:
        out_path = output_dir / f"{vectorizer_name}_{model_name}.joblib"
        if out_path.exists() and not args.train:
            print(f"Plik {out_path.as_posix()} juz istnieje.")
            continue
        to_train.append((vectorizer_name, model_name, out_path))

    if not to_train:
        print("\nNic do trenowania.")
        return

    print("Wczytuje dane treningowe...")
    X, y = load_data()
    print(f"Liczba probek: {len(X):,}, liczba targetow: {len(TARGET_NAMES)}")

    for vectorizer_name, model_name, out_path in to_train:
        print(f"\nTrening: {vectorizer_name} + {model_name}")
        start = perf_counter()

        model_info = models[model_name]
        pipeline = build_pipeline(
            vectorizer_factory=vectorizers[vectorizer_name],
            model_factory=model_info["factory"],
            requires_dense=model_info["requires_dense"],
            supports_multioutput=model_info["supports_multioutput"],
        )
        pipeline.fit(X, y)
        joblib.dump(pipeline, out_path)

        elapsed = perf_counter() - start
        print(f"Zapisano: {out_path.as_posix()} ({elapsed:.1f}s)")

    print("\nGotowe.")


if __name__ == "__main__":
    main()