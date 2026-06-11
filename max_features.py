from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor
from src.tokenize import tokenize


MAX_ROWS = 120_000
PERCENTILE_FILTER = 99.0
RANDOM_STATE = 42
VAL_SIZE = 0.2
TARGET_NAME = "calories"

SELECTED_MODELS = ["ridge", "random_forest"]

MAX_FEATURES_VALUES = [10, 50, 100, 300, 600, 1000, 1500, 2000, 5000, 10000]

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
        "random_forest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=10,
                max_depth=15,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "requires_dense": True,
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


def build_pipeline(max_features, model_factory, requires_dense):
    steps = [
        (
            "vectorizer",
            TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                max_features=max_features,
            ),
        )
    ]
    if requires_dense:
        steps.append(("to_dense", ToDenseTransformer()))
    steps.append(("model", model_factory()))
    return Pipeline(steps)


def load_data():
    rows = load_training_rows(Path("data/recipes.csv"))[:MAX_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    y = np.array([float(row[TARGET_NAME]) for row in rows], dtype=np.float64)
    return texts, y


def validate_selection(selected_models, all_models):
    unknown = [name for name in selected_models if name not in all_models]
    if unknown:
        raise ValueError(
            f"Nieznane modele: {', '.join(unknown)}. Dostepne: {', '.join(sorted(all_models))}"
        )


def main():
    models = build_models()
    validate_selection(SELECTED_MODELS, models.keys())

    print("Wczytuje dane...")
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    print(f"Train: {len(X_train):,}, val: {len(X_val):,}")

    scores_by_model = {name: [] for name in SELECTED_MODELS}

    for model_name in SELECTED_MODELS:
        print(f"\nModel: {model_name}")
        model_info = models[model_name]

        for max_features in MAX_FEATURES_VALUES:
            pipeline = build_pipeline(
                max_features=max_features,
                model_factory=model_info["factory"],
                requires_dense=model_info["requires_dense"],
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            scores_by_model[model_name].append(mae)
            print(f"  max_features={max_features:4d} -> MAE={mae:.4f}")

    img_dir = Path("img")
    img_dir.mkdir(parents=True, exist_ok=True)
    for model_name in SELECTED_MODELS:
        out_path = img_dir / f"max_features_{model_name}.png"

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(MAX_FEATURES_VALUES, scores_by_model[model_name], marker="o", label=model_name)

        ax.set_title(f"TF-IDF: blad vs max_features ({model_name})")
        ax.set_xlabel("max_features")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        print(f"\nZapisane: {out_path.as_posix()}")

if __name__ == "__main__":
    main()