from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.features import ManualTfidfVectorizer
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor
from src.tokenize import tokenize


MAX_ROWS = 120_000
PERCENTILE_FILTER = 99.0
RANDOM_STATE = 42
VAL_SIZE = 0.2
TARGET_NAME = "calories"

MAX_FEATURES_VALUES = [10, 50, 100, 300, 600, 1000, 1500, 2000, 4000]

EXPERIMENTS = [
    # ("tfidf", "ridge"),
    ("tfidf", "random_forest"),
    ("tfidf", "lgbm"),          
    ("tfidf", "lgbm_servings"),  
    ("manual_tfidf", "lgbm"),          
    ("manual_tfidf", "lgbm_servings"),  
]


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


def build_vectorizers(max_features):
    return {
        "tfidf": lambda: TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=max_features,
        ),
        "manual_tfidf": lambda: ManualTfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=max_features,
        ),
    }


def build_models():
    def make_lgbm():
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return {
        "ridge": {
            "factory": lambda: Ridge(alpha=1.0),
            "requires_dense": False,
            "uses_servings": False,
        },
        "elasticnet_gd": {
            "factory": lambda: ElasticNetGDRegressor(
                learning_rate=0.01,
                max_iter=1000,
                l1=0.0001,
                l2=0.0001,
            ),
            "requires_dense": False,
            "uses_servings": False,
        },
        "random_forest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=10,
                max_depth=15,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            ),
            "requires_dense": True,
            "uses_servings": False,
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
            "uses_servings": False,
        },
        "lgbm": {
            "factory": make_lgbm,
            "requires_dense": False,
            "uses_servings": False,  
        },
        "lgbm_servings": {
            "factory": make_lgbm,
            "requires_dense": False,
            "uses_servings": True,  
        },
    }


def build_pipeline(max_features, vectorizer_name, model_info):
    vec_factory = build_vectorizers(max_features)[vectorizer_name]
    
    if model_info.get("uses_servings", False):
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', vec_factory(), 'instructions'),
                ('num', 'passthrough', ['servings'])
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', vec_factory(), 'instructions')
            ],
            remainder='drop'
        )

    steps = [("preprocessor", preprocessor)]
    
    if model_info.get("requires_dense", False):
        steps.append(("to_dense", ToDenseTransformer()))
        
    steps.append(("model", model_info["factory"]()))
    return Pipeline(steps)


def load_data():
    rows = load_training_rows(Path("data/recipes.csv"))[:MAX_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    servings = [float(row["servings"]) for row in rows]
    y = np.array([float(row[TARGET_NAME]) for row in rows], dtype=np.float64)

    X = pd.DataFrame({
        "instructions": texts,
        "servings": servings
    })

    return X, y


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def main():
    models = build_models()
    
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

    scores_by_experiment = {exp: [] for exp in EXPERIMENTS}

    for vec_name, model_name in EXPERIMENTS:
        print(f"\nEksperyment: {vec_name} + {model_name}")
        model_info = models[model_name]

        for max_features in MAX_FEATURES_VALUES:
            pipeline = build_pipeline(
                max_features=max_features,
                vectorizer_name=vec_name,
                model_info=model_info,
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            scores_by_experiment[(vec_name, model_name)].append(mae)
            print(f"  max_features={max_features:4d} -> MAE={mae:.4f}")

    img_dir = Path("img")
    img_dir.mkdir(parents=True, exist_ok=True)
    
    for vec_name, model_name in EXPERIMENTS:
        safe_vec_name = sanitize_name(vec_name)
        safe_model_name = sanitize_name(model_name)
        out_path = img_dir / f"max_features_{safe_vec_name}_{safe_model_name}.png"

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(MAX_FEATURES_VALUES, scores_by_experiment[(vec_name, model_name)], marker="o", label=f"{vec_name} + {model_name}")

        ax.set_title(f"Blad vs max_features ({vec_name} + {model_name})")
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