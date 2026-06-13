import argparse
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.features import ManualTfidfVectorizer, Word2VecVectorizer, MyTfidfVectorizer
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor
from src.tokenize import tokenize

from config import (
    RANDOM_SEED,
    MAX_TRAIN_ROWS,
    PERCENTILE_FILTER,
    TARGET_NAMES,
    TFIDF_MAX_FEATURES,
    DATA_PATH,
    MODELS_DIR
)

SELECTED_EXPERIMENTS = [
    ("tfidf", "ridge"),
    ("tfidf", "random_forest"),
    ("tfidf", "lgbm"),          
    ("tfidf", "lgbm_servings"),  
    ("manual_tfidf", "lgbm"),          
    ("manual_tfidf", "lgbm_servings"),  
    ("my_tfidf", "elasticnet_gd"),
]


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


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
                random_state=RANDOM_SEED, 
                verbose=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return {
        "ridge": {
            "factory": lambda: Ridge(alpha=1.0),
            "requires_dense": False,
            "supports_multioutput": True,
            "uses_servings": False,
        },
        "elasticnet_gd": {
            "factory": lambda: TransformedTargetRegressor(
                regressor=ElasticNetGDRegressor(
                    learning_rate=0.5, 
                    max_iter=2000,   
                    l1=0.001, 
                    l2=0.001,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "requires_dense": False,
            "supports_multioutput": False,
            "uses_servings": False,
        },
        "random_forest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=10,
                max_depth=15,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            "requires_dense": True,
            "supports_multioutput": True,
            "uses_servings": False,
        },
        "mlp": {
            "factory": lambda: MLPRegressor(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=RANDOM_SEED, 
            ),
            "requires_dense": True,
            "supports_multioutput": True,
            "uses_servings": False,
        },
        "lgbm": {
            "factory": make_lgbm,
            "requires_dense": False,
            "supports_multioutput": False, 
            "uses_servings": False,
        },
        "lgbm_servings": {
            "factory": make_lgbm,
            "requires_dense": False,
            "supports_multioutput": False,
            "uses_servings": True,
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
        "my_tfidf": lambda: MyTfidfVectorizer(
            min_df=3,
            max_df=0.8,
            max_features=TFIDF_MAX_FEATURES,
        ),
        "manual_tfidf": lambda: ManualTfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=TFIDF_MAX_FEATURES,
        ),
        "word2vec": lambda: Word2VecVectorizer(
            vector_size=64,
            window=5,
            min_count=2,
            epochs=5,
        ),
    }


def build_pipeline(vectorizer_factory, model_factory, requires_dense, supports_multioutput, uses_servings):
    model = model_factory()
    if not supports_multioutput:
        model = MultiOutputRegressor(model)

    if uses_servings:
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', vectorizer_factory(), 'instructions'),
                ('num', 'passthrough', ['servings'])
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', vectorizer_factory(), 'instructions')
            ],
            remainder='drop'
        )

    steps = [("preprocessor", preprocessor)]
    
    if requires_dense:
        steps.append(("to_dense", ToDenseTransformer()))
        
    steps.append(("model", model))
    
    return Pipeline(steps)


def load_data():
    rows = load_training_rows(DATA_PATH)[:MAX_TRAIN_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    servings = [float(row["servings"]) for row in rows]

    y = np.array(
        [[float(row[target_name]) for target_name in TARGET_NAMES] for row in rows],
        dtype=np.float64,
    )

    X = pd.DataFrame({
        "instructions": texts,
        "servings": servings
    })

    return X, y


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

    output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    to_train = []
    for vectorizer_name, model_name in SELECTED_EXPERIMENTS:
        out_path = output_dir / f"{vectorizer_name}_{model_name}.joblib"
        if out_path.exists() and not args.train:
            print(f"Plik {out_path.as_posix()} juz istnieje. Pomijam.")
            continue
        to_train.append((vectorizer_name, model_name, out_path))

    if not to_train:
        print("\nNic do trenowania. Uzyj flagi -t, aby wymusic retrening.")
        return

    print("Wczytuje dane treningowe...")
    X, y = load_data()
    
    print(f"Liczba probek treningowych: {len(X):,}, liczba targetow: {len(TARGET_NAMES)}")

    for vectorizer_name, model_name, out_path in to_train:
        print(f"\n{'-'*50}")
        print(f"Eksperyment: {vectorizer_name} + {model_name}")
        start = perf_counter()

        model_info = models[model_name]
        pipeline = build_pipeline(
            vectorizer_factory=vectorizers[vectorizer_name],
            model_factory=model_info["factory"],
            requires_dense=model_info["requires_dense"],
            supports_multioutput=model_info["supports_multioutput"],
            uses_servings=model_info.get("uses_servings", False),
        )
        
        print("  1. Wektoryzacja danych...")
        X_transformed = X
        for step_name, step in pipeline.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed, y)
            
        num_features = X_transformed.shape[1]
        print(f"  => Rzeczywista liczba cech (features) wejściowych: {num_features}")
        
        print("  2. Trenowanie modelu (może chwilę potrwać)...")
        final_step_name, final_model = pipeline.steps[-1]
        final_model.fit(X_transformed, y)

        joblib.dump(pipeline, out_path)

        elapsed = perf_counter() - start
        print(f"Gotowe! Zapisano: {out_path.as_posix()} (Całkowity czas: {elapsed:.1f}s)")

    print("\nZakończono wszystkie eksperymenty.")


if __name__ == "__main__":
    main()