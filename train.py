import argparse
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from config_models import build_models, build_vectorizers, build_pipeline

from config_constants import (
    MAX_TRAIN_ROWS,
    PERCENTILE_FILTER,
    TARGET_NAMES,
    DATA_PATH,
    MODELS_DIR
)

SELECTED_EXPERIMENTS = [
    ("tfidf", "custom_adaboost"),
    # ("tfidf", "ridge"),
    # ("manual_tfidf", "ridge"),
    # ("tfidf", "random_forest"),
    # ("tfidf", "lgbm"),          
    # ("tfidf", "lgbm_servings"),  
    # ("manual_tfidf", "lgbm"),          
    # ("manual_tfidf", "lgbm_servings"),  
]

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
            model_info=model_info,
            is_multioutput=True
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