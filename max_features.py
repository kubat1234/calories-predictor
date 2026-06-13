from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from config_models import build_models, build_vectorizers, build_pipeline

from config_constants import (MAX_TRAIN_ROWS, PERCENTILE_FILTER, RANDOM_SEED, TEST_SIZE)

MAX_ROWS = 60000
PERCENTILE_FILTER = PERCENTILE_FILTER
RANDOM_STATE = RANDOM_SEED
VAL_SIZE = TEST_SIZE
TARGET_NAME = "calories"

MAX_FEATURES_VALUES = [10, 50, 100, 300, 600, 1000, 1500, 2000, 4000]

EXPERIMENTS = [
    # ("tfidf", "ridge"),
    # ("manual_tfidf", "ridge"),
    # ("tfidf", "elasticnet_gd"),
    # ("manual_tfidf", "elasticnet_gd"),
    # ("tfidf", "random_forest"),
    # ("tfidf", "lgbm"),          
    # ("tfidf", "lgbm_servings"),  
    # ("manual_tfidf", "lgbm"),          
    ("manual_tfidf", "lgbm_servings"),  
    ("tfidf", "custom_nn"),  
]


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
    
    img_dir = Path("img")
    img_dir.mkdir(parents=True, exist_ok=True)
    
    log_csv_path = img_dir / "max_features_results_log.csv"
    with open(log_csv_path, "w", encoding="utf-8") as f:
        f.write("vectorizer,model,max_features,r2_score\n")

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

    for vec_name, model_name in EXPERIMENTS:
        print(f"\nEksperyment: {vec_name} + {model_name}")
        model_info = models[model_name]
        
        current_scores = []

        for max_features in MAX_FEATURES_VALUES:
            vectorizers = build_vectorizers(max_features=max_features)
            
            pipeline = build_pipeline(
                vectorizer_factory=vectorizers[vec_name],
                model_info=model_info,
                is_multioutput=False 
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            r2 = r2_score(y_val, y_pred) 
            current_scores.append(r2)
            
            print(f"  max_features={max_features:4d} -> R²={r2:.4f}") 
            
            with open(log_csv_path, "a", encoding="utf-8") as f:
                f.write(f"{vec_name},{model_name},{max_features},{r2:.6f}\n")

        safe_vec_name = sanitize_name(vec_name)
        safe_model_name = sanitize_name(model_name)
        out_path = img_dir / f"max_features_{safe_vec_name}_{safe_model_name}.png"

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(MAX_FEATURES_VALUES, current_scores, marker="o", label=f"{vec_name} + {model_name}")

        ax.set_title(f"R² vs max_features ({vec_name} + {model_name})")  
        ax.set_xlabel("max_features")
        ax.set_ylabel("R²")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        print(f"Zapisano wykres: {out_path.as_posix()}")

if __name__ == "__main__":
    main()