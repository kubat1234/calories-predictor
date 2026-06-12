from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from config_models import build_models, build_vectorizers, build_pipeline

from config_constants import (MAX_TRAIN_ROWS, PERCENTILE_FILTER, RANDOM_SEED)

MAX_ROWS = 60000
PERCENTILE_FILTER = PERCENTILE_FILTER
RANDOM_STATE = RANDOM_SEED

TARGET_NAME = "calories"
TRAIN_SIZES = np.linspace(0.05, 1.0, 8)
VAL_SIZE = 0.2

EXPERIMENTS = [
    ("tfidf", "ridge"),
    ("manual_tfidf", "ridge"),
    ("tfidf", "random_forest"),
    ("tfidf", "lgbm"),          
    ("tfidf", "lgbm_servings"),  
    ("manual_tfidf", "lgbm"),          
    ("manual_tfidf", "lgbm_servings"),  
]


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
    servings = [float(row["servings"]) for row in rows]
    
    y = np.array([float(row[target_to_index[TARGET_NAME]]) for row in rows], dtype=np.float64)

    X = pd.DataFrame({
        "instructions": texts,
        "servings": servings
    })

    return X, y


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def compute_holdout_curve(pipeline, X_train, y_train, X_val, y_val, train_sizes):
    train_r2 = []  
    val_r2 = []
    train_sizes_abs = []

    n_train = len(X_train)
    for frac in train_sizes:
        size = max(2, int(n_train * float(frac)))
        size = min(size, n_train)

        X_part = X_train.iloc[:size]
        y_part = y_train[:size]

        pipeline.fit(X_part, y_part)

        y_train_pred = pipeline.predict(X_part)
        y_val_pred = pipeline.predict(X_val)

        train_r2.append(r2_score(y_part, y_train_pred))
        val_r2.append(r2_score(y_val, y_val_pred))
        train_sizes_abs.append(size)

    return np.array(train_sizes_abs), np.array(train_r2), np.array(val_r2)


def plot_curve(train_sizes_abs, train_r2, val_r2, title, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes_abs, train_r2, marker="o", label="train")
    ax.plot(train_sizes_abs, val_r2, marker="o", label="val")

    ax.set_title(title)
    ax.set_xlabel("train size")
    ax.set_ylabel("R²")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    vectorizers = build_vectorizers(max_features=2000)
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
            model_info=model_info,
            is_multioutput=False
        )

        job_start = perf_counter()
        train_sizes_abs, train_r2, val_r2 = compute_holdout_curve(
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
            train_r2=train_r2,
            val_r2=val_r2,
            title=f"{vec_name} + {model_name}",
            output_path=out_path,
        )

        elapsed = perf_counter() - job_start
        best_val_r2 = float(np.max(val_r2)) 
        summary.append((vec_name, model_name, best_val_r2, elapsed, out_path.as_posix()))
        print(f"Zapisane: {out_path.as_posix()} (best val R²: {best_val_r2:.4f}, {elapsed:.1f}s)")

    print("\nPodsumowanie:")
    for vec_name, model_name, r2, elapsed, out_path in sorted(summary, key=lambda x: x[2], reverse=True):
        print(
            f"  {vec_name:14s} + {model_name:14s} | "
            f"best val R²={r2:.4f} | czas={elapsed:.1f}s | {out_path}"
        )

    print(f"\nGotowe. Caly run: {perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()