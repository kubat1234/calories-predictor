from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DATA_PATH,
    MAX_TRAIN_ROWS,
    PERCENTILE_FILTER,
    RANDOM_SEED,
    TARGET_NAMES,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
)
from src.features import MyTfidfVectorizer
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows

L2_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
OUTPUT_PATH = Path("img/regularization_plot.png")


def load_data():
    rows = load_training_rows(DATA_PATH)[:MAX_TRAIN_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    X = pd.DataFrame(
        {
            "instructions": [row["instructions"] for row in rows],
        }
    )

    y = np.array(
        [[float(row[name]) for name in TARGET_NAMES] for row in rows],
        dtype=np.float64,
    )

    return X, y


def build_pipeline(l2_value):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                MyTfidfVectorizer(
                    min_df=3,
                    max_df=0.8,
                    max_features=TFIDF_MAX_FEATURES,
                ),
                "instructions",
            )
        ],
        remainder="drop",
    )

    model = TransformedTargetRegressor(
        regressor=Ridge(alpha=l2_value, random_state=RANDOM_SEED),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_r2_curve_for_target(X_train, X_val, y_train_target, y_val_target, target_name):
    r2_scores = []
    best_r2 = None
    best_l2 = None

    start = perf_counter()
    for l2 in L2_VALUES:
        pipeline = build_pipeline(l2_value=l2)
        pipeline.fit(X_train, y_train_target)
        y_pred = pipeline.predict(X_val)
        r2 = r2_score(y_val_target, y_pred)
        r2_scores.append(r2)

        if best_r2 is None or r2 > best_r2:
            best_r2 = r2
            best_l2 = l2

        print(f"[{target_name}] l2={l2:.1e} -> val R2={r2:.4f}")

    elapsed = perf_counter() - start
    print(
        f"\n[{target_name}] NAJLEPSZY WYNIK: R2={best_r2:.4f}, l2={best_l2:.1e}, czas={elapsed:.1f}s"
    )

    return np.array(r2_scores, dtype=np.float64), best_r2, best_l2


def plot_r2_curves(scores_by_target):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes = axes.ravel()

    for idx, target_name in enumerate(TARGET_NAMES):
        target_scores = scores_by_target[target_name]
        ax = axes[idx]

        ax.plot(
            L2_VALUES,
            target_scores["r2_scores"],
            marker="o",
            linewidth=2,
            color="#1f77b4",
        )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{target_name}: walidacyjne R2")
        ax.set_xlabel("l2 (alpha w Ridge)")
        ax.set_ylabel("R2")

        best_l2 = target_scores["best_l2"]
        best_r2 = target_scores["best_r2"]
        ax.scatter([best_l2], [best_r2], color="#d62728", zorder=3)
        ax.annotate(
            f"best: l2={best_l2:.1e}\nR2={best_r2:.3f}",
            xy=(best_l2, best_r2),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728", alpha=0.8),
        )

    fig.suptitle("Ridge: wykres walidacyjnego R2 od regularyzacji l2", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=170)
    plt.close(fig)


def main():
    print("Wczytuje dane...")
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    print(f"Train: {len(X_train):,}, walidacja: {len(X_val):,}")

    scores_by_target = {}

    for target_idx, target_name in enumerate(TARGET_NAMES):
        print(f"\n{'=' * 70}")
        print(f"Target: {target_name}")

        r2_scores, best_r2, best_l2 = compute_r2_curve_for_target(
            X_train=X_train,
            X_val=X_val,
            y_train_target=y_train[:, target_idx],
            y_val_target=y_val[:, target_idx],
            target_name=target_name,
        )

        scores_by_target[target_name] = {
            "r2_scores": r2_scores,
            "best_r2": best_r2,
            "best_l2": best_l2,
        }

    print("\n" + "-" * 70)
    print("Podsumowanie najlepszych parametrow (walidacyjne R2):")
    for target_name in TARGET_NAMES:
        best_r2 = scores_by_target[target_name]["best_r2"]
        best_l2 = scores_by_target[target_name]["best_l2"]
        print(f"  {target_name:14s} -> R2={best_r2:.4f}, l2={best_l2:.1e}")

    plot_r2_curves(scores_by_target)
    print(f"\nZapisano wykres: {OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
