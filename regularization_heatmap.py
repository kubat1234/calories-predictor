from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

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
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor

L1_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
L2_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
LEARNING_RATE = 0.5
MAX_ITER = 2000
OUTPUT_PATH = Path("img/regularization_heatmaps.png")
VERBOSE = False

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


def build_pipeline(l1, l2):
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
        regressor=ElasticNetGDRegressor(
            learning_rate=LEARNING_RATE,
            max_iter=MAX_ITER,
            l1=l1,
            l2=l2,
            verbose=VERBOSE,
            tol=1e-4,
            patience=30,
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_heatmap_for_target(X_train, X_val, y_train_target, y_val_target, target_name):
    mae_scores = np.zeros((len(L1_VALUES), len(L2_VALUES)), dtype=np.float64)
    r2_scores = np.zeros((len(L1_VALUES), len(L2_VALUES)), dtype=np.float64)
    best_mae = None
    best_r2 = None
    best_params = None

    start = perf_counter()
    for i, l1 in enumerate(L1_VALUES):
        for j, l2 in enumerate(L2_VALUES):
            pipeline = build_pipeline(l1=l1, l2=l2)
            pipeline.fit(X_train, y_train_target)
            y_pred = pipeline.predict(X_val)
            mae = mean_absolute_error(y_val_target, y_pred)
            r2 = r2_score(y_val_target, y_pred)
            mae_scores[i, j] = mae
            r2_scores[i, j] = r2

            if best_r2 is None or r2 > best_r2:
                best_mae = mae
                best_r2 = r2
                best_params = (l1, l2)

            print(
                f"[{target_name}] l1={l1:.1e}, l2={l2:.1e} -> val MAE={mae:.4f}, val R2={r2:.4f}"
            )

    elapsed = perf_counter() - start
    print(
        f"\n[{target_name}] NAJLEPSZY WYNIK: R2={best_r2:.4f}, MAE={best_mae:.4f}, "
        f"l1={best_params[0]:.1e}, l2={best_params[1]:.1e}, czas={elapsed:.1f}s"
    )

    return mae_scores, r2_scores, best_mae, best_r2, best_params


def plot_heatmaps(scores_by_target):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, target_name in enumerate(TARGET_NAMES):
        scores = scores_by_target[target_name]["r2_scores"]
        ax = axes[idx]

        im = ax.imshow(scores, cmap="viridis", aspect="auto", origin="upper")
        ax.set_title(f"{target_name}: walidacyjne R2")
        ax.set_xlabel("l2")
        ax.set_ylabel("l1")

        ax.set_xticks(range(len(L2_VALUES)))
        ax.set_xticklabels([f"{v:.1e}" for v in L2_VALUES], rotation=45, ha="right")
        ax.set_yticks(range(len(L1_VALUES)))
        ax.set_yticklabels([f"{v:.1e}" for v in L1_VALUES])

        for i in range(len(L1_VALUES)):
            for j in range(len(L2_VALUES)):
                ax.text(
                    j,
                    i,
                    f"{scores[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if scores[i, j] > np.nanmedian(scores) else "white",
                    fontsize=8,
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R2")

    fig.suptitle("ElasticNetGDRegressor: heatmapy l1/l2 (walidacyjne R2)", fontsize=14)
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

        mae_scores, r2_scores, best_mae, best_r2, best_params = compute_heatmap_for_target(
            X_train=X_train,
            X_val=X_val,
            y_train_target=y_train[:, target_idx],
            y_val_target=y_val[:, target_idx],
            target_name=target_name,
        )

        scores_by_target[target_name] = {
            "mae_scores": mae_scores,
            "r2_scores": r2_scores,
            "best_mae": best_mae,
            "best_r2": best_r2,
            "best_params": best_params,
        }

    print("\n" + "-" * 70)
    print("Podsumowanie najlepszych parametrow (walidacyjne R2):")
    for target_name in TARGET_NAMES:
        best_mae = scores_by_target[target_name]["best_mae"]
        best_r2 = scores_by_target[target_name]["best_r2"]
        best_l1, best_l2 = scores_by_target[target_name]["best_params"]
        print(
            f"  {target_name:14s} -> R2={best_r2:.4f}, MAE={best_mae:.4f}, l1={best_l1:.1e}, l2={best_l2:.1e}"
        )

    plot_heatmaps(scores_by_target)
    print(f"\nZapisano heatmapy: {OUTPUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
