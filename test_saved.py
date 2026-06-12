from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows


MAX_ROWS = 120_000
PERCENTILE_FILTER = 99.0
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_PATH = Path("saved/app/manual_tfidf_lgbm.joblib")
TARGET_NAMES = ["calories", "fat", "carbohydrates", "protein"]


def load_data():
    rows = load_training_rows(Path("data/recipes.csv"))[120000:200000]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    servings = [float(row["servings"]) for row in rows]
    targets = np.array(
        [[float(row[name]) for name in TARGET_NAMES] for row in rows],
        dtype=np.float64,
    )

    X = pd.DataFrame(
        {
            "instructions": texts,
            "servings": servings,
        }
    )
    return X, targets


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nie znaleziono modelu: {MODEL_PATH.as_posix()}"
        )

    print(f"Wczytywanie modelu: {MODEL_PATH.as_posix()}")
    model = joblib.load(MODEL_PATH)

    print("Wczytywanie danych i podział train/test...")
    X, y = load_data()
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Predykcja na zbiorze testowym...")
    predictions = model.predict(X_test)

    print(f"Test size: {len(X_test)}")
    if np.ndim(predictions) == 1:
        predictions = predictions.reshape(-1, 1)

    for idx, target_name in enumerate(TARGET_NAMES):
        if idx >= predictions.shape[1]:
            break

        y_true_target = y_test[:, idx]
        y_pred_target = predictions[:, idx]

        mae = mean_absolute_error(y_true_target, y_pred_target)
        mse = mean_squared_error(y_true_target, y_pred_target)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_target, y_pred_target)

        print(f"{target_name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2: {r2:.4f}")


if __name__ == "__main__":
    main()
