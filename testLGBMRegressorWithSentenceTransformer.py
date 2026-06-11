from pathlib import Path
import numpy as np
import lightgbm as lgb
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.features import build_text_features
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows

def main():
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    rows = load_training_rows(Path("data/recipes.csv"))[:200000]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=98.0)

    texts = [row["instructions"] for row in rows]
    targets = [float(row["calories"]) for row in rows]

    print("Generowanie embeddingów z użyciem Transformera...")
    X_dense = encoder.encode(texts, show_progress_bar=True, batch_size=256)

    X_train, X_test, y_train, y_test = train_test_split(
        X_dense, targets, test_size=0.2, random_state=42
    )
    model_path = Path("lgbm_recipe_model_with_ts.joblib")

    if model_path.exists():
        print(f"Wczytywanie modelu")
        model = joblib.load(model_path)
    else:
        print("Trenowanie modelu")
        lgbm_base = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )

        model = TransformedTargetRegressor(
            regressor=lgbm_base,
            func=np.log1p,
            inverse_func=np.expm1
        )

        model.fit(X_train, y_train)

        joblib.dump(model, model_path)

    print("Ewaluacja...")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Liczba cech tekstowych: {X_train.shape[1]}")
    print(f"Trening: {len(X_train)}")
    print(f"Test: {len(X_test)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()