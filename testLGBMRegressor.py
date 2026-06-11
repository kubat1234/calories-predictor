from pathlib import Path
from statistics import mean, median, stdev
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows

def main():
    rows = load_training_rows(Path("data/recipes.csv"))[:200000]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=98.0)

    texts = [row["instructions"] for row in rows]
    targets = [float(row["calories"]) for row in rows]
    servings = [float(row["servings"]) for row in rows]

    print("Statystyki kalorii:")
    print(f"  Srednia: {mean(targets):.2f}")
    print(f"  Mediana: {median(targets):.2f}")
    print(f"  Odchylenie standardowe: {stdev(targets):.2f}")
    print(f"  Min: {min(targets):.2f}")
    print(f"  Max: {max(targets):.2f}")

    X = pd.DataFrame({
        "instructions": texts,
        "servings": servings
    })
    y = np.array(targets)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf', TfidfVectorizer(), 'instructions'),
            ('num', 'passthrough', ['servings'])
        ]
    )

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

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    param_grid = [
        {
            'preprocessor__tfidf__max_features': [2000],
            'preprocessor__tfidf__ngram_range': [(1, 2)],
            'preprocessor__tfidf__min_df': [2],
            'preprocessor__tfidf__max_df': [0.8],
            
        }
    ]

    model_path = Path("lgbm_pipeline_model.joblib")

    if model_path.exists():
        print(f"Wczytywanie wytrenowanego potoku z pliku {model_path}...")
        best_model = joblib.load(model_path)
    else:
        print("Trenowanie modelu i optymalizacja parametrów (GridSearchCV)...")
        search = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, verbose=3)
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        joblib.dump(best_model, model_path)
        print(f"Zapisano optymalny estymator do pliku: {model_path}")

    print("\nEwaluacja...")
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    X_train_transformed_shape = best_model.named_steps['preprocessor'].transform(X_train).shape

    print(f"Liczba cech wejściowych (po preprocesingu): {X_train_transformed_shape[1]}")
    print(f"Trening: {len(X_train)}")
    print(f"Test: {len(X_test)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()