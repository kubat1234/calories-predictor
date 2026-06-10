from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import build_text_features
from src.get_data import load_training_rows
from src.ElasticNetGDRegressor import ElasticNetGDRegressor

def main() -> None:
    rows = load_training_rows(Path("data/recipes.csv"))[:50000]

    texts = [row["instructions"] for row in rows]
    targets = [float(row["calories"]) for row in rows]
    
    servings = [float(row["servings"]) for row in rows]
    
    print("Statystyki kalorii:")
    print(f"  Srednia: {mean(targets):.2f}")
    print(f"  Mediana: {median(targets):.2f}")
    print(f"  Odchylenie standardowe: {stdev(targets):.2f}")

    X_train_text, X_test_text, X_train_serv, X_test_serv, y_train, y_test = train_test_split(
        texts,
        servings,
        targets,
        test_size=0.2,
        random_state=42,
    )

    X_train_vec, X_test_vec = build_text_features(
        X_train_text,
        X_test_text,
        method="word2vec",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.8,
        vector_size=300
    )

    X_train_serv_col = np.array(X_train_serv).reshape(-1, 1)
    X_test_serv_col = np.array(X_test_serv).reshape(-1, 1)

    X_train = np.hstack((X_train_vec, X_train_serv_col))
    X_test = np.hstack((X_test_vec, X_test_serv_col))

    model =make_pipeline(
        StandardScaler(),
        ElasticNetGDRegressor(learning_rate=0.01, max_iter=3000, l1=0.0, l2=0.01)
    )
    
    print("Trenuję model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, a_min=0, a_max=2000)

    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Liczba cech: {X_train.shape[1]}")
    print(f"Trening: {len(X_train_text)}")
    print(f"Test: {len(X_test_text)}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

if __name__ == "__main__":
    main()