from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import joblib

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.features import build_text_features
from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.models.CustomNeuralNetworkRegressor import CustomNeuralNetworkRegressor

def main() -> None:
	rows = load_training_rows(Path("data/recipes.csv"))[:200000]
	rows = filter_rows_by_nutrient_percentile(rows, percentile=98.0)

	texts = [row["instructions"] for row in rows]
	targets = [float(row["calories"]) for row in rows]
	
	print("Statystyki kalorii:")
	print(f"  Srednia: {mean(targets):.2f}")
	print(f"  Mediana: {median(targets):.2f}")
	print(f"  Odchylenie standardowe: {stdev(targets):.2f}")
	print(f"  Min: {min(targets):.2f}")
	print(f"  Max: {max(targets):.2f}")

	X_train_text, X_test_text, y_train, y_test = train_test_split(
		texts,
		targets,
		test_size=0.2,
		random_state=42,
	)

	X_train, X_test = build_text_features(
		X_train_text,
		X_test_text,
		method="word2vec",
		ngram_range=(1, 2),
		min_df=10,
		max_df=0.8,
		vector_size = 200
	)

	print(f"Liczba cech: {X_train.shape[1]}")

	if hasattr(X_train, "toarray"):
		print("Konwersja rzadkiej macierzy na gęstą dla sieci neuronowej...")
		X_train = X_train.toarray()
		X_test = X_test.toarray()

	input_dim = X_train.shape[1]

	layer_sizes = [input_dim, 128, 32, 1]
        

	# model = RandomForestRegressor(
	# 	n_estimators=50,
	# 	max_depth=20,
	# 	random_state=42,
	# 	n_jobs=-1,
	# )

	# model = TransformedTargetRegressor(
	# 	regressor=Ridge(),
	# 	func=np.log1p,
	# 	inverse_func=np.expm1,
	# )
	base_model = CustomNeuralNetworkRegressor(
		layer_sizes = layer_sizes,
		epochs = 200,
		learning_rate = 0.01,
		print_every=1,
	)

	model = TransformedTargetRegressor(
		regressor = base_model,
		transformer=StandardScaler()
	)
	
	print("Trenuję model...")
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)

	mae = mean_absolute_error(y_test, predictions)
	rmse = root_mean_squared_error(y_test, predictions)
	r2 = r2_score(y_test, predictions)

	print(f"Liczba cech tekstowych: {X_train.shape[1]}")
	print(f"Trening: {len(X_train_text)}")
	print(f"Test: {len(X_test_text)}")
	print(f"MAE: {mae:.2f}")
	print(f"RMSE: {rmse:.2f}")
	print(f"R2: {r2:.4f}")

	model_path = "recipe_nn_model.joblib"
	print(f"Zapisuję model do pliku {model_path}...")
	joblib.dump(model, model_path)
	print("Zapis zakończony sukcesem!")


if __name__ == "__main__":
	main()