from pathlib import Path

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows

def main() -> None:
	rows = load_training_rows(Path("data/recipes.csv"))[:200000]
	rows = filter_rows_by_nutrient_percentile(rows, percentile=99.0)

	texts = [row["instructions"] for row in rows]
	targets = [float(row["calories"]) for row in rows]

	X_train_text, X_test_text, y_train, y_test = train_test_split(
		texts,
		targets,
		test_size=0.2,
		random_state=42,
	)

	print(f"Trenowanie modelu...")

	pipeline = Pipeline([
		('vectorizer', TfidfVectorizer()),
		('model', Ridge()),
	])

	param_grid = [
		{
			'vectorizer': [TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8)],
			'vectorizer__ngram_range': [(1, 2)],
			'vectorizer__max_features': [2000],
			'model': [Ridge()],
			'model__alpha': [1.0]
		},
		# {
		# 	'vectorizer': [TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8)],
		# 	'vectorizer__ngram_range': [(1, 2)],
		# 	'vectorizer__max_features': [2000],
		# 	'model': [TransformedTargetRegressor(regressor=Ridge(), func=np.log1p, inverse_func=np.expm1)],
		# 	'model__regressor__alpha': [1.0]
		# },
		# {
		# 	'vectorizer': [TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8)],
		# 	'vectorizer__ngram_range': [(1, 2)],
		# 	'vectorizer__max_features': [2000],
		# 	'model': [RandomForestRegressor()],
		# 	'model__n_estimators': [50],
		# 	'model__max_depth': [20]
		# },
		# {
		# 	'vectorizer': [TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8)],
		# 	'vectorizer__ngram_range': [(1, 2)],
		# 	'vectorizer__max_features': [2000],
		# 	'model': [MLPRegressor()],
		# 	'model__hidden_layer_sizes': [(100, 50)],
		# 	'model__activation': ["relu"],
		# 	'model__solver': ["adam"],
		# 	'model__max_iter': [200],
		# 	'model__random_state': [42],
		# 	'model__batch_size': [64]
		# },
	]

	search = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, verbose=3)

	search.fit(X_train_text, y_train)

	best_model = search.best_estimator_

	y_pred = best_model.predict(X_test_text)

	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	r2 = r2_score(y_test, y_pred)

	print("WYNIKI NA ZBIORZE TESTOWYM:\n" + "="*50)
	print(f"MAE : {mae:.2f}")
	print(f"MSE  : {mse:.2f}")
	print(f"RMSE      : {rmse:.2f}")
	print(f"R2 : {r2:.4f}")
	print("="*50)


if __name__ == "__main__":
	main()