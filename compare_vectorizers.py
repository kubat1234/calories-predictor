from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.features import ManualTfidfVectorizer, Word2VecVectorizer

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows
from src.tokenize import tokenize


TARGET_NAMES = ["calories", "fat", "carbohydrates", "protein"]
MAX_ROWS = 120_000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def format_seconds(seconds: float) -> str:
	seconds = max(0.0, float(seconds))
	minutes, remainder = divmod(int(round(seconds)), 60)
	hours, minutes = divmod(minutes, 60)
	if hours:
		return f"{hours}h {minutes:02d}m {remainder:02d}s"
	if minutes:
		return f"{minutes}m {remainder:02d}s"
	return f"{remainder}s"


def build_ridge_pipeline(vectorizer):
	return TransformedTargetRegressor(
		regressor=Pipeline(
			[
				("vectorizer", vectorizer),
				("ridge", Ridge(alpha=1.0, solver="auto")),
			]
		),
		func=np.log1p,
		inverse_func=np.expm1,
	)


def build_vectorizers():
	vectorizers = [
		(
			"count",
			CountVectorizer(
				tokenizer=tokenize,
				token_pattern=None,
				ngram_range=(1, 2),
				min_df=3,
				max_df=0.9,
				max_features=5000,
			),
		),
		(
			"tfidf",
			TfidfVectorizer(
				tokenizer=tokenize,
				token_pattern=None,
				ngram_range=(1, 2),
				min_df=3,
				max_df=0.9,
				max_features=5000,
			),
		),
        (
            "manual_tfidf",
            ManualTfidfVectorizer(
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
            ),
        ),
        (
            "word2vec",
            Word2VecVectorizer(
                vector_size=64,
                window=5,
                min_count=2,
                epochs=5,
            ),
        ),
    ]

	return vectorizers


def load_data():
	rows = load_training_rows(Path("data/recipes.csv"))[:MAX_ROWS]
	rows = filter_rows_by_nutrient_percentile(rows, percentile=99.0)

	texts = [row["instructions"] for row in rows]
	targets = np.array(
		[
			[
				float(row["calories"]),
				float(row["fat"]),
				float(row["carbohydrates"]),
				float(row["protein"]),
			]
			for row in rows
		],
		dtype=np.float64,
	)
	return texts, targets


def evaluate_predictions(y_true, y_pred):
	metrics = {}
	for idx, target_name in enumerate(TARGET_NAMES):
		y_true_target = y_true[:, idx]
		y_pred_target = y_pred[:, idx]
		metrics[target_name] = {
			"mae": mean_absolute_error(y_true_target, y_pred_target),
			"rmse": np.sqrt(mean_squared_error(y_true_target, y_pred_target)),
			"r2": r2_score(y_true_target, y_pred_target),
		}
	return metrics


def print_metrics(name, train_metrics, val_metrics, elapsed, estimated_total=None):
	print()
	print(f"[{name}] gotowe po {format_seconds(elapsed)}")
	if estimated_total is not None:
		print(f"[{name}] szacowany czas całkowity: {format_seconds(estimated_total)}")
	print("  train:")
	for target_name in TARGET_NAMES:
		target_metrics = train_metrics[target_name]
		print(
			f"    {target_name:13s} | MAE: {target_metrics['mae']:.3f} | "
			f"RMSE: {target_metrics['rmse']:.3f} | R2: {target_metrics['r2']:.4f}"
		)
	print("  val:")
	for target_name in TARGET_NAMES:
		target_metrics = val_metrics[target_name]
		print(
			f"    {target_name:13s} | MAE: {target_metrics['mae']:.3f} | "
			f"RMSE: {target_metrics['rmse']:.3f} | R2: {target_metrics['r2']:.4f}"
		)


def main():
	print("[start] Wczytywanie danych i przygotowanie splitu walidacyjnego...")
	texts, targets = load_data()
	X_train, X_val, y_train, y_val = train_test_split(
		texts,
		targets,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
	)

	print(
		f"[start] Dane: {len(texts):,} próbek, train={len(X_train):,}, val={len(X_val):,}"
	)

	vectorizers = build_vectorizers()
	print(f"[start] Do sprawdzenia: {len(vectorizers)} vectorizerów")

	results = []
	start_all = perf_counter()
	for index, (name, vectorizer) in enumerate(vectorizers, start=1):
		model_start = perf_counter()
		print()
		print(f"[{index}/{len(vectorizers)}] Trening: {name}")
		print(f"[{index}/{len(vectorizers)}] Buduję pipeline Ridge + log1p...")
		model = build_ridge_pipeline(vectorizer)
		print(f"[{index}/{len(vectorizers)}] Fit na train...")
		model.fit(X_train, y_train)
		print(f"[{index}/{len(vectorizers)}] Predykcja na train...")
		y_train_pred = model.predict(X_train)
		print(f"[{index}/{len(vectorizers)}] Predykcja na walidacji...")
		y_pred = model.predict(X_val)
		elapsed = perf_counter() - model_start
		train_metrics = evaluate_predictions(y_train, y_train_pred)
		val_metrics = evaluate_predictions(y_val, y_pred)
		results.append((name, val_metrics, elapsed))

		average_elapsed = sum(item[2] for item in results) / len(results)
		remaining = average_elapsed * (len(vectorizers) - index)
		estimated_total = (perf_counter() - start_all) + remaining
		print_metrics(name, train_metrics, val_metrics, elapsed, estimated_total)

	print()
	print("[summary] Porównanie vectorizerów po MAE dla calories:")
	best_name = None
	best_mae = None
	for name, metrics, _ in results:
		mae = metrics["calories"]["mae"]
		print(f"  {name:15s} -> MAE calories: {mae:.3f}")
		if best_mae is None or mae < best_mae:
			best_name = name
			best_mae = mae
	print(f"[summary] Najlepszy wynik dla calories: {best_name} ({best_mae:.3f})")


if __name__ == "__main__":
	main()