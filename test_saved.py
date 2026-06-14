import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.get_data import filter_rows_by_nutrient_percentile, load_training_rows

from config_constants import (
    MAX_TRAIN_ROWS,
    MAX_TEST_ROWS,
    PERCENTILE_FILTER,
    TARGET_NAMES,
    MODELS_DIR,
    DATA_PATH
)


def load_test_data():
    rows = load_training_rows(DATA_PATH)[MAX_TRAIN_ROWS : MAX_TRAIN_ROWS + MAX_TEST_ROWS]
    rows = filter_rows_by_nutrient_percentile(rows, percentile=PERCENTILE_FILTER)

    texts = [row["instructions"] for row in rows]
    servings = [float(row["servings"]) for row in rows]
    targets = np.array(
        [[float(row[name]) for name in TARGET_NAMES] for row in rows],
        dtype=np.float64,
    )

    X_test = pd.DataFrame(
        {
            "instructions": texts,
            "servings": servings,
        }
    )
    return X_test, targets


def print_markdown_table(model_name: str, results: list):
    print(f"\nWyniki dla modelu: **{model_name}**")
    print("| Target | MAE | RMSE | R2 |")
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['Target']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.4f} |")
    print("", flush=True)


def save_results_to_png(all_results: list, output_filename: str = "results_summary.png", title: str = "Zestawienie wyników modeli"):
    df = pd.DataFrame(all_results)
    if df.empty:
        return
        
    for col in ["MAE", "RMSE", "R2"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")
    
    fig, ax = plt.subplots(figsize=(12, 1 + len(df) * 0.3))
    ax.axis('tight')
    ax.axis('off')
    
    if len(df.columns) == 5:
        custom_col_widths = [0.35, 0.15, 0.15, 0.15, 0.15]
    else:
        custom_col_widths = None

    table = ax.table(
        cellText=df.values, 
        colLabels=df.columns,
        colWidths=custom_col_widths,
        loc='center', 
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
    
    plt.title(title, pad=20, fontsize=14, weight='bold')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Zapisano zestawienie wyników do pliku: {output_filename}")


def parse_args():
    parser = argparse.ArgumentParser(description="Testuje modele z folderu saved/app")
    parser.add_argument(
        "model",
        nargs="?",
        default="all",
        help="Ścieżka do modelu, nazwa pliku z saved/app lub 'all' aby przetestować wszystko (domyślnie)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    models_to_test = []
    
    if args.model.lower() == "all":
        if not MODELS_DIR.exists():
            print(f"Katalog {MODELS_DIR} nie istnieje. Nie ma nic do testowania.")
            return
        models_to_test = list(MODELS_DIR.glob("*.joblib"))
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            fallback_path = MODELS_DIR / args.model
            if fallback_path.exists():
                models_to_test = [fallback_path]
            else:
                raise FileNotFoundError(f"Nie znaleziono modelu: {model_path.as_posix()} ani {fallback_path.as_posix()}")
        else:
            models_to_test = [model_path]

    if not models_to_test:
        print("Nie znaleziono żadnych modeli do przetestowania.")
        return

    print("Wczytywanie i filtrowanie danych testowych...")
    X_test, y_test = load_test_data()
    print(f"Dane wczytane. Rozmiar zbioru testowego: {len(X_test):,} próbek.\n")

    stats_dir = Path("img/stats")

    all_results = []

    for path in models_to_test:
        model_name = path.stem
        print(f"--> Testowanie modelu: {model_name} ...")
        
        try:
            model = joblib.load(path)
            predictions = model.predict(X_test)
            
            if np.ndim(predictions) == 1:
                predictions = predictions.reshape(-1, 1)

            model_metrics = []
            for idx, target_name in enumerate(TARGET_NAMES):
                if idx >= predictions.shape[1]:
                    break

                y_true_target = y_test[:, idx]
                y_pred_target = predictions[:, idx]

                mae = mean_absolute_error(y_true_target, y_pred_target)
                mse_val = mean_squared_error(y_true_target, y_pred_target)
                rmse = np.sqrt(mse_val)
                r2 = r2_score(y_true_target, y_pred_target)

                metric_row = {
                    "Model": model_name,
                    "Target": target_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                }
                model_metrics.append(metric_row)
                all_results.append(metric_row)

            print_markdown_table(model_name, model_metrics)
            
            individual_output_path = stats_dir / f"{model_name}.png"
            save_results_to_png(
                model_metrics, 
                output_filename=str(individual_output_path), 
                title=f"Wyniki modelu: {model_name}"
            )

        except Exception as e:
            print(f"Błąd podczas testowania {model_name}: {e}\n")

    if all_results:
        save_results_to_png(
            all_results, 
            output_filename="results_summary.png", 
            title="Zbiorcze zestawienie wyników modeli"
        )


if __name__ == "__main__":
    main()