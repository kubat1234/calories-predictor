from __future__ import annotations

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(CURRENT_DIR) in sys.path:
	sys.path.remove(str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.get_data import load_training_rows

FIELDS = [
	("calories",      "Kalorie (kcal)",             "calories"),
	("fat",           "Tłuszcz (g)",                "fat"),
	("carbohydrates", "Węglowodany (g)",             "carbohydrates"),
	("protein",       "Białko (g)",                  "protein"),
]

PERCENTILE_CLIP = 99
ZERO_SAMPLE_LIMIT = 5


def print_stats(name: str, values: list[float]) -> None:
	print(f"\n{'─' * 40}")
	print(f"  {name}")
	print(f"{'─' * 40}")
	print(f"  Liczba rekordów : {len(values)}")
	print(f"  Min             : {min(values):.2f}")
	print(f"  Max             : {max(values):.2f}")
	print(f"  Średnia         : {np.mean(values):.2f}")
	print(f"  Mediana         : {np.median(values):.2f}")
	print(f"  Odch. std.      : {np.std(values):.2f}")
	perc = np.percentile(values, PERCENTILE_CLIP)
	print(f"  Percentyl {PERCENTILE_CLIP} : {perc:.2f}")


def plot_distribution(
	values: list[float],
	label: str,
	filename: str,
	out_dir: Path,
) -> None:
	clip_val = float(np.percentile(values, PERCENTILE_CLIP))
	clipped = [v for v in values if v <= clip_val]

	fig, ax = plt.subplots(figsize=(10, 5))
	fig.suptitle(f"Rozkład: {label}", fontsize=14, fontweight="bold")

	ax.hist(clipped, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
	ax.axvline(np.mean(clipped), color="#DD4444", linewidth=1.5, linestyle="--", label=f"Średnia: {np.mean(clipped):.1f}")
	ax.axvline(np.median(clipped), color="#44AA44", linewidth=1.5, linestyle="-", label=f"Mediana: {np.median(clipped):.1f}")
	ax.set_xlabel(label)
	ax.set_ylabel("Liczba przepisów")
	ax.set_title(f"Histogram (do {PERCENTILE_CLIP}. percentyla)")
	ax.legend()

	plt.tight_layout()
	out_path = out_dir / f"{filename}.png"
	plt.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"  Wykres zapisany: {out_path}")


def plot_calories_vs_text_length(rows: list[dict[str, str]], out_dir: Path) -> None:
	text_lengths = [len(row["instructions"]) for row in rows]
	calories = [float(row["calories"]) for row in rows]

	fig, ax = plt.subplots(figsize=(10, 6))
	fig.suptitle("Kalorie vs długość tekstu przepisu", fontsize=14, fontweight="bold")

	ax.scatter(
		text_lengths,
		calories,
		s=8,
		alpha=0.2,
		color="#2E8B57",
		edgecolors="none",
	)
	ax.set_xlabel("Długość instrukcji (liczba znaków)")
	ax.set_ylabel("Kalorie (kcal)")
	ax.set_title("Scatter plot")
	ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

	plt.tight_layout()
	out_path = out_dir / "calories_vs_text_length.png"
	plt.savefig(out_path, dpi=150)
	plt.close(fig)
	print(f"  Wykres zapisany: {out_path}")


def print_zero_value_recipes(rows: list[dict[str, str]], sample_limit: int = ZERO_SAMPLE_LIMIT) -> None:
	print(f"\n{'═' * 60}")
	print("  Podejrzane rekordy z zerowymi wartościami")
	print(f"{'═' * 60}")

	for field, label, _ in FIELDS:
		zero_rows = [row for row in rows if float(row[field]) == 0.0]
		print(f"\n{label}: znaleziono {len(zero_rows)} rekordów z wartością 0")

		if not zero_rows:
			continue

		for idx, row in enumerate(zero_rows[:sample_limit], start=1):
			instructions_preview = " ".join(row["instructions"].split())
			instructions_preview = instructions_preview[:220] + ("..." if len(instructions_preview) > 220 else "")

			print(f"  [{idx}] calories={float(row['calories']):.2f}, fat={float(row['fat']):.2f}, carbohydrates={float(row['carbohydrates']):.2f}, protein={float(row['protein']):.2f}")
			print(f"      Instrukcje: {instructions_preview}")


def main() -> None:
	data_path = Path("data/recipes.csv")
	img_dir = Path("img")
	img_dir.mkdir(exist_ok=True)

	print("Wczytywanie danych...")
	rows = load_training_rows(data_path)
	print(f"Załadowano {len(rows)} rekordów.")

	for field, label, filename in FIELDS:
		values = [float(row[field]) for row in rows]
		print_stats(label, values)
		plot_distribution(values, label, filename, img_dir)

	print_zero_value_recipes(rows)
	plot_calories_vs_text_length(rows, img_dir)

	print(f"\nGotowe. Wykresy zapisane w folderze: {img_dir.resolve()}")


if __name__ == "__main__":
	main()
