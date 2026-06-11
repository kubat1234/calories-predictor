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
	clipped = np.array([v for v in values if v <= clip_val], dtype=float)
	transformed = np.log1p(clipped)

	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle(f"Rozkład: {label} vs log1p({label})", fontsize=14, fontweight="bold")

	axes[0].hist(clipped, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.4)
	axes[0].axvline(np.mean(clipped), color="#DD4444", linewidth=1.5, linestyle="--", label=f"Średnia: {np.mean(clipped):.1f}")
	axes[0].axvline(np.median(clipped), color="#44AA44", linewidth=1.5, linestyle="-", label=f"Mediana: {np.median(clipped):.1f}")
	axes[0].set_xlabel(label)
	axes[0].set_ylabel("Liczba przepisów")
	axes[0].set_title(f"Histogram (do {PERCENTILE_CLIP}. percentyla)")
	axes[0].legend()

	axes[1].hist(transformed, bins=80, color="#2A9D8F", edgecolor="white", linewidth=0.4)
	axes[1].axvline(np.mean(transformed), color="#DD4444", linewidth=1.5, linestyle="--", label=f"Średnia: {np.mean(transformed):.2f}")
	axes[1].axvline(np.median(transformed), color="#44AA44", linewidth=1.5, linestyle="-", label=f"Mediana: {np.median(transformed):.2f}")
	axes[1].set_xlabel(f"log1p({label})")
	axes[1].set_ylabel("Liczba przepisów")
	axes[1].set_title("Histogram po transformacji np.log1p")
	axes[1].legend()

	plt.tight_layout(rect=(0, 0, 1, 0.95))
	out_path = out_dir / f"{filename}.png"
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

	print(f"\nGotowe. Wykresy zapisane w folderze: {img_dir.resolve()}")


if __name__ == "__main__":
	main()
