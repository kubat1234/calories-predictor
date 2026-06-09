from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

def load_training_rows(path: Path) -> list[dict[str, str]]:
	training_rows: list[dict[str, str]] = []

	with path.open(newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		for row in reader:
			instructions = row.get("RecipeInstructions", "")
			calories = row.get("Calories", "")
			servings = row.get("RecipeServings", "")

			if (
				not instructions
				or not calories
				or calories == "NA"
				or calories == "0"
				or float(calories) > 2000
				or not servings
				or servings == "NA"
			):
				continue

			training_rows.append(
				{
					"instructions": instructions,
					"calories": calories,
					"servings": servings,
				}
			)

	return training_rows

