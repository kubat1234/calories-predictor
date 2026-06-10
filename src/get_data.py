import csv

def _percentile(values, percentile):
	if not values:
		raise ValueError("Empty list")

	if percentile <= 0:
		return min(values)
	if percentile >= 100:
		return max(values)

	sorted_values = sorted(values)
	position = (len(sorted_values) - 1) * (percentile / 100.0)
	lower_index = int(position)
	upper_index = min(lower_index + 1, len(sorted_values) - 1)
	weight = position - lower_index

	lower_value = sorted_values[lower_index]
	upper_value = sorted_values[upper_index]
	return lower_value + (upper_value - lower_value) * weight


def filter_rows_by_nutrient_percentile(
	rows,
	percentile=98.0,
):
	if not rows:
		return []

	calories_values = [float(row["calories"]) for row in rows]
	fat_values = [float(row["fat"]) for row in rows]
	carb_values = [float(row["carbohydrates"]) for row in rows]
	protein_values = [float(row["protein"]) for row in rows]

	calories_limit = _percentile(calories_values, percentile)
	fat_limit = _percentile(fat_values, percentile)
	carb_limit = _percentile(carb_values, percentile)
	protein_limit = _percentile(protein_values, percentile)

	filtered_rows: list[dict[str, str]] = []
	for row in rows:
		calories = float(row["calories"])
		fat = float(row["fat"])
		carbohydrates = float(row["carbohydrates"])
		protein = float(row["protein"])

		if (
			calories > calories_limit
			or fat > fat_limit
			or carbohydrates > carb_limit
			or protein > protein_limit
		):
			continue

		filtered_rows.append(row)

	return filtered_rows


def load_training_rows(path):
	training_rows: list[dict[str, str]] = []

	with path.open(newline="", encoding="utf-8") as file:
		reader = csv.DictReader(file)
		for row in reader:
			instructions = row.get("RecipeInstructions", "")
			calories = row.get("Calories", "")
			servings = row.get("RecipeServings", "")
			fat = row.get("FatContent", "")
			carbohydrates = row.get("CarbohydrateContent", "")
			protein = row.get("ProteinContent", "")

			if (
				not instructions
				or not calories
				or calories == "NA"
				or not servings
				or servings == "NA"
				or not fat
				or fat == "NA"
				or not carbohydrates
				or carbohydrates == "NA"
				or not protein
				or protein == "NA"
				or calories == "0"
			):
				continue

			training_rows.append(
				{
					"instructions": instructions,
					"calories": calories,
					"servings": servings,
					"fat": fat,
					"carbohydrates": carbohydrates,
					"protein": protein,
				}
			)

	return training_rows

