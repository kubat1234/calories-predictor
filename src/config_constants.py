from pathlib import Path

RANDOM_SEED = 42

MAX_TRAIN_ROWS = 120_000
MAX_TEST_ROWS = 80_000
PERCENTILE_FILTER = 98.0
TEST_SIZE = 0.2
TFIDF_MAX_FEATURES = 2000

TARGET_NAMES = ["calories", "fat", "carbohydrates", "protein"]

DATA_PATH = Path("data/recipes.csv")
MODELS_DIR = Path("saved/app")