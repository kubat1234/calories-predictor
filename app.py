import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time

import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

import joblib


TARGET_NAMES = ["calories", "fat", "carbohydrates", "protein"]
MODELS_DIR = Path("saved/app")

class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)

def find_saved_models(models_dir: Path):
    if not models_dir.exists():
        return {}

    model_files = sorted(models_dir.glob("*.joblib"))
    return {model_path.stem: model_path for model_path in model_files}


def humanize_model_name(model_stem: str) -> str:
    token_map = {
        "tfidf": "TF-IDF",
        "count": "Count",
        "manual": "Manual",
        "manual_tfidf": "Manual TF-IDF",
        "word2vec": "Word2Vec",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "elasticnet": "ElasticNet",
        "gd": "GD",
        "elasticnet_gd": "ElasticNet GD",
        "mlp": "MLP",
    }

    if model_stem in token_map:
        return token_map[model_stem]

    parts = model_stem.split("_")
    pretty_parts = []
    for part in parts:
        pretty_parts.append(token_map.get(part, part.capitalize()))
    return " + ".join(pretty_parts)


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def parse_prediction(prediction):
    arr = np.asarray(prediction)
    if arr.ndim == 2 and arr.shape[0] == 1:
        values = arr[0]
    elif arr.ndim == 1:
        values = arr
    else:
        values = arr.reshape(-1)

    if values.size >= len(TARGET_NAMES):
        return values[: len(TARGET_NAMES)]

    # Dla starszych modeli jednowymiarowych wypelniamy tylko calories.
    padded = np.full(len(TARGET_NAMES), np.nan, dtype=np.float64)
    if values.size > 0:
        padded[0] = values[0]
    return padded


def model_requires_servings(model_stem: str, model) -> bool:
    if model_stem.endswith("_servings"):
        return True

    preprocessor = None
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")

    if preprocessor is None or not hasattr(preprocessor, "transformers"):
        return False

    for _, _, columns in preprocessor.transformers:
        if columns == "servings":
            return True
        if isinstance(columns, (list, tuple)) and "servings" in columns:
            return True
    return False


def build_model_input(recipe_text: str, requires_servings: bool, servings: float):
    payload = {"instructions": [recipe_text]}
    if requires_servings:
        payload["servings"] = [float(servings)]
    return pd.DataFrame(payload)

st.set_page_config(page_title="Calories Predictor", page_icon="🍳", layout="centered")

st.title("🍳 AI Calories Predictor")
st.write("Paste your recipe, select a trained model, and predict nutrition values.")

saved_models = find_saved_models(MODELS_DIR)

if not saved_models:
    st.warning("No trained models found in saved/app. Train and save models first.")
    st.stop()

label_to_stem = {
    f"{humanize_model_name(model_stem)} ({model_stem})": model_stem
    for model_stem in saved_models.keys()
}
selected_label = st.selectbox("Available models", list(label_to_stem.keys()))
selected_model = label_to_stem[selected_label]

model_path = saved_models[selected_model]
loaded_model = load_model(model_path.as_posix())
requires_servings = model_requires_servings(selected_model, loaded_model)

recipe_text = st.text_area(
    "Recipe text:", 
    height=200, 
    placeholder="E.g. Mix 200g of butter with two cups of flour and add 3 eggs..."
)

servings_value = 1.0
if requires_servings:
    servings_value = st.number_input(
        "Servings",
        min_value=0.1,
        value=1.0,
        step=0.1,
        help="This model uses servings as an additional numeric feature.",
    )

if st.button("Predict nutrition", type="primary"):
    if recipe_text.strip() == "":
        st.warning("Please enter a recipe first!")
    else:
        with st.spinner("Model is analyzing the text..."):
            try:
                time.sleep(1.2)
                model_input = build_model_input(recipe_text, requires_servings, servings_value)
                prediction = loaded_model.predict(model_input)
                values = parse_prediction(prediction)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.success(f"Done. Model used: {selected_label}")

        output_df = pd.DataFrame(
            {
                "Metric": [
                    "Calories (kcal)",
                    "Fat (g)",
                    "Carbohydrates (g)",
                    "Protein (g)",
                ],
                "Estimated Value": [
                    f"{values[0]:.2f}" if not np.isnan(values[0]) else "N/A",
                    f"{values[1]:.2f}" if not np.isnan(values[1]) else "N/A",
                    f"{values[2]:.2f}" if not np.isnan(values[2]) else "N/A",
                    f"{values[3]:.2f}" if not np.isnan(values[3]) else "N/A",
                ],
            }
        )
        st.table(output_df.set_index("Metric"))