# Calories Predictor

Machine learning project predicting calories, fat, carbohydrates and protein of a recipe based only on its text instructions (and optionally the number of servings). Numbers are removed from the text, so models only see the words.

We compared several text representations (Count, TF-IDF, TF-IDF with hand-crafted keyword features, Word2Vec) and models: Ridge, Elastic Net, Random Forest, LightGBM and neural networks. Some of them (TF-IDF, Elastic Net with gradient descent, a simple neural network, AdaBoost) were implemented from scratch.

Authors: Jakub Trela, Karol Bielaszka. The full report (in Polish) is available in [raport/raport.pdf](raport/raport.pdf).

## Results

R² on the test set (~80k recipes):

| Model | Features | Calories | Fat | Carbs | Protein |
|---|---|:---:|:---:|:---:|:---:|
| LightGBM | TF-IDF + manual features + servings | **0.428** | **0.388** | 0.375 | **0.580** |
| PyTorch NN | embeddings + servings | 0.407 | 0.387 | **0.383** | 0.574 |
| LightGBM | TF-IDF + servings | 0.411 | 0.375 | 0.359 | 0.567 |
| Ridge | TF-IDF + manual features | 0.322 | 0.329 | 0.300 | 0.536 |
| Custom NN | custom TF-IDF | 0.289 | 0.279 | 0.283 | 0.531 |
| Elastic Net GD | custom TF-IDF | 0.244 | 0.224 | 0.185 | 0.415 |
| Random Forest | custom TF-IDF | 0.235 | 0.212 | 0.141 | 0.390 |
| Baseline (mean) | – | 0.000 | 0.000 | 0.000 | 0.000 |

Protein was the easiest target for every model, probably because it comes from a few specific ingredients (meat, eggs, fish), while fat and sugar appear almost everywhere.

## Dataset

[Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews) from Kaggle. To train models, download `recipes.csv` and place it in the `data/` folder.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app lets you paste a recipe, choose one of the trained models from `saved/app/` and get the predicted nutrition values.
