import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.features import ManualTfidfVectorizer, Word2VecVectorizer, MyTfidfVectorizer
from src.models.ElasticNetGDRegressor import ElasticNetGDRegressor
from src.models.CustomAdaBoostRegressor import CustomAdaBoostRegressor
from src.models.CustomNeuralNetworkRegressor import CustomNeuralNetworkRegressor
from src.models.DumbRegressor import DumbRegressor
from src.tokenize import tokenize

from config_constants import RANDOM_SEED, TFIDF_MAX_FEATURES


class ToDenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sp.issparse(X):
            return X.toarray()
        return np.asarray(X)


def build_models():
    def make_lgbm():
        return TransformedTargetRegressor(
            regressor=lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=70,
                random_state=RANDOM_SEED, 
                verbose=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return {
        "ridge": {
            "factory": lambda: TransformedTargetRegressor(
                regressor=Ridge(alpha=1.0),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "requires_dense": False,
            "supports_multioutput": True,
            "uses_servings": False,
        },
        "ridge_servings": {
            "factory": lambda: TransformedTargetRegressor(
                regressor=Ridge(alpha=1.0),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "requires_dense": False,
            "supports_multioutput": True,
            "uses_servings": True,
        },
        "elasticnet_gd": {
            "factory": lambda: TransformedTargetRegressor(
                regressor=ElasticNetGDRegressor(
                    learning_rate=0.1, 
                    max_iter=1000,   
                    l1=0.4, 
                    l2=0.15,
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "requires_dense": False,
            "supports_multioutput": False,
            "uses_servings": False,
        },
        "random_forest": {
            "factory": lambda: RandomForestRegressor(
                n_estimators=10,
                max_depth=15,
                random_state=RANDOM_SEED,
                n_jobs=24,
            ),
            "requires_dense": True,
            "supports_multioutput": True,
            "uses_servings": False,
        },
        "dumb": {
            "factory": lambda: DumbRegressor(),
            "requires_dense": True,
            "supports_multioutput": False,
            "uses_servings": False,
        },
        "lgbm": {
            "factory": make_lgbm,
            "requires_dense": False,
            "supports_multioutput": False, 
            "uses_servings": False,
        },
        "lgbm_servings": {
            "factory": make_lgbm,
            "requires_dense": False,
            "supports_multioutput": False,
            "uses_servings": True,
        },
        "custom_adaboost": {
            "factory": lambda: TransformedTargetRegressor(
                regressor=CustomAdaBoostRegressor(
                    n_estimators=40,       
                    max_depth=4, 
                    min_samples_split=4, 
                    min_samples_leaf=2, 
                    max_features=None 
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "requires_dense": True,
            "supports_multioutput": False,
            "uses_servings": False,
        },
        "custom_nn": {
            "factory": lambda: TransformedTargetRegressor(
                regressor=CustomNeuralNetworkRegressor(
                    layer_sizes=[64, 32, 1],
                    epochs=40,
                    learning_rate=0.01,
                    batch_size=64,
                    random_state=RANDOM_SEED
                ),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
            "requires_dense": True,
            "supports_multioutput": False,
            "uses_servings": False,
        }
    }


def build_vectorizers(max_features=None):
    if max_features is None:
        max_features = TFIDF_MAX_FEATURES

    return {
        "count": lambda: CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=max_features,
        ),
        "tfidf": lambda: TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=max_features,
        ),
        "manual_tfidf": lambda: ManualTfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            max_features=max_features,
        ),
        "word2vec": lambda: Word2VecVectorizer(
            vector_size=64,
            window=5,
            min_count=2,
            epochs=5,
        ),
        "my_tfidf": lambda: MyTfidfVectorizer(
            min_df=3,
            max_df=0.8,
            max_features=max_features,
        ),
    }


def build_pipeline(vectorizer_factory, model_info, is_multioutput=True):
    model = model_info["factory"]()
    
    if is_multioutput and not model_info.get("supports_multioutput", True):
        model = MultiOutputRegressor(model)

    uses_servings = model_info.get("uses_servings", False)
    requires_dense = model_info.get("requires_dense", False)

    if uses_servings:
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', vectorizer_factory(), 'instructions'),
                ('num', 'passthrough', ['servings'])
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', vectorizer_factory(), 'instructions')
            ],
            remainder='drop'
        )

    steps = [("preprocessor", preprocessor)]
    
    if requires_dense:
        steps.append(("to_dense", ToDenseTransformer()))
        
    steps.append(("model", model))
    
    return Pipeline(steps)