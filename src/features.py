import numpy as np
import importlib
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from src.tokenize import add_ngrams, tokenize


class ManualFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, keyword_categories=None):
        self.keyword_categories = keyword_categories or {
            "fats_and_sweets": ["butter", "oil", "cream", "cheese", "sugar", "chocolate", "syrup", "mayo", "bacon", "caramel"],
            "proteins": ["chicken", "beef", "pork", "meat", "egg", "fish", "salmon", "steak", "tofu"],
            "carbs": ["flour", "pasta", "rice", "bread", "potato", "dough", "oats"],
            "cooking_methods_heavy": ["fry", "fried", "deep-fry", "roast", "crispy"],
            "cooking_methods_light": ["boil", "steam", "raw", "fresh", "salad"],
            "measurements": ["cup", "tbsp", "tsp", "gram", "oz", "pound", "ml", "liter", "pinch"]
        }

    def _manual_features(self, texts):
        rows = []
        for text in texts:
            tokens = tokenize(text)
            token_count = len(tokens)
            
            if token_count == 0:
                rows.append([0.0] * (2 + len(self.keyword_categories)))
                continue

            digit_token_count = sum(any(char.isdigit() for char in token) for token in tokens)
            
            row_features = [
                float(token_count),
                float(digit_token_count),
            ]

            for category, keywords in self.keyword_categories.items():
                category_count = sum(
                    sum(1 for keyword in keywords if keyword in token) 
                    for token in tokens
                )
                row_features.append(float(category_count))

            rows.append(row_features)

        return np.array(rows, dtype=np.float32)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return csr_matrix(self._manual_features(X))


class ManualTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        keyword_categories=None,
        **tfidf_kwargs,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.keyword_categories = keyword_categories
        self.tfidf_kwargs = tfidf_kwargs
        self.column_transformer = None

    def _to_2d(self, X):
        return np.asarray(list(X), dtype=object).reshape(-1, 1)

    def fit(self, X, y=None):
        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=tokenize,
                        token_pattern=None,
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        **self.tfidf_kwargs,
                    ),
                    0,
                ),
                (
                    "manual",
                    ManualFeatures(keyword_categories=self.keyword_categories),
                    0,
                ),
            ],
            sparse_threshold=1.0,
        )
        self.column_transformer.fit(self._to_2d(X), y)
        return self

    def transform(self, X):
        if self.column_transformer is None:
            raise ValueError("ManualTfidfVectorizer nie jest dopasowany. Najpierw wywołaj fit.")
        return self.column_transformer.transform(self._to_2d(X))

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=10, phrase_threshold=10.0, **kwargs):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.phrase_threshold = phrase_threshold
        self.kwargs = kwargs
        self.model = None
        self.phraser = None

    def fit(self, X, y=None):
        tokenized_X = [tokenize(text) for text in X]
        
        phrases = Phrases(tokenized_X, min_count=self.min_count, threshold=self.phrase_threshold)
        self.phraser = Phraser(phrases)
        
        phrased_X = [self.phraser[tokens] for tokens in tokenized_X]
        
        self.model = Word2Vec(
            sentences=phrased_X,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            **self.kwargs
        )
        return self

    def transform(self, X):
        tokenized_X = [tokenize(text) for text in X]
        
        phrased_X = [self.phraser[tokens] for tokens in tokenized_X]
        
        features = []
        for tokens in phrased_X: 
            valid_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            
            if valid_vectors:
                features.append(np.mean(valid_vectors, axis=0))
            else:
                features.append(np.zeros(self.vector_size))
                
        return np.array(features)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        batch_size=128,
        normalize_embeddings=False,
        show_progress_bar=False,
        device=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        if self.model is not None:
            return self

        try:
            sentence_transformers_module = importlib.import_module("sentence_transformers")
            SentenceTransformer = sentence_transformers_module.SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Brak pakietu sentence-transformers. Zainstaluj: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("SentenceTransformerVectorizer nie jest dopasowany. Najpierw wywołaj fit.")

        embeddings = self.model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def create_vectorizer(
	method,
	ngram_range,
	min_df,
	max_df,
	**kwargs,
):
    if method == "tfidf":
        return TfidfVectorizer(
                tokenizer=tokenize,
                token_pattern=None,
                ngram_range=ngram_range, 
                min_df=min_df, 
                max_df=max_df, 
                **kwargs)

    if method == "count":
        return CountVectorizer(
                tokenizer=tokenize,
            	token_pattern=None,
            	ngram_range=ngram_range,
                min_df=min_df, 
                max_df=max_df, 
                **kwargs)
      
    if method == "word2vec":
        return Word2VecVectorizer(min_count=min_df, **kwargs)

    if method == "manual":
        return ManualTfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            **kwargs,
        )

    if method in {"sentence_transformer", "internet_embedding", "pretrained_embedding"}:
        return SentenceTransformerVectorizer(**kwargs)

    raise ValueError(f"Nieznana metoda wektoryzacji: {method}")

def build_text_features(
	train_texts,
	test_texts,
	method,
	ngram_range,
	min_df,
	max_df,
	vectorizer=None,
	**vectorizer_kwargs,
):
	if vectorizer is None:
		vectorizer = create_vectorizer(
			method=method,
			ngram_range=ngram_range,
			min_df=min_df,
			max_df=max_df,
			**vectorizer_kwargs,
		)

	X_train = vectorizer.fit_transform(train_texts)
	X_test = vectorizer.transform(test_texts)
	return X_train, X_test
