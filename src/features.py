import numpy as np
import importlib
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csr_matrix

from src.tokenize import add_ngrams, tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class MyTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_df=1.0, min_df=1, max_features=None):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        N = len(X)
        df_counts = defaultdict(int)
        tf_counts = defaultdict(int)
        parsed_docs = []

        for doc in X:
            tokens = tokenize(doc)
            term_counts = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
            for token, count in term_counts.items():
                df_counts[token] += 1
                tf_counts[token] += count
            parsed_docs.append(term_counts)

        max_doc_count = self.max_df if isinstance(self.max_df, int) else self.max_df * N
        min_doc_count = self.min_df if isinstance(self.min_df, int) else self.min_df * N

        valid_terms = {
            t for t, df in df_counts.items()
            if min_doc_count <= df <= max_doc_count
        }

        if self.max_features is not None:
            sorted_terms = sorted(
                [(t, tf_counts[t]) for t in valid_terms],
                key=lambda x: x[1],
                reverse=True
            )
            valid_terms = {t for t, _ in sorted_terms[:self.max_features]}

        self.vocabulary_ = {term: idx for idx, term in enumerate(valid_terms)}
        vocab_size = len(self.vocabulary_)

        row_indices, col_indices, values = [], [], []
        for doc_idx, term_counts in enumerate(parsed_docs):
            for token, count in term_counts.items():
                if token in self.vocabulary_:
                    row_indices.append(doc_idx)
                    col_indices.append(self.vocabulary_[token])
                    values.append(count)

        tf_matrix = sp.csr_matrix(
            (values, (row_indices, col_indices)), 
            shape=(N, vocab_size), 
            dtype=np.float64
        )

        if vocab_size > 0:
            df = np.bincount(tf_matrix.indices, minlength=vocab_size)
            self.idf_ = np.log((1.0 + N) / (1.0 + df)) + 1.0
        else:
            self.idf_ = np.array([])

        return self._apply_idf_and_normalize(tf_matrix)

    def transform(self, X, y=None):
        check_is_fitted(self, ['vocabulary_', 'idf_'])
        
        row_indices, col_indices, values = [], [], []
        vocab_size = len(self.vocabulary_)
        N = len(X)
        
        for doc_idx, doc in enumerate(X):
            tokens = tokenize(doc)
            term_counts = defaultdict(int)
            for token in tokens:
                if token in self.vocabulary_:
                    term_counts[token] += 1
                    
            for token, count in term_counts.items():
                row_indices.append(doc_idx)
                col_indices.append(self.vocabulary_[token])
                values.append(count)

        tf_matrix = sp.csr_matrix(
            (values, (row_indices, col_indices)), 
            shape=(N, vocab_size), 
            dtype=np.float64
        )
        
        return self._apply_idf_and_normalize(tf_matrix)

    def _apply_idf_and_normalize(self, tf_matrix):
        N, vocab_size = tf_matrix.shape
        
        if vocab_size == 0:
            return tf_matrix
            
        idf_diag = sp.diags(self.idf_, offsets=0, shape=(vocab_size, vocab_size))
        X_tfidf = tf_matrix * idf_diag

        norms = sp.linalg.norm(X_tfidf, axis=1)
        norms[norms == 0] = 1.0
        norm_diag = sp.diags(1.0 / norms, offsets=0, shape=(N, N))
        X_tfidf = norm_diag * X_tfidf

        return X_tfidf

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
        return self._manual_features(X)


class ManualTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        max_features=None,          
        keyword_categories=None,
        **tfidf_kwargs,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.keyword_categories = keyword_categories
        self.tfidf_kwargs = tfidf_kwargs
        self.feature_union = None

    def fit(self, X, y=None):
        self.feature_union = FeatureUnion(
            transformer_list=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=tokenize,
                        token_pattern=None,
                        ngram_range=self.ngram_range,
                        min_df=self.min_df,
                        max_df=self.max_df,
                        max_features=self.max_features,
                        **self.tfidf_kwargs,
                    ),
                ),
                (
                    "manual",
                    ManualFeatures(keyword_categories=self.keyword_categories),
                ),
            ]
        )
        self.feature_union.fit(X, y)
        return self

    def transform(self, X):
        if self.feature_union is None:
            raise ValueError("ManualTfidfVectorizer nie jest dopasowany. Najpierw wywołaj fit.")
        return self.feature_union.transform(X)

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
