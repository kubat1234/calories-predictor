import numpy as np
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.tokenize import add_ngrams, tokenize


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=10, **kwargs):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.kwargs = kwargs
        self.model = None

    def fit(self, X, y=None):
        tokenized_X = [tokenize(text) for text in X]
        
        self.model = Word2Vec(
            sentences=tokenized_X,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            **self.kwargs
        )
        return self

    def transform(self, X):
        tokenized_X = [tokenize(text) for text in X]
        
        features = []
        for tokens in tokenized_X:
            valid_vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
            
            if valid_vectors:
                features.append(np.mean(valid_vectors, axis=0))
            else:
                features.append(np.zeros(self.vector_size))
                
        return np.array(features)

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
