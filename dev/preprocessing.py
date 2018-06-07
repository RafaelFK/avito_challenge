import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin # Para definição de transformadores personalizados
from sklearn.preprocessing import OneHotEncoder, Imputer, FunctionTransformer
# from category_encoders import OrdinalEncoder
from future_encoders import OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import time
from IPython.display import display

import os.path

class PreProcessor():
	def loadData(self, filename, update=False):
		self.filename = filename

		start = time.time()

		processed_filename = os.path.splitext(self.filename)[0] + '_processed.csv'

		if os.path.isfile(processed_filename) and not update:
			df_train = pd.read_csv(processed_filename)
			df_train = df_train.apply(pd.to_numeric)
			X_train_processed, y_train = df_train[df_train.columns.difference(['deal_probability'])].copy(), df_train[['deal_probability']].copy()

			end = time.time()
			print('Elapsed time:', end - start)

			return X_train_processed, y_train
		else:
			converters = {key: lambda x: np.nan if not x else str(x) for key in ('param_1', 'param_2', 'param_3')}

			df_train = pd.read_csv(
			    self.filename,
			    parse_dates=['activation_date'],
			    converters=converters                  
			)

			# X_test = pd.read_csv(
			#     './data/test.csv',
			#     parse_dates=['activation_date'],
			#     converters=converters                  
			# )

			X_train, y_train = df_train[df_train.columns.difference(['deal_probability'])].copy(), df_train[['deal_probability']].copy()

			# O atributo as_df determina se o resultado do transformador é um DataFrame. Isso é necessário caso queiramos efetuar
			# seleções posteriores sobre o resultado desse transformador
			class DataFrameSelector(BaseEstimator, TransformerMixin):
			    def __init__(self, attribute_names, as_ndarray=False):
			        self.attribute_names = attribute_names
			        self.as_ndarray = as_ndarray
			        
			    def fit(self, X, y=None):
			        return self
			    
			    def transform(self, X):
			        return X[self.attribute_names] if not self.as_ndarray else X[self.attribute_names].values
			    
			class ToDataFrameTransformer(BaseEstimator, TransformerMixin):
			    def __init__(self, columns=None):
			        self.columns = columns
			        
			    def fit(self, X, y=None):
			        return self
			    
			    def transform(self, X):
			        return pd.DataFrame(X, columns=self.columns)

			features_selector = DataFrameSelector(X_train.columns.difference(['item_id', 'user_id', 'image']))

			# Essa transformação espera uma única série de timestamps e retorna um DataFrame
			class TimestampTransformer(BaseEstimator, TransformerMixin):
			    def fit(self, X, y=None):
			        return self
			    
			    def transform(self, X):
			        return pd.DataFrame(
			            np.c_[
			                X.dt.month,
			                X.dt.day,
			                X.dt.weekday
			            ],
			            columns=['month', 'day', 'weekday']
			        )

			process_date = Pipeline([
			    ('date_series_selector', DataFrameSelector('activation_date')),
			    ('date_features_transformer', TimestampTransformer())
			])

			price_imputer = Imputer(strategy='mean')

			process_price = Pipeline([
			    ('price_selector', DataFrameSelector(['price'])),
			    ('price_imputer', price_imputer),
			    ('to_data_frame', ToDataFrameTransformer(['price']))
			])

			image_imputer = Imputer(strategy='most_frequent')

			process_image_class = Pipeline([
			    ('image_selector', DataFrameSelector(['image_top_1'])),
			    ('image_imputer', image_imputer),
			    ('to_data_frame', ToDataFrameTransformer(['image_top_1']))
			])

			class CategoricalImputer(BaseEstimator, TransformerMixin):
			    def fit(self, X, y=None):
			        self.most_frequent = pd.Series(X.values.ravel()).value_counts().index[0]
			        return self
			        
			    def transform(self, X):
			        return pd.DataFrame(pd.Series(X.values.ravel()).fillna(self.most_frequent))

			impute_cat = Pipeline([
			    ('cat_imputer', FeatureUnion([
			        ('region_imputer', Pipeline([
			            ('region_selector', DataFrameSelector('region')),
			            ('region_imputer', CategoricalImputer())
			        ])),
			        ('city_imputer', Pipeline([
			            ('city_selector', DataFrameSelector('city')),
			            ('city_imputer', CategoricalImputer())
			        ])),
			        ('parent_category_name_imputer', Pipeline([
			            ('parent_category_name_selector', DataFrameSelector('parent_category_name')),
			            ('parent_category_name_imputer', CategoricalImputer())
			        ])),
			        ('category_name_imputer', Pipeline([
			            ('category_name_selector', DataFrameSelector('category_name')),
			            ('category_name_imputer', CategoricalImputer())
			        ])),
			        ('param_1_imputer', Pipeline([
			            ('param_1_selector', DataFrameSelector('param_1')),
			            ('param_1_imputer', CategoricalImputer())
			        ])),
			        ('param_2_imputer', Pipeline([
			            ('param_2_selector', DataFrameSelector('param_2')),
			            ('param_2_imputer', CategoricalImputer())
			        ])),
			        ('param_3_imputer', Pipeline([
			            ('param_3_selector', DataFrameSelector('param_3')),
			            ('param_3_imputer', CategoricalImputer())
			        ])),
			        ('user_type_imputer', Pipeline([
			            ('user_type_selector', DataFrameSelector('user_type')),
			            ('user_type_imputer', CategoricalImputer())
			        ])),   
			    ])),
			    ('to_data_frame', ToDataFrameTransformer([
			        'region',
			        'city',
			        'parent_category_name',
			        'category_name',
			        'param_1',
			        'param_2',
			        'param_3',
			        'user_type'
			    ]))
			])

			class TextImputer(BaseEstimator, TransformerMixin):
			    def fit(self, X, y=None):
			        return self
			    
			    def transform(self, X):
			        return pd.DataFrame(X.fillna(''))

			process_description = Pipeline([
			    ('description_selector', DataFrameSelector('description')),
			    ('imputer', TextImputer())
			])

			encode_cat = Pipeline([
			    ('cat_selector', DataFrameSelector([
			        'region',
			        'city',
			        'parent_category_name',
			        'category_name',
			        'param_1',
			        'param_2',
			        'param_3',
			        'user_type'
			    ])),
			    ('ordinal_encoder', OrdinalEncoder(dtype=np.int64)),
			    ('to_data_frame', ToDataFrameTransformer([
			        'region',
			        'city',
			        'parent_category_name',
			        'category_name',
			        'param_1',
			        'param_2',
			        'param_3',
			        'user_type'
			    ]))
			])

			vect = TfidfVectorizer(sublinear_tf=True)

			encode_title = Pipeline([
			    ('title_selector', DataFrameSelector(['title'])),
			    ('ravel_feature', FunctionTransformer(lambda f: f.values.ravel(), validate=False)),
			    ('text_vectorizer', vect),
			    ('dim_reduction', TruncatedSVD(random_state=1))
			])

			encode_description = Pipeline([
			    ('description_selector', DataFrameSelector('description')),
			    ('description_imputer', TextImputer()),
			    ('ravel_feature', FunctionTransformer(lambda f: f.values.ravel(), validate=False)),
			    ('text_vectorizer', vect),
			    ('dim_reduction', TruncatedSVD(random_state=1))
			])

			encode_text = Pipeline([
			    ('process_text', FeatureUnion([
			        ('process_title', encode_title),
			        ('process_description', encode_description)
			    ])),
			    ('to_data_frame', ToDataFrameTransformer(['title_svd_1', 'title_svd_2', 'description_svd_1', 'description_svd_2']))
			])

			preprocess_transformer = Pipeline([
			    ('working_feature_selector', features_selector),
			    ('date_transf', FeatureUnion([
			        ('other_features', DataFrameSelector(['category_name', 'city', 'description',
			       'image_top_1', 'item_seq_number', 'param_1', 'param_2', 'param_3',
			       'parent_category_name', 'price', 'region', 'title', 'user_type'])),
			        ('process_date', process_date),
			    ])),
			    ('to_data_frame_1', ToDataFrameTransformer([['category_name', 'city', 'description',
			       'image_top_1', 'item_seq_number', 'param_1', 'param_2', 'param_3',
			       'parent_category_name', 'price', 'region', 'title', 'user_type', 'month', 'day', 'weekday']])),
			    ('imputers', FeatureUnion([
			        ('other_features', DataFrameSelector(['item_seq_number', 'title', 'month', 'day', 'weekday'])),
			        ('impute_price', process_price),
			        ('impute_image', process_image_class),
			        ('impute_cat', impute_cat),
			        ('impute_description', process_description)
			    ])),
			    ('to_data_frame_2', ToDataFrameTransformer([
			        'item_seq_number',
			        'title',
			        'month',
			        'day',
			        'weekday',
			        'price',
			        'image_top_1',
			        'region',
			        'city',
			        'parent_category_name',
			        'category_name',
			        'param_1',
			        'param_2',
			        'param_3',
			        'user_type',
			        'description'
			    ])),
			    ('encoders', FeatureUnion([
			        ('other_features', DataFrameSelector(['item_seq_number', 'month', 'day', 'weekday', 'price', 'image_top_1'])),
			        ('encode_cat', encode_cat),
			        ('encode_text', encode_text)
			    ])),
			    ('to_data_frame_3', ToDataFrameTransformer([
			        'item_seq_number',
			        'month',
			        'day',
			        'weekday',
			        'price',
			        'image_top_1',
			        'region',
			        'city',
			        'parent_category_name',
			        'category_name',
			        'param_1',
			        'param_2',
			        'param_3',
			        'user_type',
			        'title_svd_1',
			        'title_svd_2',
			        'description_svd_1',
			        'description_svd_2'
			    ]))
			])

			X_train_processed = preprocess_transformer.fit_transform(X_train)

			df_processed = X_train_processed.assign(deal_probability=y_train)
			df_processed.to_csv(processed_filename)

			end = time.time()
			print('Elapsed time:', end - start)

			return X_train_processed, y_train