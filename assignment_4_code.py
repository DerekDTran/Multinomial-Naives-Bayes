# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import keras
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
#import tensorflow_datasets as tfds

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import plot_model, to_categorical

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# read file using sys.argv
#str(sys.argv)
#len(sys.argv)
#script = sys.argv[0]
#fileName = sys.argv[1]

dataset_url = 'https://github.com/calebcaptain1/Twitter-Reviews-Naive-Bayes/archive/main.zip'

datapath = tf.keras.utils.get_file('Twitter-Reviews-Naive-Bayes-main.zip', cache_subdir=os.path.abspath('.'), origin = dataset_url, extract = True)

data = os.path.abspath('Twitter-Reviews-Naive-Bayes-main')
tweets = '/Tweets.csv'


data
path = data + tweets
print(path)
twitter_data = pd.read_csv(path)

# read 3 columns into a dataframe
twitter = twitter_data[['airline_sentiment', 'airline', 'text']]

# perform test pre-processing
    # convert text to lowercase
    twitter['text'] = twitter['text'].str.lower()
    twitter['text']
    
    # transform the text data using CountVectorizer and TfidfTransformer
    count_vector = CountVectorizer()
    count = count_vector.fit_transform(twitter['text'])
    
    transform = TfidfTransformer().fit(count)
    count = transform.transform(count)
    
    # convert "airline_sentiment" from categorical to numerical values; use LabelEncoder class in sklearn
    encode = LabelEncoder()
    encode.fit(twitter['airline_sentiment'])
    encode.transform(twitter['airline_sentiment'])
    twitter['airline_sentiment'] = encode.transform(twitter['airline_sentiment'])
    
# split the data into two parts: train and test; use train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(count, twitter['airline_sentiment'], test_size = 0.1)
X_train.shape
Y_train.shape

# build a multinomial Naive Bayes model using train
model = MultinomialNB()
model.fit(X_train, Y_train)

# apply model to test and output accuracy
predict = model.predict(X_test)
model.score(X_test, Y_test)

confusion_matrix(Y_test, predict)
print(classification_report(Y_test, predict, digits = 4))
np.mean(Y_test == predict)

# repeat 5 times with different parameter choices and output parameters and accuracy
results = pd.DataFrame(columns = ['alpha', 'prior', 'accuracy'])

alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
prior = [True, False]

for p in prior:
    for a in alpha:
        model = MultinomialNB(alpha = a, fit_prior = p)
        model.fit(X_train, Y_train)
        predict = model.predict(X_test)
        model.score(X_test, Y_test)
        confusion_matrix(Y_test, predict)
        np.mean(Y_test == predict)
        new_element = {'alpha':a, 'prior':p, 'accuracy':np.mean(Y_test == predict)}
        results = results.append(new_element, ignore_index = True)
print(results)

# using the numeric value of airline_sentiment, output the average sentiment of each airline
airline_list = twitter['airline'].unique()
airline_list

twitter['airline'].value_counts(normalize=True)
twitter.groupby('airline').mean()[['airline_sentiment']]
