import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("train.csv", header=0)
train, validation = train_test_split(df, train_size = 0.7, random_state = 42)
genres = train["GenreCorrected"].str.split('|').values
multi_binarizer = MultiLabelBinarizer(sparse_output=True)
print(genres)
labels = multi_binarizer.fit_transform(genres)
print(labels.shape)

plot_train = train["Plot"]
plot_validation = validation["Plot"]

vec = TfidfVectorizer(stop_words="english")
train_plot = vec.fit_transform(plot_train)
validation_plot = vec.transform(plot_train)


KNNclf = KNeighborsClassifier(n_neighbors=3)
KNNclf.fit(train_plot, labels)
predictions = KNNclf.predict(validation_plot[0])
print(predictions.shape, predictions[0])
predictions = multi_binarizer.inverse_transform(predictions)



